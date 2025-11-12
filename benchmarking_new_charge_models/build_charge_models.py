"""Script used to test the new charge models across the train, test, validate sets

"""


from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from chargecraft.storage.db import DBMoleculePropRecord, DBConformerPropRecord
import pyarrow.parquet as pq
from sqlalchemy.orm import Session, sessionmaker, contains_eager, joinedload
from openff.toolkit.topology import Molecule
from openff.units import unit
from collections import defaultdict
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
from more_itertools import batched
from rdkit.Chem import rdmolfiles
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm
from typing import Sequence
from naglmbis.models import load_charge_model
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
import logging
import gc

import traceback
import json
import tempfile
import numpy as np
import pandas as pd
import rdkit
import pyarrow
import hashlib
import os
import polars as pl

from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.grids import GridSettingsType, GridGenerator
from openff.recharge.grids import LatticeGridSettings, MSKGridSettings
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.recharge.esp import ESPSettings
from openff.recharge.charges.resp.solvers import IterativeSolver
import builtins
import psutil

process = psutil.Process(os.getpid())

def log_memory_usage(msg=""):
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)  # Resident Set Size in MB
    logging.info(f"{msg} - RSS memory usage: {rss_mb:.2f} MB")

def read_smiles_from_json(file_path):
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    data = data.keys()
    print(data)
    smiles = [entry for entry in data]
    return smiles

def print(*args):
    builtins.print(*args, sep=' ', end='\n', file=None, flush=True)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='build_charge_models.log', level=logging.INFO)

AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge
HA_TO_KCAL_P_MOL =  627.509391  # Hartrees to kilocalories per mole
TRAIN_SMILES_JSON = 'maxmin-train2.json'

charge_model_esp= 'nagl-gas-charge-dipole-esp-wb-default'
charge_model_charge = "nagl-gas-charge-wb"
charge_model_dipole =  "nagl-gas-charge-dipole-wb"

charge_model_water_esp = 'nagl-water-charge-dipole-esp-wb-default'
charge_model_water_charge = 'nagl-water-charge-wb'
charge_model_water_dipole = 'nagl-water-charge-dipole-wb'

gas_charge_model = load_charge_model(charge_model=charge_model_charge)
gas_charge_dipole_model = load_charge_model(charge_model=charge_model_dipole)
gas_charge_dipole_esp_model = load_charge_model(charge_model_esp)

water_charge_model = load_charge_model(charge_model=charge_model_water_charge)
water_charge_dipole_model = load_charge_model(charge_model=charge_model_water_dipole)
water_charge_dipole_esp_model = load_charge_model(charge_model = charge_model_water_esp)

#depending on which charge model you want to use, uncomment this part. 
models = {
    # "charge_model": gas_charge_model,
    # "dipole_model": gas_charge_dipole_model,
    # "esp_model": gas_charge_dipole_esp_model,
    "charge_model": water_charge_model,
    "dipole_model": water_charge_dipole_model,
    "esp_model": water_charge_dipole_esp_model,
}

FP_RADIUS = 2
FP_NBITS = 2048

def smiles_to_fps(smiles: Sequence[str], radius=FP_RADIUS, nbits=FP_NBITS):
    fps, valid_idx = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits))
        valid_idx.append(i)
    return fps, valid_idx

train_smiles = list(set(read_smiles_from_json(TRAIN_SMILES_JSON)))
fps_train, kept_train = smiles_to_fps(train_smiles)

def calculate_max_tanimoto_similarity(
    train_smiles: Sequence[str],
    test_smiles: Sequence[str],
    radius: int = FP_RADIUS,
    nbits: int = FP_NBITS,
):
    """
    For each test molecule:
      - Find its most similar training molecule
      - Record any with similarity > threshold in a CSV
    """
    fps_test, kept_test   = smiles_to_fps(test_smiles,  radius, nbits)

    max_sims = np.zeros(len(fps_test), dtype=np.float32)
    best_train_idx = np.zeros(len(fps_test), dtype=int)
    high_pairs = []

    for i, fp_test in enumerate(fps_test):
        sims = np.asarray(BulkTanimotoSimilarity(fp_test, fps_train), dtype=np.float32)
        j_max = int(np.argmax(sims))
        sim_max = float(sims[j_max])
    
    return sim_max

def make_openff_molecule(mapped_smiles: str, coordinates: unit.Quantity) -> Molecule:
    """Make an openff_molecule from smiles and coordinates
    
    Parameters
    ----------
    mapped_smiles: str
        mapped smiles of the molecule
        
    coordinates: unit.Quantity
        coordinates of the molecule
        
    Returns
    -------
    Molecule
        openff molecule 

    """
    molecule = Molecule.from_mapped_smiles(
        mapped_smiles=mapped_smiles,
        allow_undefined_stereo=True
    )
    molecule.add_conformer(coordinates=coordinates)
    return molecule


def build_mol(openff_molecule: Molecule) -> str:
    """Build a molblock from an openff_molecule
    
    Parameters
    ----------
    openff_molecule
        openff molecule 
    
    """
    return rdmolfiles.MolToMolBlock(openff_molecule.to_rdkit())


def calculate_esp_monopole_au(
    grid_coordinates: unit.Quantity,  # N x 3
    atom_coordinates: unit.Quantity,  # M x 3
    charges: unit.Quantity,  # M
    ):
    """Generate the esp from the on atom monopole
    
    Parameters
    ----------
    grid_coordinates: unit.Quantity
        grid on which to build the esp on 

    atom_coordinates: unit.Quantity
        coordinates of atoms to build the esp  
    
    charges: unit.Quantity
        monopole or charges

    Returns
    -------
    monopole_esp: unit.Quantity
        monopole esp
    """
    #prefactor  
    ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)

    if isinstance(charges, unit.Quantity):
       charges = charges.flatten()
    else:
        charges = np.array(charges).flatten() * unit.e
    #Ensure everything is in AU and correct dimensions
    grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
    atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr
    #displacement and distance
    displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N x M x 3 B
    distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M
    inv_distance = 1 / distance  #N, M

    esp = ke*np.sum(inv_distance * charges[None,:], axis=1)  # (N,M)*(1,M) -> (N,M) numpy broadcasts all charges. Over all atoms  =  Sum over M (atoms), resulting shape: (N,) charges broadcast over each N
    
    return esp.to(AU_ESP)
        
def calculate_dipole_magnitude(charges: unit.Quantity,
                               conformer: unit.Quantity) -> float:
    """Calculate dipole magnitude
    
    Parameters
    ----------
    charges: np.ndarray
    
    conformer: np.ndarray
    
    Returns
    -------
    float
        dipole magnitude (no units)
    
    """
    reshaped_charges = np.reshape(charges,(-1,1))
    dipole_vector = np.sum(conformer.to(unit.bohr) * reshaped_charges,axis=0)
    dipole_magnitude = np.linalg.norm(dipole_vector)

    return dipole_magnitude.m

def make_hash(openff_mol: Molecule) -> str:
    """Make a molblock for the purposes of batching
    
    Parameters
    ----------
    
    openff_mol: Molecule
        open force field molecule with conformers embedded
    
    Returns
    -------
    str 
        hash output unique to each conformer
    
    """

    conformer =  openff_mol.conformers[0].m.flatten().tolist()
    hash_input = openff_mol.to_smiles() + ''.join(f"{c:.6f}" for c in conformer)
    
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
def process_molecule(parquet: dict, models: dict, skip_smiles=set()) -> dict:
    """Process molecules with multiple charge models
    
    Parameters
    ----------
    parquet : dict
        Record containing QM info.
    models : dict
        Dictionary of loaded charge models:
            - "charge_model"
            - "dipole_model"
            - "esp_model"
    
    Returns
    -------
    batch_dict : dict
        Dictionary containing all the charge info for multiple charge models.
    """
    batch_dict = {}
    coordinates = (parquet['conformation'] * unit.bohr).reshape((-1, 3))
    mapped_smiles = parquet['smiles']
    # canonicalize the smiles to check if in skip list
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is not None:
        can_smi = Chem.MolToSmiles(mol, canonical=True)
        if can_smi in skip_smiles:
            logger.info(f"Skipping molecule {can_smi} as it is in the skip list.")
            return None

    openff_mol: Molecule = make_openff_molecule(
        mapped_smiles=mapped_smiles,
        coordinates=coordinates
    )
    rdkit_mol = openff_mol.to_rdkit()
    batch_dict['molecule'] = mapped_smiles
    batch_dict['geometry'] = coordinates.m.flatten().tolist()
    batch_dict['molblock'] = rdkit.Chem.rdmolfiles.MolToMolBlock(rdkit_mol)
    batch_dict['mol_id'] = make_hash(openff_mol)
    
    # ------ Charges and Dipoles for each model -------#
    charge_models_data = {}
    for model_name, model in models.items():
        predicted_charges = model.compute_properties(rdkit_mol)["mbis-charges"].detach().numpy().flatten()
        charge_models_data[f'{model_name}_charges'] = predicted_charges.tolist()

        # Calculate dipoles
        predicted_dipole = calculate_dipole_magnitude(
            charges=predicted_charges * unit.e,
            conformer=coordinates
        )
        charge_models_data[f'{model_name}_dipoles'] = predicted_dipole.tolist()
        
        # Calculate ESP and RMSE
        grid_coordinates = (parquet['grid'] * unit.bohr).reshape(-1, 3)
        predicted_esp = calculate_esp_monopole_au(
            grid_coordinates=grid_coordinates,
            atom_coordinates=coordinates,
            charges=predicted_charges * unit.e
        )
        qm_esp = parquet['esp'] * unit.hartree / unit.e
        esp_rms = (((predicted_esp - qm_esp) ** 2).mean() ** 0.5).magnitude
        charge_models_data[f'{model_name}_esp'] = predicted_esp.m.flatten().tolist()
        charge_models_data[f'{model_name}_esp_rmse'] = esp_rms * HA_TO_KCAL_P_MOL

    # ------ QM and MBIS properties -------#
    batch_dict['mbis_charges'] = parquet['mbis-charges']
    batch_dict['qm_dipoles_magnitude'] = np.linalg.norm(parquet['dipole']).tolist()
    batch_dict['mbis_dipoles_magnitude'] = np.linalg.norm(parquet['mbis-dipoles']).tolist()
    batch_dict.update(charge_models_data)

    # ------ Tanimoto similarity to training set -------#
    batch_dict['tanimoto_similarity_to_train'] = calculate_max_tanimoto_similarity(
        smiles=[batch_dict['molecule']],
        fps_train=fps_train
    )

    return batch_dict

def create_mol_block_tmp_file(pylist: list[dict], temp_dir: str) -> None:
    """Create a tmp file with all the molblocks
    
    Parameters
    ----------
    pylist: list[dict]
        dictionary of the pylist results
    
    """
    json_dict = {}
    for item in pylist:
        json_dict[item['mol_id']] = (item['molblock'],item['grid'])
    json_file = os.path.join(temp_dir, 'molblocks.json')
    json.dump(json_dict, open(json_file, "w"))
    
    return json_file

def process_and_write_batch(batch_models, schema, writer, skip_smiles=set()):
    """Write a batch to the parquet.
    
    Parameters
    ----------
    batch_models: list[dict]
        list of dictionaries to be written to parquet
    
    Returns
    -------
    schema: pyarrow.schema
        data formats going into the parquet. 
        
    writer: pyarrow.parquet.ParquetWriter
        writer to write the parquet file to
    
    """
    results_batch = []

    for model in tqdm(batch_models, total=len(batch_models[0]), desc='Processing molecules'):
        # skip molecules in the skip smiles set
        if processed:= process_molecule(model, models=models, skip_smiles=skip_smiles):
            results_batch.append(processed)
        
    rec_batch = pyarrow.RecordBatch.from_pylist(results_batch, schema=schema)
    writer.write_batch(rec_batch)
    

def main(output: str, data: str):

    schema = pyarrow.schema([
        ('mbis_charges', pyarrow.list_(pyarrow.float64())),
        ('predicted_charges', pyarrow.list_(pyarrow.float64())),
        ('molecule', pyarrow.string()),
        ('geometry', pyarrow.list_(pyarrow.float64())),
        ('molblock', pyarrow.string()),
        ('mol_id', pyarrow.string()),
        ('qm_dipoles_magnitude', pyarrow.float64()),
        ('mbis_dipoles_magnitude', pyarrow.float64()),
        ('predicted_dipoles', pyarrow.float64()),
        ('qm_esp', pyarrow.list_(pyarrow.float64())),
        ('predicted_esp_rmse', pyarrow.float64()),
        ('predicted_esp', pyarrow.list_(pyarrow.float64())),
        ('charge_model_charges', pyarrow.list_(pyarrow.float64())),
        ('charge_model_dipoles', pyarrow.float64()),
        ('charge_model_esp', pyarrow.list_(pyarrow.float64())),
        ('charge_model_esp_rmse', pyarrow.float64()),
        ('dipole_model_charges', pyarrow.list_(pyarrow.float64())),
        ('dipole_model_dipoles', pyarrow.float64()),
        ('dipole_model_esp', pyarrow.list_(pyarrow.float64())),
        ('dipole_model_esp_rmse', pyarrow.float64()),
        ('esp_model_charges', pyarrow.list_(pyarrow.float64())),
        ('esp_model_dipoles', pyarrow.float64()),
        ('esp_model_esp', pyarrow.list_(pyarrow.float64())),
        ('esp_model_esp_rmse', pyarrow.float64()),
        ('tanimoto_similarity_to_train', pyarrow.float64()),
    ])

    batch_size = 500
    batch_models = []
    parquet_location = data
    parquet_file = pq.ParquetFile(parquet_location)
    total_rows = parquet_file.metadata.num_rows

    SIMILAIR_PAIRS_FP = './similar_pairs.csv'
    with open(SIMILAIR_PAIRS_FP, 'r') as file:
        df = pd.read_csv(file)
        similar_pairs = df['test_smiles'].tolist()
    DUPLICATES_FP = './duplicated_smiles.csv'
    with open(DUPLICATES_FP, 'r') as file:
        df = pd.read_csv(file)
        duplicated_smiles = df['test_smiles'].tolist()

    SKIP_SMILES = set(similar_pairs).union(set(duplicated_smiles))

    #canonicalize the skip smiles
    canonical_skip_smiles = set()
    for smi in SKIP_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            can_smi = Chem.MolToSmiles(mol, canonical=True)
            canonical_skip_smiles.add(can_smi)
    SKIP_SMILES = canonical_skip_smiles
    logging.info(f"Number of SMILES to skip: {len(SKIP_SMILES)}")

    with pyarrow.parquet.ParquetWriter(where=output, schema=schema, compression='snappy') as writer:
        batch_models = []
        for item in tqdm(parquet_file.iter_batches(batch_size=batch_size), desc='Processing table'):
            logging.info(f'{log_memory_usage("Before to_pylist()")}')
            batch_models.append(converted:=item.to_pylist())
            logging.info(f'{log_memory_usage("After to_pylist()")}')
            logger.info(f"{len(converted)}")
            if len(converted) >= batch_size:
                logger.info('processing batch')
                process_and_write_batch(converted, schema, writer, skip_smiles=SKIP_SMILES)
                del batch_models
                del converted
                gc.collect()
                batch_models = []

        # Optionally process any remaining models if you haven't reached 4 batches
        if batch_models: #and batch_count < 1:
            process_and_write_batch(converted, schema, writer)

        
if __name__ == "__main__":
    #data_path = './testing_water_esp.parquet'
    # data_path = '/scratch/users/k2584788/mbis_splits/test/testing_gas_esp.parquet'
    data_path = '/scratch/users/k2584788/mbis_splits/testing_water_esp.parquet'
    main(output='./test_water_esp_model.parquet', data=data_path)