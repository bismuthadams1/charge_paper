"""Script used to generate the large scale comparisons between the charge models


"""


from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from chargecraft.storage.db import DBMoleculePropRecord, DBConformerPropRecord
import pyarrow.parquet
from sqlalchemy.orm import Session, sessionmaker, contains_eager, joinedload
from openff.toolkit.topology import Molecule
from openff.units import unit
from collections import defaultdict
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
from more_itertools import batched
from rdkit.Chem import rdmolfiles
from rdkit import Chem
from MultipoleNet import load_model, build_graph_batched, D_Q
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from ChargeAPI.API_infrastructure.esp_request.module_version_esp import handle_esp_request
from tqdm import tqdm
from typing import Iterator
from naglmbis.models import load_charge_model

import traceback
import json
import tempfile
import numpy as np
import rdkit
import pyarrow
import hashlib
import os

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

AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge
HA_TO_KCAL_P_MOL =  627.509391  # Hartrees to kilocalories per mole

charge_model_esp= 'nagl-water-charge-dipole-esp-wb-default'
charge_model_charge = "nagl-water-charge-wb"
charge_model_dipole =  "nagl-water-charge-dipole-wb"

gas_charge_model = load_charge_model(charge_model=charge_model_charge)
gas_charge_dipole_model = load_charge_model(charge_model=charge_model_dipole)
gas_charge_dipole_esp_model = load_charge_model(charge_model_esp)

def make_openff_molecule(mapped_smiles: str, coordinates: unit.Quantity) -> Molecule:
    
    molecule = Molecule.from_mapped_smiles(
        mapped_smiles=mapped_smiles,
        allow_undefined_stereo=True
    )
    molecule.add_conformer(coordinates=coordinates)
    return molecule


def build_mol(openff_molecule: Molecule) -> str:
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
    
def process_molecule(parquet: dict, models: dict):
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
        charge_models_data[f'{model_name}_dipoles'] = predicted_dipole.magnitude.tolist()
        
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

    print(batch_dict)
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

def process_and_write_batch(batch_models, schema, writer):
    # with ProcessPoolExecutor(max_workers=40) as pool:
        # Submit jobs to process the models in parallel
        # jobs = [pool.submit(process_molecule, model) for model in batch_models]
        # results_batch = []
        # for future in tqdm(as_completed(jobs), total=len(jobs), desc='Processing molecules'):
        #     try:
        #         result = future.result()
        #         results_batch.append(result)
        #     except Exception as e:
        #         print(f'Failure of job due to {e}')
        #         print(traceback.format_exc())
                # continue  # Skip if the molecule was skipped or had no results
    results_batch = []
    for model in tqdm(batch_models, total=len(batch_models), desc='Processing molecules'):
        results_batch.append(process_molecule(model))
    rec_batch = pyarrow.RecordBatch.from_pylist(results_batch, schema=schema)
    writer.write_batch(rec_batch)
    

def main(output: str):

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
    ])

    # batch_count = 2
    batch_size = 20000
    batch_models = []
    parquet_location = '/mnt/storage/nobackup/nca121/test_data_sets/water/testing_water_esp.parquet'
    parquet_table = pyarrow.parquet.read_table(parquet_location)
    with pyarrow.parquet.ParquetWriter(where=output, schema=schema, compression='snappy') as writer:
        batch_count = 0  # Initialize the batch counter
        batch_models = []
        
        for item in tqdm(parquet_table.to_pylist(),total= parquet_table.shape[0], desc='processing table'):
            # # Skip charged species as Riniker cannot accept them
            # if "+" in molecule_smiles or "-" in molecule_smiles or "Br" in molecule_smiles or "P" in molecule_smiles:
            #     continue
            batch_models.append(item)
            if len(batch_models) >= batch_size:
                # Process the batch
                process_and_write_batch(batch_models, schema, writer)
                batch_models = []
                batch_count += 1  # Increment the batch counter
                # if batch_count >= 1:
                #     break  # Exit the loop after processing 4 batches

        # Optionally process any remaining models if you haven't reached 4 batches
        if batch_models: #and batch_count < 1:
            process_and_write_batch(batch_models, schema, writer)

        
if __name__ == "__main__":
    main(output='./test_models.parquet')