
from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from chargecraft.storage.db import DBMoleculePropRecord, DBConformerPropRecord
from sqlalchemy.orm import Session, sessionmaker, contains_eager, joinedload
from openff.toolkit.topology import Molecule
from chargecraft.inputsetup.SmilesInputs import ReadInput
from openff.units import unit
from collections import defaultdict
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
from more_itertools import batched
from rdkit.Chem import rdmolfiles
import gc
from rdkit import Chem
# from MultipoleNet import load_model, build_graph_batched, D_Q
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from ChargeAPI.API_infrastructure.esp_request.module_version_esp import handle_esp_request
from tqdm import tqdm
from naglmbis.models import load_charge_model
from typing import Iterator

import traceback
import logging
import json
import tempfile
import psutil
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

charge_model_esp= 'nagl-gas-charge-dipole-esp-wb-default'
charge_model_charge = "nagl-gas-charge-wb"
charge_model_dipole =  "nagl-gas-charge-dipole-wb"

gas_charge_model = load_charge_model(charge_model=charge_model_charge)
gas_charge_dipole_model = load_charge_model(charge_model=charge_model_dipole)
gas_charge_dipole_esp_model = load_charge_model(charge_model_esp)

models = {
    "charge_model": gas_charge_model,
    "dipole_model": gas_charge_dipole_model,
    "esp_model": gas_charge_dipole_esp_model,
}

charge_model_esp= 'nagl-gas-charge-dipole-esp-wb-default'
charge_model_charge = "nagl-gas-charge-wb"
charge_model_dipole =  "nagl-gas-charge-dipole-wb"

gas_charge_model = load_charge_model(charge_model=charge_model_charge)
gas_charge_dipole_model = load_charge_model(charge_model=charge_model_dipole)
gas_charge_dipole_esp_model = load_charge_model(charge_model_esp)

AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge
HA_TO_KCAL_P_MOL =  627.509391  # Hartrees to kilocalories per mole

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

def process(smiles: str) -> dict:
    """Process molecules to their charge model envs
    
    Parameters
    ----------
    retrieved: MoleculePropRecord
        record containing qm info
    
    Returns
    -------
    batch_dict: dict
        dictionary containing all the charge info
    
    """
    openff_mol = Molecule.from_smiles(smiles=smiles, allow_undefined_stereo=True)
    openff_mol.generate_conformers(n_conformers=1)
    batch_dict = {}
    #------Charges-------#
    coordinates = openff_mol.conformers[0]
    mapped_smiles = openff_mol.to_smiles(mapped=True)

    rdkit_mol = openff_mol.to_rdkit()
    batch_dict['molecule'] = mapped_smiles
    batch_dict['smiles'] = openff_mol.to_smiles()
    batch_dict['geometry'] = coordinates.m.flatten().tolist()
    batch_dict['molblock'] = rdkit.Chem.rdmolfiles.MolToMolBlock(rdkit_mol)

    batch_dict['mol_id'] = make_hash(openff_mol)
    #mbis charges
    #am1bcc chargeso
    am1bccmol = openff_mol
    am1bccmol.assign_partial_charges(partial_charge_method='am1bcc', use_conformers=[coordinates])
    batch_dict['am1bcc_charges']= (am1_bcc_charges := am1bccmol.partial_charges.magnitude.flatten().tolist())

    #------Dipoles-------#
    
    #am1bcc dipoles
    batch_dict['am1bcc_dipole'] = calculate_dipole_magnitude(
        charges=am1_bcc_charges * unit.e, 
        conformer=openff_mol.conformers[0]
    ).tolist()


    #------Charge Model RMSE Calculations-------#

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
 
    batch_dict.update(charge_models_data)
    
    return batch_dict

def main(output: str):

    # prop_store = MoleculePropStore("/mnt/storage/nobackup/nca121/paper_charge_comparisons/async_chargecraft_more_workers/conformer_test/qc_archive_run/conformers.db", cache_size=1000)
    smiles = ReadInput.read_smiles('inputs.smi')
    print(smiles)
    schema = pyarrow.schema([
        ('am1bcc_charges', pyarrow.list_(pyarrow.float64())),
     
        ('am1bcc_dipole', pyarrow.float64()),
       
        ('mol_id', pyarrow.string()),
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
        ('molecule', pyarrow.string()),
        ('grid', pyarrow.list_(pyarrow.list_(pyarrow.float64()))),
        ('geometry',pyarrow.list_(pyarrow.float64())),
        ('conformer_no', pyarrow.int16()),
        ('smiles', pyarrow.string()),
        ('energy', pyarrow.float64()),
    ])
    
    with pyarrow.parquet.ParquetWriter(where=output, schema=schema, compression='snappy') as writer:
        batches = []
        with ProcessPoolExecutor() as pool:
            # Submit jobs to process the models in parallel
            jobs = [pool.submit(process, mol) for mol in smiles]
            for future in tqdm(as_completed(jobs), total=len(jobs), desc='Processing molecules'):
                try:
                    result = future.result()
                    batches.append(result)
                    if len(batches) > 30:
                        rec_batch = pyarrow.RecordBatch.from_pylist(batches, schema=schema)
                        writer.write_batch(rec_batch)
                        batches = []
                except Exception as e:
                    print(f'Failure of job due to {e}')
                    print(traceback.format_exc())
                    continue  # Skip if the molecule was skipped or had no results


if __name__ == "__main__":
    main(output='./fda_drugs_comparison.parquet')
    
    