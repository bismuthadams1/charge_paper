"""Script used to generate the large scale comparisons between the charge models


"""


from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from chargecraft.storage.db import DBMoleculePropRecord, DBConformerPropRecord
from sqlalchemy.orm import Session, sessionmaker, contains_eager, joinedload
from openff.toolkit.topology import Molecule
from openff.units import unit
from collections import defaultdict
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
from more_itertools import batched
from rdkit.Chem import rdmolfiles
from rdkit import Chem
# from MultipoleNet import load_model, build_graph_batched, D_Q
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from ChargeAPI.API_infrastructure.esp_request.module_version_esp import handle_esp_request
from tqdm import tqdm
from typing import Iterator

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


toolkit_registry = EspalomaChargeToolkitWrapper()
# riniker_model = load_model()
AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge
HA_TO_KCAL_P_MOL =  627.509391  # Hartrees to kilocalories per mole
resp_solver = IterativeSolver()


def make_openff_molecule(mapped_smiles: str, coordinates: unit.Quantity) -> Molecule:
    
    molecule = Molecule.from_mapped_smiles(
        mapped_smiles=mapped_smiles,
        allow_undefined_stereo=True
    )
    molecule.add_conformer(coordinates=coordinates)
    return molecule


def build_mol(openff_molecule: Molecule) -> str:
    return rdmolfiles.MolToMolBlock(openff_molecule.to_rdkit())


def calc_riniker_esp(
                grid: unit.Quantity,
                monopole: unit.Quantity,
                dipole: list,
                quadrupole: list,
                coordinates: unit.Quantity,
                ) -> list[int]:
    """Assign charges according to charge model selected

    Parameters
    ----------
    ob_mol: generic python object depending on the charge model
        Charge model appropriate python object on which to assign the charges

    Returns
    -------
    partial_charges: list of partial charges 
    """
    #multipoles with correct units
    monopoles_quantity = monopole
    dipoles_quantity = np.array(dipole).reshape((-1,3))*unit.e*unit.angstrom
    quadropoles_quantity = np.array(quadrupole).reshape((-1,3,3))*unit.e*unit.angstrom*unit.angstrom
    coordinates_ang = coordinates.to(unit.angstrom)

    monopole_esp = calculate_esp_monopole_au(grid_coordinates=grid,
                                        atom_coordinates=coordinates_ang,
                                        charges = monopoles_quantity)

    dipole_esp = calculate_esp_dipole_au(grid_coordinates=grid,
                                    atom_coordinates=coordinates_ang,
                                    dipoles= dipoles_quantity)

    quadrupole_esp = calculate_esp_quadropole_au(grid_coordinates=grid,
                                    atom_coordinates=coordinates_ang,
                                    quadrupoles= quadropoles_quantity)

    #NOTE: ESP units, hartree/e and grid units are angstrom
    return (monopole_esp + dipole_esp + quadrupole_esp)

def convert_to_charge_format(openff_mol: Molecule) -> tuple[np.ndarray,list[str]]:
    """Convert openff molecule to appropriate format on which to assign charges

    Parameters
    ----------
    conformer_mol: string
        File path to the mol to convert to appropriate format
    
    Returns
    -------
    coordinates, elements: tuple   
        Tuple of coordinates and elements
    
    """
    #read file is an iterator so can read multiple eventually
    rdkit_conformer = openff_mol.to_rdkit()
    elements = [a.GetSymbol() for a in rdkit_conformer.GetAtoms()]
    coordinates = rdkit_conformer.GetConformer(0).GetPositions().astype(np.float32)
    return coordinates, elements  

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

def calculate_esp_dipole_au(
    grid_coordinates: unit.Quantity,  # N , 3
    atom_coordinates: unit.Quantity,  # M , 3
    dipoles: unit.Quantity,  # M , 3       
    ) -> unit.Quantity:
    """Generate the esp from the on atom dipoles
    
    Parameters
    ----------
    grid_coordinates: unit.Quantity
        grid on which to build the esp on 

    atom_coordinates: unit.Quantity
        coordinates of atoms to build the esp  
    
    dipoles: unit.Quantity
        dipoles or charges

    Returns
    -------
    dipoles_esp: unit.Quantity
        monopole esp
    """

    #prefactor
    ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)

    #Ensure everything is in AU
    dipoles = dipoles.to(unit.e*unit.bohr)
    grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
    atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr

    displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N , M , 3 
    distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M 
    inv_distance_cubed = 1 / distance**3 #1/B
    #Hadamard product for element-wise multiplication
    dipole_dot = np.sum(displacement * dipoles[None,:,:], axis=-1) # dimless * e.a

    esp = ke*np.sum(inv_distance_cubed* dipole_dot,axis=1) # e.a/a**2 

    return esp.to(AU_ESP)

def calculate_esp_quadropole_au(
    grid_coordinates: unit.Quantity,  # N x 3
    atom_coordinates: unit.Quantity,  # M x 3
    quadrupoles: unit.Quantity,  # M N 
    ) -> unit.Quantity:
    """Generate the esp from the on atom quandropoles
    
    Parameters
    ----------
    grid_coordinates: unit.Quantity
        grid on which to build the esp on 

    atom_coordinates: unit.Quantity
        coordinates of atoms to build the esp  
    
    quandropoles: unit.Quantity
        dipoles or charges

    Returns
    -------
    quandropoles_esp: unit.Quantity
        monopole esp
    """

    #prefactor
    ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)
    #Ensure everything is in AU
    quadrupoles = _detrace(quadrupoles.to(unit.e * unit.bohr * unit.bohr))
    grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
    atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr

    displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N , M , 3 
    distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M 
    inv_distance = 1 / distance #1/B

    quadrupole_dot_1 = np.sum(quadrupoles[None,:,:] * displacement[:,:,None],axis=-1)
    quadrupole_dot_2 = np.sum(quadrupole_dot_1*displacement,axis=-1)
    esp = ke*np.sum((3*quadrupole_dot_2*(1/2 * inv_distance**5)),axis=-1)

    return esp.to(AU_ESP)

def _detrace( quadrupoles: unit.Quantity) -> unit.Quantity:
    """Make sure we have the traceless quadrupole tensor.

    Parameters
    ----------
    quadrupoles : unit.Quantity
        Quadrupoles.

    Returns
    -------
    unit.Quantity
        Detraced quadrupoles.
    """
    quadrupoles = quadrupoles.m
    for i in range(quadrupoles.shape[0]):
        trace = np.trace(quadrupoles[i])
        trace /= 3
        quadrupoles[i][0][0] -= trace
        quadrupoles[i][1][1] -= trace
        quadrupoles[i][2][2] -= trace

        return quadrupoles * unit.e * unit.bohr * unit.bohr
        
def calculate_resp_charges(openff_mol: Molecule,
                           grid: unit.Quantity,
                           esp: unit.Quantity,
                           qc_data_settings: ESPSettings) -> list[float]:
    """Calculate resp charges given a set of input data
    
    Parameters
    ----------
    grid: unit.Quantity
        grid in which the ESP was calculate don
    esp: unit.Quantity
        esp calculated from the qm calculation
    qc_data_settings: ESPSettings
        esp settings the esp was calculated at
        
    Returns
    -------
    list[float]
        list of resp charges
    
    
    """
    qc_data_record = MoleculeESPRecord.from_molecule(
        openff_mol, openff_mol.conformers[0], grid, esp, None, qc_data_settings
    )

    resp_charge_parameter = generate_resp_charge_parameter(
        [qc_data_record], resp_solver
    )
    
    matchs = openff_mol.chemical_environment_matches(query=resp_charge_parameter.smiles)
    resp_charges = [0.0 for _ in range(openff_mol.n_atoms)]
    for match in matchs:
        for i, atom_indx in enumerate(match):
            resp_charges[atom_indx] = resp_charge_parameter.value[i]
    
    return np.round(resp_charges, 4).tolist()

def calculate_resp_multiconformer_charges(
    openff_mols: list[Molecule],
    grids: list[unit.Quantity],
    esps: list[unit.Quantity],
    qc_data_settings: ESPSettings) -> list[float]:
    """Calculate resp charges given a set of input data
    
    Parameters
    ----------
    grid: unit.Quantity
        grid in which the ESP was calculate don
    esp: unit.Quantity
        esp calculated from the qm calculation
    qc_data_settings: ESPSettings
        esp settings the esp was calculated at
        
    Returns
    -------
    list[float]
        list of resp charges
    
    
    """
    qc_data_records = []
    for idx,(conformer, grid, esp) in enumerate(zip(openff_mols,grids,esps)):
        print(f"Processing conformer {idx}, number of atoms: {conformer.n_atoms}")
        qc_data_record = MoleculeESPRecord.from_molecule(
            conformer, conformer.conformers[0], grid, esp, None, qc_data_settings
        )
        qc_data_records.append(qc_data_record)
        print(f"QC Data Record {idx} molecule has {qc_data_record.molecule.n_atoms} atoms.")

        
    resp_charge_parameter = generate_resp_charge_parameter(
        qc_data_records, resp_solver
    )
    print(f"Number of RESP charges: {len(resp_charges)}")

    resp_charges = [resp_charge_parameter.value[i] for i in range(len(resp_charge_parameter.value))]
    
    for idx, charge in enumerate(resp_charges):
        print(f"Atom {idx}: Charge = {charge}")

    return np.round(resp_charges, 4).tolist()
    
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
    
def process_molecule(retrieved: MoleculePropRecord, conformer_no: int):
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
    batch_dict = {}
    #------Charges-------#
    coordinates = retrieved.conformer_quantity
    mapped_smiles = retrieved.tagged_smiles
    openff_mol: Molecule =  make_openff_molecule(
        mapped_smiles=mapped_smiles,
        coordinates=coordinates
    )
    rdkit_mol = openff_mol.to_rdkit()
    batch_dict['molecule'] = mapped_smiles
    batch_dict['smiles'] = openff_mol.to_smiles()
    batch_dict['conformer_no'] = conformer_no
    batch_dict['geometry'] = coordinates.m.flatten().tolist()
    batch_dict['molblock'] = rdkit.Chem.rdmolfiles.MolToMolBlock(rdkit_mol)
    batch_dict['grid'] = retrieved.grid_coordinates.tolist()
    batch_dict['esp'] = retrieved.esp.tolist()
    batch_dict['esp_settings'] = retrieved.esp_settings
    batch_dict['energy'] = retrieved.energy.tolist()
    batch_dict['mol_id'] = make_hash(openff_mol)
    #mbis charges
    batch_dict['mbis_charges'] = (mbis_charges := retrieved.mbis_charges.flatten().tolist())
    # Chem.MolToMolFile(openff_mol.to_rdkit(),file)
    #am1bcc chargeso
    am1bccmol = Molecule.from_mapped_smiles(mapped_smiles=retrieved.tagged_smiles)
    #NOTE: when computing AM1-BCC charge ensemble, remove the 'use_conformers' argument to calculate charges over an ensemble of conformers.
    am1bccmol.assign_partial_charges(
        partial_charge_method='am1bcc',
     #   use_conformers=[coordinates]
    )
    batch_dict['am1bcc_charges']= (am1_bcc_charges := am1bccmol.partial_charges.magnitude.flatten().tolist())
    #espaloma charges
    espalomamol = openff_mol
    espalomamol.assign_partial_charges('espaloma-am1bcc', toolkit_registry=toolkit_registry, use_conformers=[coordinates])
    batch_dict['espaloma_charges']= (espaloma_charges := espalomamol.partial_charges.magnitude.flatten().tolist())

    #resp charges
    grid = retrieved.grid_coordinates_quantity
    esp = retrieved.esp_quantity
    esp_settings = retrieved.esp_settings
    resp_charges = calculate_resp_charges(openff_mol, grid = grid, esp=esp,qc_data_settings=esp_settings)
    batch_dict['resp_charges'] = np.array(resp_charges).flatten().tolist()
    
    #------Dipoles-------#
    
    qm_dipole = retrieved.dipole_quantity 
    batch_dict['qm_dipoles'] = np.linalg.norm(qm_dipole.m).tolist()
    
    #mbis dipole
    batch_dict['mbis_dipoles'] = calculate_dipole_magnitude(
        charges=retrieved.mbis_charges_quantity, 
        conformer=retrieved.conformer_quantity
    ).tolist()
    #am1bcc dipoles
    batch_dict['am1bcc_dipole'] = calculate_dipole_magnitude(
        charges=am1_bcc_charges * unit.e, 
        conformer=retrieved.conformer_quantity
    ).tolist()
    #espaloma dipole
    batch_dict['espaloma_dipole'] = calculate_dipole_magnitude(
        charges=espaloma_charges* unit.e, 
        conformer=retrieved.conformer_quantity
    ).tolist()
 
    #resp dipole
    batch_dict['resp_dipole'] =  calculate_dipole_magnitude(
        charges=resp_charges* unit.e, 
        conformer=retrieved.conformer_quantity
    ).tolist()
        
    #------ESP RMSE Calculations-------#

    # Convert grid and atom coordinates to Bohr
    grid_coordinates = retrieved.grid_coordinates_quantity.to(unit.bohr)  # Units: Bohr
    atom_coordinates = retrieved.conformer_quantity.to(unit.bohr)  # Units: Bohr

    # QM ESP (already in atomic units)
    qm_esp = retrieved.esp_quantity.flatten()  # Units: Hartree / e
    batch_dict['qm_esp'] = qm_esp.m.flatten().tolist()
    # AM1-BCC ESP
    am1bcc_esp = calculate_esp_monopole_au(
        grid_coordinates=grid_coordinates,
        atom_coordinates=atom_coordinates,
        charges=am1_bcc_charges
    )
    batch_dict['am1bcc_esp'] = am1bcc_esp.m
    am1bcc_esp_rms = (((am1bcc_esp - qm_esp) ** 2).mean() ** 0.5).magnitude
    batch_dict['am1bcc_esp_rms'] = am1bcc_esp_rms * HA_TO_KCAL_P_MOL

    # Espaloma ESP
    espaloma_esp = calculate_esp_monopole_au(
        grid_coordinates=grid_coordinates,
        atom_coordinates=atom_coordinates,
        charges=espaloma_charges
    )
    batch_dict['espaloma_esp'] = espaloma_esp.m
    espaloma_esp_rms = (((espaloma_esp - qm_esp) ** 2).mean() ** 0.5).magnitude
    batch_dict['espaloma_esp_rms'] = espaloma_esp_rms * HA_TO_KCAL_P_MOL

    # RESP ESP
    resp_esp = calculate_esp_monopole_au(
        grid_coordinates=grid_coordinates,
        atom_coordinates=atom_coordinates,
        charges=resp_charges
    )
    batch_dict['resp_esp'] = resp_esp.m
    resp_esp_rms = (((resp_esp - qm_esp) ** 2).mean() ** 0.5).magnitude
    batch_dict['resp_esp_rms'] = resp_esp_rms * HA_TO_KCAL_P_MOL

    # MBIS ESP
    mbis_esp = calculate_esp_monopole_au(
        grid_coordinates=grid_coordinates,
        atom_coordinates=atom_coordinates,
        charges=mbis_charges
    )
    batch_dict['mbis_esp'] = mbis_esp.m
    mbis_esp_rms = (((mbis_esp - qm_esp) ** 2).mean() ** 0.5).magnitude
    batch_dict['mbis_esp_rms'] = mbis_esp_rms * HA_TO_KCAL_P_MOL

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
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(process_molecule, model, conformer_no) for (model, conformer_no) in batch_models]
        results_batch = []
        for future in tqdm(as_completed(jobs), total=len(jobs), desc='Processing molecules'):
            try:
                result = future.result()
                results_batch.append(result)
                # Write every 5 results
                if len(results_batch) >= 5:
                    print('writing 5 items')
                    rec_batch = pyarrow.RecordBatch.from_pylist(results_batch, schema=schema)
                    writer.write_batch(rec_batch)
                    results_batch.clear()
            except Exception as e:
                print(f'Failure of job due to {e}')
                print(traceback.format_exc())
                continue

        # After processing all futures, write any leftover results
        if len(results_batch) > 0:
            print(f"Writing leftover {len(results_batch)} results")
            rec_batch = pyarrow.RecordBatch.from_pylist(results_batch, schema=schema)
            writer.write_batch(rec_batch)
            results_batch.clear()

    
def process_esp(results_batch):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary molblock file in the temp directory
        tmp_input_file = create_mol_block_tmp_file(pylist=results_batch, temp_dir=temp_dir)
        # Run the ESP computation in batched mode
        output_file = handle_esp_request(
            charge_model='RIN',
            conformer_mol=tmp_input_file,
            broken_up=True,
            batched=True,
            batched_grid=True,
        )

        with open(output_file['file_path'], 'r') as f:
            esps_dict = json.load(f)

 
        for item in results_batch:
            mol_id = item['mol_id']
            esp_result = esps_dict.get(mol_id)
            rdkit_mol = Chem.rdmolfiles.MolFromMolBlock(item['molblock'], removeHs = False)
            openff_mol = Molecule.from_rdkit(rdkit_mol)

            if esp_result:
                riniker_monopoles = esp_result['esp_values'][0]
                item['riniker_monopoles'] = riniker_monopoles
                #include charge and dipole contributions
                D_charge = np.sum(np.array(riniker_monopoles)[:, np.newaxis] * openff_mol.conformers[0].to(unit.bohr).m, axis=0) 
                summed_dipole = np.sum(np.array(esp_result['esp_values'][1]).reshape(-1,3), axis=0) + D_charge
                item['riniker_dipoles'] = np.linalg.norm(summed_dipole).tolist()
                
                riniker_esp = calc_riniker_esp(
                    grid= (esp_result['esp_grid'] * unit.angstrom).reshape(3,-1),
                    monopole= esp_result['esp_values'][0] * unit.e,
                    dipole= esp_result['esp_values'][1],
                    quadrupole= esp_result['esp_values'][2],
                    coordinates= openff_mol.conformers[0]
                )

                qm_esp = np.array(item['qm_esp']) * unit.hartree/unit.e
                item['riniker_esp_rms'] =  ((((riniker_esp - qm_esp)) ** 2).mean() ** 0.5).magnitude * HA_TO_KCAL_P_MOL
            else:
                print(f'No ESP result found for molecule ID {mol_id}')


def process_resp_multiconfs(results_batch):
    complete_smiles = []
    for item in results_batch:
        if item['smiles'] in complete_smiles:
            continue
        else:
            target_smiles = item['smiles']
            complete_smiles.append(target_smiles)
            esp_settings = item['esp_settings']
            
            # Collect matching items and their data
            conformers = []
            grids = []
            esps = []
            matching_items = []
            for item2 in results_batch:
                if item2['smiles'] == target_smiles:
                    rdkit_mol = rdmolfiles.MolFromMolBlock(item2['molblock'], removeHs=False)
                    conformer = Molecule.from_rdkit(
                        rdmol=rdkit_mol, 
                        allow_undefined_stereo=True, 
                        hydrogens_are_explicit=True
                    )
                    conformers.append(conformer)
                    grid = np.array(item2['grid']).reshape(-1, 3) * unit.angstrom
                    grids.append(grid)
                    esp = np.array(item2['esp']) * unit.hartree / unit.e
                    esps.append(esp)
                    matching_items.append(item2)

            # Calculate RESP charges for all conformers of the molecule
            resp_charges = calculate_resp_multiconformer_charges(
                openff_mols=conformers,
                grids=grids,
                esps=esps,
                qc_data_settings=esp_settings
            )
            print(target_smiles)
            print(conformers)
            
            # Iterate over the matching items and their corresponding conformers
            for item2, conf, grid in zip(matching_items, conformers, grids):
                item2['resp_multiconf_charges'] = resp_charges
                print(resp_charges)
                item2['resp_multiconf_dipoles'] = calculate_dipole_magnitude(
                    charges=resp_charges,
                    conformer=conf.conformers[0]
                )
                resp_multi_esp = calculate_esp_monopole_au(
                    grid_coordinates=grid,
                    atom_coordinates=conf.conformers[0],
                    charges=resp_charges, 
                )
                item2['resp_multiconf_esp'] = resp_multi_esp
                qm_esp = np.array(item2['qm_esp']) * unit.hartree / unit.e
                item2['resp_multiconf_esp_rmse'] = (
                    ((((resp_multi_esp - qm_esp)) ** 2).mean() ** 0.5).magnitude * HA_TO_KCAL_P_MOL
                )


def main(output: str):

    prop_store = MoleculePropStore("./ESP_rebuilt_conformers.db", cache_size=1000)
    
    molecules_list = prop_store.list()
    number_of_molecules = len(molecules_list)
    
    schema = pyarrow.schema([
        ('mbis_charges', pyarrow.list_(pyarrow.float64())),
        ('am1bcc_charges', pyarrow.list_(pyarrow.float64())),
        ('espaloma_charges', pyarrow.list_(pyarrow.float64())),
        ('riniker_monopoles', pyarrow.list_(pyarrow.float64())),
        ('resp_charges', pyarrow.list_(pyarrow.float64())),
        ('qm_dipoles', pyarrow.float64()),
        ('mbis_dipoles', pyarrow.float64()),
        ('am1bcc_dipole', pyarrow.float64()),
        ('espaloma_dipole', pyarrow.float64()),
        ('riniker_dipoles', pyarrow.float64()),
        ('resp_dipole', pyarrow.float64()),
        ('am1bcc_esp_rms', pyarrow.float64()),
        ('am1bcc_esp',pyarrow.list_(pyarrow.float64())),
        ('espaloma_esp_rms', pyarrow.float64()),
        ('espaloma_esp',pyarrow.list_(pyarrow.float64())),
        ('resp_esp_rms', pyarrow.float64()),
        ('resp_esp', pyarrow.list_(pyarrow.float64())),
        ('mbis_esp_rms', pyarrow.float64()),
        ('mbis_esp', pyarrow.list_(pyarrow.float64())),
        ('riniker_esp_rms',pyarrow.float64()),
        ('riniker_esp',pyarrow.float64()),
        ('qm_esp', pyarrow.list_(pyarrow.float64())),
        ('molecule', pyarrow.string()),
        ('grid', pyarrow.list_(pyarrow.list_(pyarrow.float64()))),
        ('geometry',pyarrow.list_(pyarrow.float64())),
        ('conformer_no', pyarrow.int16()),
        ('smiles', pyarrow.string()),
        ('energy', pyarrow.float64()),
    ])
    
    with pyarrow.parquet.ParquetWriter(where=output, schema=schema, compression='snappy') as writer:
        
        chunk_size = 100  # Adjust as needed
        for i in range(0, number_of_molecules, chunk_size):
            subset = molecules_list[i:i+chunk_size]
            batch_models = []
            for smiles in subset:
                retrieved_records = prop_store.retrieve(smiles=smiles)
                for conformer_no, retrieved in enumerate(retrieved_records):
                    batch_models.append((retrieved, conformer_no))
            # Now process this subset only
            for batch in batched(batch_models, 100):
                process_and_write_batch(batch, schema, writer)
                
   
        
if __name__ == "__main__":
    # main(output='./charge_models_am1bcc_default.parquet')
    main(output='./conformer_test_out.parquet')
