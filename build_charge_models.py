"""Script used to generate the large scale comparisons between the charge models


"""


from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from openff.toolkit.topology import Molecule
from openff.units import unit
from collections import defaultdict
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
from more_itertools import batched
from rdkit.Chem import rdmolfiles
from rdkit import Chem
from MultipoleNet import load_model, build_graph_batched, D_Q
from concurrent.futures import ProcessPoolExecutor, as_completed
from ChargeAPI.API_infrastructure.esp_request.module_version_esp import handle_esp_request
from tqdm import tqdm
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
riniker_model = load_model()
AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge
resp_solver = IterativeSolver()


def make_openff_molecule(mapped_smiles: str, coordinates: unit.Quantity) -> Molecule:
    
    molecule = Molecule.from_mapped_smiles(mapped_smiles=mapped_smiles, allow_undefined_stereo=True)
    molecule.add_conformer(coordinates=coordinates)
    return molecule


def build_mol(openff_molecule: Molecule) -> str:
    return rdmolfiles.MolToMolBlock(openff_molecule.to_rdkit())


def riniker_esp(openff_molecule: Molecule, grid: np.ndarray) -> list[int]:
    """Assign charges according to charge model selected

    Parameters
    ----------
    ob_mol: generic python object depending on the charge model
        Charge model appropriate python object on which to assign the charges

    Returns
    -------
    partial_charges: list of partial charges 
    """
    (coordinates, elements) = convert_to_charge_format(openff_molecule)
    print(f'rdkit to openff yields {(coordinates, elements)}')
    monopoles, dipoles, quadrupoles = riniker_model.predict(coordinates, elements)
    #multipoles with correct units
    monopoles_quantity = monopoles.numpy()*unit.e
    dipoles_quantity = dipoles.numpy()*unit.e*unit.angstrom
    quadropoles_quantity = quadrupoles.numpy()*unit.e*unit.angstrom*unit.angstrom
    coordinates_ang = coordinates * unit.angstrom
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
    return (monopole_esp + dipole_esp + quadrupole_esp).m.flatten().tolist(), grid.m.tolist(), monopoles, dipoles

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

    #Ensure everything is in AU and correct dimensions
    charges = charges.flatten()
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
    quadrupoles = quadrupoles.to(unit.e*unit.bohr*unit.bohr)    
    grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
    atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr

    displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N , M , 3 
    distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M 
    inv_distance = 1 / distance #1/B

    quadrupole_dot_1 = np.sum(quadrupoles[None,:,:] * displacement[:,:,None],axis=-1)
    quadrupole_dot_2 = np.sum(quadrupole_dot_1*displacement,axis=-1)
    esp = ke*np.sum((3*quadrupole_dot_2*(1/2 * inv_distance**5)),axis=-1)

    return esp.to(AU_ESP)
        
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
    resp_charges = LibraryChargeGenerator.generate(
        openff_mol, LibraryChargeCollection(parameters=[resp_charge_parameter])
    )
    
    return np.round(resp_charges, 4).tolist()

def calculate_dipole_magnitude(charges: np.ndarray,
                               conformer: np.ndarray) -> float:
    """Calculate dipole magnitude
    
    Parameters
    ----------
    charges: np.
    
    """
    reshaped_charges = np.reshape(charges,(-1,1))
    dipole_vector = np.sum(conformer * reshaped_charges,axis=0)
    dipole_magnitude = np.linalg.norm(dipole_vector)

    return dipole_magnitude

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
    
def process_molecule(retrieved: MoleculePropRecord):
    """Process molecules to their charge model envs
    
    Parameters
    ----------
    retrieved: MoleculePropRecord
        record containing qm info
    
    Returns
    -------
    batch_dict: defaultdict(list)
        dictionary containing all the charge info
    
    """
    batch_dict = defaultdict(list)
    #------Charges-------#
    coordinates = retrieved.conformer_quantity
    mapped_smiles = retrieved.tagged_smiles
    openff_mol: Molecule =  make_openff_molecule(
        mapped_smiles=mapped_smiles,
        coordinates=coordinates
    )
    rdkit_mol = openff_mol.to_rdkit()
    print('off mol')
    print(openff_mol)
    batch_dict['molecule'] = openff_mol.to_smiles()
    batch_dict['geometry'] = coordinates
    batch_dict['molblock'] = rdkit.Chem.rdmolfiles.MolToMolBlock(rdkit_mol)
    batch_dict['grid'] = retrieved.grid_coordinates.tolist()
    batch_dict['mol_id'] = make_hash(openff_mol)
    #mbis charges
    batch_dict['mbis_charges'] = retrieved.mbis_charges.magnitude.tolist()
    # Chem.MolToMolFile(openff_mol.to_rdkit(),file)
    #am1bcc chargeso
    am1bccmol = openff_mol
    am1bccmol.assign_partial_charges(partial_charge_method='am1bcc')
    batch_dict['am1bcc_charges'].append(am1_bcc_charges := am1bccmol.partial_charges.magnitude.tolist())
    #espaloma charges
    espalomamol = openff_mol
    espalomamol.assign_partial_charges('espaloma-am1bcc', toolkit_registry=toolkit_registry)
    batch_dict['espaloma_charges'].append(espaloma_charges := espalomamol.partial_charges.magnitude.tolist())
    #riniker charges
    # esp, _, monopole, dipoles  =  riniker_esp(openff_molecule=openff_mol,
    #                                           grid =retrieved.grid_coordinates )
    # batch_dict['riniker_monopole_charges'] = monopole.tolist()
    #resp charges
    grid = retrieved.grid_coordinates_quantity
    esp = retrieved.esp_quantity
    esp_settings = retrieved.esp_settings
    resp_charges = calculate_resp_charges(openff_mol, grid = grid, esp=esp,qc_data_settings=esp_settings)
    batch_dict['resp_charges'] = resp_charges
    
    #------Dipoles-------#
    
    qm_dipole = retrieved.dipole 
    batch_dict['qm_dipoles'] = qm_dipole.tolist()
    
    #mbis dipole
    batch_dict['mbis_dipoles'] = calculate_dipole_magnitude(
        charges=retrieved.mbis_charges, 
        conformer=retrieved.conformer
    ).tolist()
    #am1bcc dipoles
    batch_dict['am1bcc_dipole'] = calculate_dipole_magnitude(
        charges=am1_bcc_charges, 
        conformer=retrieved.conformer
    ).tolist()
    #espaloma dipole
    batch_dict['espaloma_dipole'] = calculate_dipole_magnitude(
        charges=espaloma_charges, 
        conformer=retrieved.conformer
    ).tolist()
    
    #riniker dipole
    # batch_dict['riniker_dipoles'] = np.linalg.norm(np.sum(dipoles, axis=0)).tolist()
    
    #resp dipole
    batch_dict['resp_dipole'] =  calculate_dipole_magnitude(
        charges=resp_charges, 
        conformer=retrieved.conformer
    ).tolist()
    
    return dict(batch_dict)

def create_mol_block_tmp_file(pylist: list[dict], temp_dir: str) -> None:
    """Create a tmp file with all the molblocks
    
    Parameters
    ----------
    pylist: list[dict]
        dictionary of the pylist results
    
    """
    json_dict = {}
    for item in pylist:
        print(item)
        json_dict[item['mol_id']] = (item['molblock'],item['grid'].tolist())
    json_file = os.path.join(temp_dir, 'molblocks.json')
    json.dump(json_dict, open(json_file, "w"))
    
    return json_file

def main(output: str):
    
    prop_store = MoleculePropStore("./ESP_rebuilt.db", cache_size=100)
    molecules_list = prop_store.list()
    # print(molecules_list)

    schema = pyarrow.schema([
        ('mbis_charges', pyarrow.list_(pyarrow.float64())),
        ('am1bcc_charges', pyarrow.list_(pyarrow.float64())),
        ('espaloma_charges', pyarrow.list_(pyarrow.float64())),
        ('riniker_monopole_charges', pyarrow.list_(pyarrow.float64())),
        ('resp_charges', pyarrow.list_(pyarrow.float64())),
        ('qm_dipoles', pyarrow.float64()),
        ('mbis_dipoles', pyarrow.float64()),
        ('am1bcc_dipole', pyarrow.float64()),
        ('espaloma_dipole', pyarrow.float64()),
        ('riniker_dipoles', pyarrow.float64()),
        ('resp_dipole', pyarrow.float64()),
        ('molecule', pyarrow.string()),
        ('grid', pyarrow.list_(pyarrow.float64())),
    ])    
    limit = 40
    limited_molecules_list = molecules_list[:limit]  
    with pyarrow.parquet.ParquetWriter(where=output, schema=schema) as writer:
            batches = []
            for batch in tqdm(batched(limited_molecules_list, 20), total=len(limited_molecules_list),desc='Building batches'):
                batched_conformers = []
                for molecule in batch:
                    #skip charged species 
                    if "+" in molecule or "-" in molecule:
                        continue
                    try:
                        no_conformers = len(retrieved := prop_store.retrieve(smiles = molecule))
                    except Exception as e:
                        print(f'skipping this result due to {e}')
                        print(traceback.format_exc())
                        continue
                    for conformer_no in range(no_conformers):
                        conformer = retrieved[conformer_no]
                        batched_conformers.append(conformer)
                batches.append(batched_conformers)
            with ProcessPoolExecutor(max_workers=8) as pool:
                total_batch = []
                for batch in batches:
                    results_batch = []
                    jobs = [pool.submit(process_molecule, conformer) for conformer in batch]

                    for future in tqdm(as_completed(jobs), total=len(jobs), desc='Processing molecules'):
                        try:
                            result = future.result()
                        except Exception as e:
                            print(f'failure of job due to {e}')
                            print(traceback.format_exc())
                            continue  # Skip if the molecule was skipped or had no results

                        # for rec_data in result:
                        # Convert rec_data to a format suitable for pyarrow
                        print('dict')
                        print(result)
                        results_batch.append(result)
                
                    #collect molblocks from total_batch and send to charge model
                    #add results back to total_batch dictionary 
                    #results will be ['riniker_monopoles'], ['riniker_dipoles']
                    #create tmp file
                    # total_batches_riniker = []
                    # print('processing molecules for riniker model')
                    # for batch in tqdm(batched(total_batch, 20), total = len(total_batches_riniker)):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Create temporary molblock file in the temp directory
                        tmp_input_file = create_mol_block_tmp_file(pylist=results_batch, temp_dir=temp_dir)

                        # tmp_output_file = os.path.join(temp_dir, 'esp_results.json')

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
                            if esp_result:
                                item['riniker_monopoles'] = np.array(esp_result['esp_values'][0])
                                item['riniker_dipoles'] = np.linalg.norm(np.sum(esp_result['esp_values'][1], axis=0)).tolist()
                            else:
                                print(f'No ESP result found for molecule ID {mol_id}')
                        # total_batches_riniker.append(batch)
                    print('total results batch:')
                    print(results_batch)
                    rec_batch = pyarrow.RecordBatch.from_pylist(results_batch, schema=schema)
                    writer.write_batch(rec_batch)
                
if __name__ == "__main__":
    main(output='./charge_models.parquet')