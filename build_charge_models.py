from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from openff.toolkit.topology import Molecule
from openff.units import unit
from collections import defaultdict
from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
from more_itertools import batched
from rdkit.Chem import rdmolfiles
from rdkit import Chem
import numpy as np
import rdkit
from openff.recharge.grids import GridSettingsType, GridGenerator
from openff.recharge.grids import LatticeGridSettings, MSKGridSettings
from MultipoleNet import load_model, build_graph_batched, D_Q

toolkit_registry = EspalomaChargeToolkitWrapper()
riniker_model = load_model()
AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge

def make_openff_molecule(mapped_smiles: str, coordinates: unit.Quantity) -> Molecule:
    return Molecule.from_mapped_smiles(mapped_smiles=mapped_smiles).add_conformer(coordinates=coordinates)


def build_mol(openff_molecule: Molecule) -> str:
    return rdmolfiles.MolToMolBlock(openff_molecule.to_rdkit())

# def riniker_charges(openff_molecule: Molecule) -> list[int]:
#     """Assign riniker charges based on an input molecule
    
#     Parameters
#     ----------
#     openff_molecule: Molecule

    
#     Returns
#     -------
#     list
#         list of charges 
    
#     """
#     (coordinates, elements) = convert_to_charge_format(openff_molecule)
#     monopoles, _, _ = riniker_model.predict(coordinates, elements)
#     return monopoles


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
    return (monopole_esp + dipole_esp + quadrupole_esp).m.flatten().tolist(), grid.m.tolist(), monopoles

def convert_to_charge_format(conformer_mol: str) -> tuple[np.ndarray,list[str]]:
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
    rdkit_conformer = rdkit.Chem.rdmolfiles.MolFromMolBlock(conformer_mol, removeHs = False)
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

def build_grid(conformer_mol: str) -> np.ndarray:
    """Builds the grid on which to assign the esp

    Parameters
    ----------
    confermer_mol: str
        conformer mol object
        
    Returns
    -------
    np.ndarray
        grid 
    
    """

    rdkit_conformer = rdkit.Chem.rdmolfiles.MolFromMolBlock(conformer_mol, removeHs = False)
    openff_mol = Molecule.from_rdkit(rdkit_conformer, allow_undefined_stereo=True)

    grid_settings = MSKGridSettings(
            type="msk", density=2.0
        )
    grid = GridGenerator.generate(openff_mol, openff_mol.conformers[0], grid_settings)

    return grid
        

def main():
    
    prop_store = MoleculePropStore("./ESP_rebuilt.db")
    molecules_list = prop_store.list()
    print(molecules_list)
    batch_size = 1000
    batch_number = 0
    test_mol_filename = 'test.mol'
    
    for batch in batched(molecules_list, 1000):
        batch_dict = defaultdict(list)
        for num_mols, molecule in enumerate(molecules_list, start=batch_number):
            #skip charged species
            if "+" in molecule or "-" in molecule:
                continue
            try:
                no_conformers = len(retrieved := prop_store.retrieve(smiles = molecule))
            except Exception as e:
                print(f'skipping this result due to {e}')
                continue
            file = 'temp1'
            for conformer in range(no_conformers):
                #multiprocess this bit
                coordinates = retrieved[conformer].conformer_quantity
                mapped_smiles = retrieved[conformer].tagged_smiles
                openff_mol: Molecule =  make_openff_molecule(mapped_smiles=mapped_smiles, coordinates=coordinates)
                batch_dict['mbis_charges'] = retrieved[conformer].mbis_charges
                Chem.MolToMolFile(openff_mol.to_rdkit(),file)
                am1_bcc_charges = openff_mol.assign_partial_charges(partial_charge_method='am1bcc')
                batch_dict['am1bcc_charges'].append(am1_bcc_charges.partial_charges)
                espaloma_charges = openff_mol.assign_partial_charges('espaloma-am1bcc', toolkit_registry=toolkit_registry)
                batch_dict['espaloma_charges'].append(espaloma_charges.partial_charges)
                riniker_monopoles = 
                
if __name__ == "__main__":
    main()