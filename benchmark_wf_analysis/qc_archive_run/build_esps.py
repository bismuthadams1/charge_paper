from chargecraft.storage.qcarchive_transfer import QCArchiveToLocalDB
from qcportal import PortalClient
from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from openff.recharge.grids import LatticeGridSettings
from concurrent.futures import ProcessPoolExecutor, as_completed
from openff.toolkit.topology import Molecule
from openff.recharge.grids import GridGenerator, GridSettingsType
from openff.recharge.esp import DFTGridSettings
from multiprocessing import get_context
from rdkit import Chem
from tqdm import tqdm
from qcportal.singlepoint.record_models import SinglepointRecord
from typing import Union
from qcelemental.models import Molecule as QCMolecule
from openff.recharge.esp.qcresults import reconstruct_density, compute_esp
from openff.units import unit
from chargecraft.storage.data_classes import ESPSettings, PCMSettings, DDXSettings
import numpy as np
import psi4
import traceback
from typing import Optional

def build_grid(molecule: Molecule, conformer: unit.Quantity, grid_settings: GridSettingsType) -> unit.Quantity:
    grid = GridGenerator.generate(molecule, conformer, grid_settings)
    return grid


def compute_properties(qc_molecule: "qcelemental.models.Molecule", density: np.ndarray, esp_settings: ESPSettings) -> dict[str, np.ndarray]:
    psi4.core.be_quiet()

    psi4_molecule = psi4.geometry(qc_molecule.to_string("psi4", "angstrom"))
    psi4_molecule.reset_point_group("c1")

    psi4_wavefunction = psi4.core.RHF(
        psi4.core.Wavefunction.build(psi4_molecule, esp_settings.basis),
        psi4.core.SuperFunctional(),
    )
    psi4_wavefunction.Da().copy(psi4.core.Matrix.from_array(density))

    psi4.oeprop(psi4_wavefunction,
                "DIPOLE",
                "QUADRUPOLE",
                "MULLIKEN_CHARGES",
                "LOWDIN_CHARGES",
                "MBIS_CHARGES",
                "MBIS_DIPOLE",
                "MBIS_QUADRUPOLE")

    variables_dictionary = dict()
    variables_dictionary["MULLIKEN_CHARGES"] = psi4_wavefunction.variable("MULLIKEN_CHARGES") * unit.e
    variables_dictionary["LOWDIN_CHARGES"] = psi4_wavefunction.variable("LOWDIN_CHARGES") * unit.e
    variables_dictionary["MBIS CHARGES"] = psi4_wavefunction.variable("MBIS CHARGES") * unit.e
    variables_dictionary["MBIS DIPOLE"] = psi4_wavefunction.variable("MBIS DIPOLES") * unit.e * unit.bohr_radius
    variables_dictionary["MBIS QUADRUPOLE"] = psi4_wavefunction.variable("MBIS QUADRUPOLES") * unit.e * unit.bohr_radius**2
    variables_dictionary["MBIS OCTOPOLE"] = psi4_wavefunction.variable("MBIS OCTUPOLES") * unit.e * unit.bohr_radius**3
    variables_dictionary["DIPOLE"] = psi4_wavefunction.variable("DIPOLE") * unit.e * unit.bohr_radius
    variables_dictionary["QUADRUPOLE"] = psi4_wavefunction.variable("QUADRUPOLE") * unit.e * unit.bohr_radius**2
    variables_dictionary["ALPHA_DENSITY"] = psi4_wavefunction.Da().to_array()
    variables_dictionary["BETA_DENSITY"] = psi4_wavefunction.Db().to_array()

    return variables_dictionary


def dft_grid_settings(item: SinglepointRecord) -> DFTGridSettings | None:
    if 'xc grid radial points' not in item.properties:
        return DFTGridSettings.Default
    if item.properties['xc grid radial points'] == 75.0 and item.properties['xc grid spherical points'] == 302.0:
        return DFTGridSettings.Default
    elif item.properties['xc grid radial points'] == 85.0 and item.properties['xc grid spherical points'] == 434.0:
        return DFTGridSettings.Medium
    elif item.properties['xc grid radial points'] == 99.0 and item.properties['xc grid spherical points'] == 590.0:
        return DFTGridSettings.Fine
    else:
        return None


def construct_variables_dictionary(item: SinglepointRecord) -> dict:
    variables_dictionary = dict()

    def add_value(key, value):
        variables_dictionary[key] = value

    add_value("MULLIKEN_CHARGES", item.properties.get('mulliken charges') * unit.e if 'mulliken charges' in item.properties else None)
    add_value("LOWDIN_CHARGES", item.properties.get('lowdin charges') * unit.e if 'lowdin charges' in item.properties else None)
    add_value("MBIS CHARGES", item.properties.get('mbis charges') * unit.e if 'mbis charges' in item.properties else None)
    add_value("MBIS DIPOLE", item.properties.get('mbis dipoles') * unit.e * unit.bohr_radius if 'mbis dipoles' in item.properties else None)
    add_value("MBIS QUADRUPOLE", item.properties.get('mbis quadrupoles') * unit.e * unit.bohr_radius**2 if 'mbis quadrupoles' in item.properties else None)
    add_value("MBIS OCTOPOLE", item.properties.get('mbis octupoles') * unit.e * unit.bohr_radius**3 if 'mbis octupoles' in item.properties else None)
    add_value("DIPOLE", item.properties.get('scf dipole') * unit.e * unit.bohr_radius if 'scf dipole' in item.properties else None)
    add_value("QUADRUPOLE", item.properties.get('scf quadrupole') * unit.e * unit.bohr_radius**2 if 'scf quadrupole' in item.properties else None)
    add_value("ALPHA_DENSITY", getattr(item.wavefunction, 'scf_density_a', None))
    add_value("BETA_DENSITY", getattr(item.wavefunction, 'scf_density_b', None))

    return variables_dictionary


def process_item_function(
    item: SinglepointRecord,
    exclude_keys: Optional[list],
    qm_esp: bool,
    compute_properties: bool,
    return_store: bool,
    grid_settings: GridSettingsType,
    esp_calculator: compute_esp,
    dft_grid_settings_func: dft_grid_settings,
    construct_variables_dictionary_func: construct_variables_dictionary,
    build_grid_func: build_grid,
    compute_properties_func: compute_properties
):
    if exclude_keys and any(key in item.specification.keywords for key in exclude_keys):
        return None

    # Ensure orientation is correct
    qc_mol = item.molecule
    qc_data = qc_mol.dict()
    qc_data['fix_com'] = True
    qc_data['fix_orientation'] = True
    qc_mol = QCMolecule.from_data(qc_data)

    openff_molecule = Molecule.from_qcschema(qc_mol, allow_undefined_stereo=True)
    openff_conformer = openff_molecule.conformers[0]

    if item.properties is None:
        print(f'No calculation data for molecule: {openff_molecule.to_smiles()} due to {item.status}')
        return None

    esp_settings = ESPSettings(
        basis=item.specification.basis,
        method=item.specification.method,
        grid_settings=grid_settings,
        pcm_settings=PCMSettings(
            solver='',
            solvent='',
            radii_model='',
            radii_scaling='',
            cavity_area=''
        ) if 'PCM' in item.specification.keywords else None,
        ddx_settings=DDXSettings(
            solvent=None if 'ddx_solvent_epsilon' not in item.specification.keywords
            else item.specification.keywords['ddx_solvent'],
            epsilon=item.specification.keywords.get('ddx_solvent_epsilon', None),
            radii_set='uff',
            ddx_model=item.specification.keywords.get('ddx_model', '').upper()
        ) if 'ddx' in item.specification.keywords else None,
        psi4_dft_grid_settings=dft_grid_settings_func(item)
    )

    grid = build_grid_func(molecule=openff_molecule, conformer=openff_conformer, grid_settings=grid_settings)

    if compute_properties:
        density = reconstruct_density(wavefunction=item.wavefunction, n_alpha=item.properties['calcinfo_nalpha'])
        variables_dictionary = compute_properties_func(
            qc_molecule=qc_mol,
            density=density,
            esp_settings=esp_settings
        )
    else:
        variables_dictionary = construct_variables_dictionary_func(item=item)

    print('Computed variables dictionary')
    print(variables_dictionary)

    if qm_esp:
        print("Computing the QM esp")
        if density is None:
            density = reconstruct_density(
                wavefunction=item.wavefunction,
                n_alpha=item.properties['calcinfo_nalpha']
            )
        esp, electric_field = compute_esp(
            qc_molecule=qc_mol,
            density=density,
            esp_settings=esp_settings,
            grid=grid
        )
        print("ESP computed")
    else:
        esp = np.array(esp_calculator.assign_esp(
            monopoles=variables_dictionary['MBIS CHARGES'],
            dipoles=variables_dictionary['MBIS DIPOLE'].reshape(-1, 3),
            quadropules=variables_dictionary['MBIS QUADRUPOLE'].reshape(-1, 3, 3),
            grid=grid,
            coordinates=openff_conformer
        )[0]) * (unit.hartree / unit.e)
        print('ESP is:')
        print(esp)
        electric_field = np.array([]) * (unit.hartree / (unit.bohr * unit.e))

    E = item.properties['current energy']
    print("Producing Record:")
    record = MoleculePropRecord.from_molecule(
        molecule=openff_molecule,
        conformer=openff_conformer,
        grid_coordinates=grid,
        esp=esp,
        electric_field=electric_field,
        esp_settings=esp_settings,
        variables_dictionary=variables_dictionary,
        energy=E
    )

    if return_store:
        print('Returning record!')
        return record




def main():
    USERNAME = ""
    PASSWORD = ""

    client = PortalClient("http://127.0.0.1:7777")#, username=USERNAME,password=PASSWORD)   
    grid_settings = LatticeGridSettings(
        type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
    )
    prop_store = MoleculePropStore("./ESP_rebuilt_wb97.db")
    molecules = [record for record in client.query_records(dataset_id=13)]

    missing_entries = ['OCCN(CCO)c1nc(-c2ccccc2)c(-c2ccccc2)o1',
    'C[N+](C)(C)CCCCCCCCCC[N+](C)(C)C',
    'CC(=O)OC[C@@H]1O[C@H](n2ncc(=O)[nH]c2=O)[C@@H](OC(C)=O)[C@@H]1OC(C)=O',
    'COC(=O)C1C(C)=NC(C)=C(C(=O)OCC(C)C)[C@@H]1c1ccccc1[N+](=O)[O-]',
    '[NH3+]CC[NH2+]CC[NH2+]CC[NH3+]',
    'COc1ccnc(C[S@@](=O)c2nc3cc(OC(F)F)ccc3[nH]2)c1OC']
    
    missing_molecules = []

    for item in molecules:
        qc_mol = item.molecule
        qc_data = qc_mol.dict()
        qc_data['fix_com'] = True
        qc_data['fix_orientation'] = True
        qc_mol = QCMolecule.from_data(qc_data)
        openff_mol= Molecule.from_qcschema(qc_mol, allow_undefined_stereo=True)
        smi = openff_mol.to_smiles()
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon in missing_entries:
            print(openff_mol.to_smiles())
            missing_molecules.append(item)

    print(len(molecules))

    with ProcessPoolExecutor(
        max_workers=1, mp_context=get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                process_item_function,
                molecule,
                None,  # exclude_keys
                True,  # qm_esp
                True,  # compute_properties
                True,  # return_store
                grid_settings,
                None,  # Replace `None` with an instance of ESPCalculator if necessary
                dft_grid_settings,
                construct_variables_dictionary,
                build_grid,
                compute_properties,
            )
            for molecule in missing_molecules
        ]
        # to avoid simultaneous writing to the db, wait for each calculation to finish then write
        for future in tqdm(as_completed(futures), total=len(futures)):
            print('storing records')
            try:
                esp_record = future.result(timeout=600)
                prop_store.store(esp_record)

                # Store the record or process it further as needed
            except Exception as e:
                print(f"store or calc failed due to {e}")
                print('printing traceback:')
                traceback.print_exc()  # This will print the full traceback
                print(traceback.format_exc())  # This will print the full traceback as a string

if __name__ == "__main__":
    main()
