import openff.nagl
import polars as pl
from openff.toolkit.topology import Molecule
import pandas as pd
import numpy as np
from openff.units import unit
from tqdm import tqdm
from openff.nagl import GNNModel
import pyarrow

AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge
HA_TO_KCAL_P_MOL =  627.509391  # Hartrees to kilocalories per mole

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

    distance = np.linalg.norm(displacement, axis=-1)#*unit.bohr # N, M
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


def elf10_charges(molecule: Molecule) -> pl.DataFrame:
    """
    Calculate ELF10 charges for a molecule.
    """

    molecule.assign_partial_charges(
        partial_charge_method='am1bccelf10',
    )
    # print('charges')
    # print(molecule.partial_charges)

    return molecule.partial_charges

def main(output, input_file):

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
        ('espaloma_esp_rms', pyarrow.float64()),
        ('resp_esp_rms', pyarrow.float64()),
        ('mbis_esp_rms', pyarrow.float64()),
        ('molecule', pyarrow.string()),
        ('geometry',pyarrow.list_(pyarrow.float64())),
        ('grid', pyarrow.list_(pyarrow.list_(pyarrow.float64()))),
        ('qm_esp', pyarrow.list_(pyarrow.float64())),
        ('riniker_esp_rms',pyarrow.float64()),
        ('elf10_charges', pyarrow.list_(pyarrow.float64())),
        ('elf10_dipole', pyarrow.float64()),
        ('elf10_esp_rmse', pyarrow.float64()),
        ('nagl_ash_charges', pyarrow.list_(pyarrow.float64())),
        ('nagl_ash_dipole', pyarrow.float64()),
        ('nagl_ash_esp_rmse', pyarrow.float64()),
    ])

    print('scanning parquet file')
    pldf = pl.scan_parquet(input_file).collect(engine='streaming')
    print('parquet file scanned')
    df = pldf.to_pandas()
    dir_path = openff.nagl_models.get_nagl_model_dirs_paths()[0]
    nagl_model = GNNModel.load(str(dir_path) + "/openff-gnn-am1bcc-0.1.0-rc.3.pt")
    
    MAX_ROWS = 1000

    with pyarrow.parquet.ParquetWriter(where=output, schema=schema, compression='snappy') as writer:

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df),
                        desc="Calculating additional models"):
            row_NEW = {}
            molecule = Molecule.from_mapped_smiles(row["molecule"], allow_undefined_stereo=True)
            molecule.add_conformer(
                row["geometry"].reshape((-1, 3)) * unit.angstrom,        
            )

            grid = np.array([sec.tolist() for sec in row['grid'].tolist()])

            efl10_charges = elf10_charges(molecule=molecule)
            row_NEW["elf10_charges"] = efl10_charges.magnitude.flatten().tolist()

            elf10_dipole = calculate_dipole_magnitude(
                charges=efl10_charges,
                conformer= molecule.conformers[0]
            )
            # print(f"elf10 dipole: { elf10_dipole.flatten().tolist()[0]}")
            row_NEW["elf10_dipole"] = elf10_dipole.flatten().tolist()[0]

            elf10_esp = calculate_esp_monopole_au(
                grid_coordinates=grid*unit.angstrom,
                atom_coordinates=molecule.conformers[0],
                charges=efl10_charges 
            )
            qm_esp = row["qm_esp"] * AU_ESP

            elf10_esp_rmse = (((elf10_esp - qm_esp) ** 2).mean() ** 0.5).magnitude * HA_TO_KCAL_P_MOL
            row_NEW["elf10_esp_rmse"] = float(elf10_esp_rmse)

            # row_NEW["elf10_esp"] = elf10_esp

            nagl_model_charges = nagl_model.compute_properties(
            molecule=molecule       
            )['am1bcc_charges']
            row_NEW["nagl_ash_charges"] = nagl_model_charges

            nagl_dipole = calculate_dipole_magnitude(
                charges=nagl_model_charges,
                conformer=molecule.conformers[0]
            )
            # print(f"nagl dipole: { nagl_dipole.flatten().tolist()[0]}")
            row_NEW["nagl_ash_dipole"] = nagl_dipole.flatten().tolist()[0]

            nagl_esp = calculate_esp_monopole_au(
                grid_coordinates=grid*unit.angstrom,
                atom_coordinates=molecule.conformers[0],
                charges=nagl_model_charges * unit.e
            )
            nagl_esp_rmse = (((nagl_esp - qm_esp) ** 2).mean() ** 0.5).magnitude * HA_TO_KCAL_P_MOL
            print('nagl_esp_rmse:', nagl_esp_rmse)
            row_NEW["nagl_ash_esp_rmse"] = float(nagl_esp_rmse)

            row_NEW = {**row_NEW, **row.to_dict()}  # Merge dictionaries
            rows.append(row_NEW)
            if len(rows) >= MAX_ROWS:
                print(f"Writing {len(rows)} rows to Parquet file...")
                # Write the batch to the Parquet file
                batch = pyarrow.RecordBatch.from_pylist(
                    rows, schema=schema
                )
                writer.write_batch(batch)
                rows = []

    if rows:
        print(f"Writing remaining {len(rows)} rows to Parquet file...")
        batch = pyarrow.RecordBatch.from_pylist(rows, schema=schema)
        writer.write_batch(batch)

if __name__ == "__main__":
    input_file = "./charge_models_test_withgeoms.parquet"
    output = "./charge_models_test_withgeoms_and_additional_models_test.parquet"
    main(output=output, input_file=input_file)  
        


