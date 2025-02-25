"""Example usage of chargecraft to produce CCSD data which is used to benchmark
the DFT functionals. Unfortunately, qcarchive does not currently support 
post-HF properties hence the need for chargecraft here. All the data is stored in a local
sqlite file.
"""

from chargecraft.inputsetup.SmilesInputs import ReadInput
from chargecraft.optimize.esp_generator_wrapper import PropGenerator
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.toolkit.topology import Molecule
from openff.recharge.grids import LatticeGridSettings
from chargecraft.storage.data_classes import ESPSettings, PCMSettings, DDXSettings
from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from chargecraft.globals import GlobalConfig
from qcportal import PortalClient


def main():

    GlobalConfig.memory_allocation = 460e+9
    #CCSD Options
    extra_options = {"CCENERGY__CACHELEVEL":0,
                     "GLOBALS__PRINT":2}

    #Read the .smi input and add to list
    smiles = ReadInput.read_smiles('set.smi') #add additionally smiles here

    # Define the grid that the electrostatic properties will be trained on and the
    # level of theory to compute the properties at.
    grid_settings = LatticeGridSettings(
        type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
    )
    #Grab geometries from qcfractal
    client = PortalClient("")  #Add qcarchive address here. 

    items = [record for record in client.query_records(dataset_id=1)] #change databse ID as appropriate
    geom_dict = {}
    molecules = [Molecule.from_qcschema(item.molecule).to_smiles(explicit_hydrogens = False) for item in items]

    for mol in set(molecules):
      index = molecules.index(mol)
      geom_dict[mol] = Molecule.from_qcschema(items[index].molecule).conformers[0]
    print(geom_dict)

    esp_settings_CCSD = ESPSettings(basis="aug-cc-pvtz", method="ccsd",  grid_settings=grid_settings) 

    prop_store = MoleculePropStore("./benchmark.db")
    #Loop through molecules
    for mol in smiles:
        molecule = smiles_to_molecule(mol)
        mol = molecule.to_smiles(explicit_hydrogens=False)
        print(mol)
        conformer = geom_dict[mol]
        #Generate the conformers
        conformer_list_hf = [conformer]
        molecule.add_conformer(conformer)
        #ccsd
        ESP_gen_CCSD = PropGenerator(
            molecule=molecule,
            conformers=conformer_list_hf,
            esp_settings=esp_settings_CCSD,
            grid_settings=grid_settings,
            prop_data_store=prop_store,
            optimise_with_ff = False,
            geom_opt=False,
            optimise_in_method=False
        )
        ESP_gen_CCSD.run_props(extra_options=extra_options)
   
if __name__ == "__main__":
    main()
