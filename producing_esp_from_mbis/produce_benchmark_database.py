import chargecraft
from chargecraft.inputSetup.SmilesInputs import ReadInput
from chargecraft.optimize.esp_generator_wrapper import PropGenerator
from chargecraft.conformers.conformer_gen import Conformers
from openff.recharge.utilities.molecule import smiles_to_molecule
from openff.recharge.grids import LatticeGridSettings
from chargecraft.storage.data_classes import ESPSettings, PCMSettings
from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from chargecraft.globals import GlobalConfig


def main():

    molecules = [
    'CC(=O)O',
    'CO[Si](C)(C)O',
    'CP(=O)([O-])[O-]',
    'CC(C)=O',
    'CS(=O)(=O)[O-]',
    '[O-]c1ccsn1',
    '[S-]c1ccccc1',
    'CC',
    'CCO',
    'Clc1ccccc1',
    'CC(C)F',
    'c1ccsc1',
    'Fc1cccc(F)c1F',
    'C1CCNC1',
    'C1CC[NH2+]CC1',
    'O=C1CCCN1',
    'COC',
    'N#Cc1ccccc1',
    'Brc1cc[nH]c1',
    'C1CCOC1',
    'CCOB(O)O',
    'Nc1ccccc1',
    'Oc1ccccc1',
    'Clc1cccnc1',
    'Fc1ccccc1',
    'C[Si]1(C)CCCC1',
    'COCC(F)(F)F',
    'CC(=O)[O-]',
    'c1c[nH+]c[nH]1',
    'Cc1c[nH]c2ccccc12',
    'Brc1ccccc1',
    'c1ccccc1',
    'CCSC',
    'CCS',
    'COC[B-](F)(F)F',
    'Ic1ccccc1'
    ]
    GlobalConfig.num_threads_per_core = 1
    GlobalConfig.memory = 60e+9 
    GlobalConfig.ncores = 8

    # Define the grid that the electrostatic properties will be trained on and the
    # level of theory to compute the properties at.
    grid_settings = LatticeGridSettings(
        type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
    )
    esp_settings = ESPSettings(basis="def2-tzvpp", method="wb97x-d", grid_settings=grid_settings) #-D3BJ

    #Loop through molecules
    for mol in molecules:
        molecule = smiles_to_molecule(mol)
        #Generate the conformers
        conformer_list = Conformers.generate(molecule, generation_type='rdkit', max_conformers=1)

        ESP_gen = PropGenerator(molecule = molecule, 
                                conformers = conformer_list, 
                                esp_settings = esp_settings,
                                grid_settings = grid_settings,
                                prop_data_store = MoleculePropStore(database_path='properties_store_density.db'),
                                geom_opt= False)
  
        conformer_list_new  = ESP_gen.run_props()
        print(conformer_list_new)
        

if __name__ == "__main__":
    main()