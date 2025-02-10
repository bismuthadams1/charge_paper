from qcportal import PortalClient
from openff.qcsubmit.factories import OptimizationDatasetFactory
from openff.qcsubmit import workflow_components
from openff.toolkit import Molecule
from openff.qcsubmit.datasets import load_dataset
from openff.qcsubmit.common_structures import QCSpec
from qcportal import PortalClient

def main():

    client = PortalClient(address="", username="", password="")

    molecules = [
    'c1ccccc1',
    'CC(=O)O',
    'CC1=CNc2c1cccc2',
    'C1=CSC=C1',
    'C1CC(=O)NC1',
    'C1=CSN=C1[O-]',
    'CC(C)F',
    'c1ccc(cc1)F',
    'c1cc(cnc1)Cl',
    '[B-](COC)(F)(F)F',
    'CC',
    'COC',
    'C1CC[NH2+]CC1',
    'c1ccc(cc1)[S-]',
    'c1ccc(cc1)C#N',
    'CP(=O)([O-])[O-]',
    'c1ccc(cc1)Cl',
    'COCC(F)(F)F',
    'C1=CNC=C1Br',
    'C[Si]1(CCCC1)C',
    'c1cc(c(c(c1)F)F)F',
    'CCO',
    'CC(=O)C',
    'B(O)(O)OCC',
    'CS(=O)(=O)[O-]',
    'c1ccc(cc1)Br',
    'C1=C[NH+]=CN1',
    'C1CCNC1',
    'CCSC',
    'CCS',
    'CO[Si](C)(C)O',
    'c1ccc(cc1)O',
    'C1CCOC1',
    'CC(=O)[O-]',
    'c1ccc(cc1)N',
    'c1ccc(cc1)I'
    ]
    mols = [
    Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    for smiles in molecules
    ]
    #Create factory for FF
    hf_factory = OptimizationDatasetFactory(
        qc_specifications={
            "HF": QCSpec(
                program="psi4",
                method="HF",
                basis="def2-svp",
                spec_name="hf-def2-svp",
                spec_description="hf specification"
            )
        }
    )
    
    #Conformer generator worklow
    conf_gen = workflow_components.StandardConformerGenerator(
        max_conformers=1,
        rms_cutoff=0,
        toolkit="rdkit"
    )
    
    hf_factory.add_workflow_components(conf_gen)
    
    #Create dataset for FF factory
    data = hf_factory.create_dataset(
        molecules=mols,
        dataset_name="hf_geoms",
        description="one conformers per molecule in the set",
        tagline="An example dataset."
    )

    data.submit(client)
    
if __name__ == "__main__":
    main()  
