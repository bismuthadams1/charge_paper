
from qcportal import PortalClient
from qcportal.singlepoint import QCSpecification, SinglepointDataset
from openff.qcsubmit.common_structures import QCSpec, SCFProperties
from qcelemental.models.results import WavefunctionProtocolEnum
from openff.qcsubmit.results import OptimizationResultCollection
from openff.qcsubmit.results.filters import ElementFilter

def main():
    scf_properties = [
        SCFProperties.Dipole, 
        SCFProperties.LowdinCharges, 
        SCFProperties.MayerIndices, 
        SCFProperties.MBISCharges, 
        SCFProperties.MullikenCharges, 
        SCFProperties.WibergLowdinIndices,
        SCFProperties.Quadrupole
        ]

    OPT_DATASET = "conformer-test"
    # for functional in functionals:
        # for basis in basis_sets:
    spec = QCSpec(
        method="WB97X-D",
        basis="def2-tzvpp",
        program="psi4",
        spec_name="WB97X-D-def2-tzvpp",
        spec_description="",
        store_wavefunction=WavefunctionProtocolEnum.orbitals_and_eigenvalues,
        scf_properties=scf_properties,
        keywords= {
        "dft_spherical_points": 590,
        "dft_radial_points": 99}
    )
    
    client = PortalClient(address="http://10.64.1.130:7778", username="charlie", password="kuano123")
    result_dataset = OptimizationResultCollection.from_server(client=client, datasets=OPT_DATASET, spec_name="AIMNET2")
    

    basic_dataset = result_dataset.create_basic_dataset(
        dataset_name="Flexible set wb97xd",
        description="Flexible set wb97xd",
        tagline="ESP DATASET",
        driver="energy",
        qc_specifications=[spec]
    )
  
    # submit the calculations
    basic_dataset.submit(client)

if __name__ == "__main__":
    main()