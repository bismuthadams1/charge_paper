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

    OPT_DATASET = "hf_geoms"
    basis_sets = ["def2-tzvpp", "def2-tzvpd", "def2-tzvp", "def2-svpd","def2-tzvppd","6-31G*","6-311G*"]
    functionals = ["B3LYP","PBE0","TPSSH","WB97M-D3BJ","WB97X-D","HF"]
    NEW_SPECS = []
    for functional in functionals:
        for basis in basis_sets:
            spec = QCSpec(
                method=functional,
                basis=basis,
                program="psi4",
                spec_name=f"{functional}-{basis}",
                spec_description="",
                store_wavefunction=WavefunctionProtocolEnum.orbitals_and_eigenvalues,
                scf_properties=scf_properties,
                keywords= {
                "dft_spherical_points": 590,
                "dft_radial_points": 99}
            )
    
            NEW_SPECS.append(spec)
            
    print(NEW_SPECS)
    client = PortalClient(address="", username="", password="")
    # dataset: SinglepointDataset = client.get_dataset(dataset_type="singlepoint", dataset_name=DATASET_NAME)
    result_dataset = OptimizationResultCollection.from_server(client=client, datasets=OPT_DATASET, spec_name="HF")
    #element filter Iodine
    element_filter = ElementFilter(allowed_elements=["C","H","O","N","F","S","Cl","Br","Si","B"])
    result_dataset = result_dataset.filter(element_filter)

    basic_dataset = result_dataset.create_basic_dataset(
        dataset_name="DFT Functionals Database",
        description="A dataset composed of DFT functionals and basis sets",
        tagline="ESP DATASET",
        driver="energy",
        qc_specifications=NEW_SPECS
    )
  
    # submit the calculations
    basic_dataset.submit(client)

if __name__ == "__main__":
    main()