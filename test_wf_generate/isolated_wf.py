import time

# Your existing imports
from qcportal import PortalClient
from chargecraft.storage.qcarchive_transfer import QCArchiveToLocalDB
from qcportal import PortalClient
from qcportal.record_models import RecordQueryIterator
from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from openff.recharge.grids import LatticeGridSettings
from openff.recharge.esp.qcresults import reconstruct_density, compute_esp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, get_context
from tqdm import tqdm
from openff.recharge.esp import DFTGridSettings
from chargecraft.storage.data_classes import ESPSettings, PCMSettings, DDXSettings
from qcportal.singlepoint.record_models import SinglepointRecord
import pickle
import numpy
from openff.units import unit
import psi4
import tempfile
import os

client = PortalClient("api.qcarchive.molssi.org")

# prop_store = MoleculePropStore("/mnt/storage/nobackup/nca121/QC_archive_50K_esp_gen/async_chargecraft/ESP_rebuilt.db")

grid_settings = LatticeGridSettings(
    type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
)

records: RecordQueryIterator = client.query_records(
    record_id="96507636",
)

record: SinglepointRecord =  next(records)

density = reconstruct_density(wavefunction=record.wavefunction, n_alpha=record.properties['calcinfo_nalpha'])
dft_grid_settings = None
#HF will not contain this keyword, use default grid settings
dft_grid_settings = DFTGridSettings.Default

esp_settings = ESPSettings(
    basis = record.specification.basis,
    method = record.specification.method,
    grid_settings = grid_settings,
    #TODO update PCMSettings if use. Fix radii set and solvent choices in the database
    pcm_settings = PCMSettings(
        solver = '', 
        solvent = '',
        radii_model = '',
        radii_scaling = '',
        cavity_area = ''
    ) if 'PCM' in record.specification.keywords else None,
    ddx_settings = DDXSettings(
        solvent = None if not 'ddx_solvent_epsilon' in record.specification.keywords
        else record.specification.keywords['ddx_solvent'],
        epsilon = record.specification.keywords['ddx_solvent_epsilon'] 
        if 'ddx_solvent_epsilon' in record.specification.keywords is not None else None,
        radii_set = 'uff',
        ddx_model = record.specification.keywords['ddx_model'].upper() 
        if record.specification.keywords['ddx_model'] is not None else None)
        if 'ddx' in record.specification.keywords else None,
    psi4_dft_grid_settings = dft_grid_settings
)

qc_molecule = record.molecule

# psi4.core.be_quiet()

psi4_molecule = psi4.geometry(qc_molecule.to_string("psi4", "angstrom"))
psi4_molecule.set_name("regen_mol")
mol_name = psi4_molecule.name()
psi4_molecule.reset_point_group("c1")

psi4_wavefunction = psi4.core.RHF(
    psi4.core.Wavefunction.build(psi4_molecule, esp_settings.basis),
    psi4.core.SuperFunctional(),
)

wavefunction = psi4_wavefunction.Da().copy(psi4.core.Matrix.from_array(density))
print(wavefunction)

start_time_2 = time.time()

# with tempfile.TemporaryDirectory() as tmpdir:
wfn_file_path = "wfn.npy"
print(os.listdir())
pid = os.getpid()
# psi4.core.Wavefunction.from_file('wfn')
scratchdir1=psi4.core.IOManager.shared_object().get_default_path()
f_name = scratchdir1+'stdout.'+'{}'.format(mol_name)+'.' +'{}'.format(pid)+'.'+str(180)+'.npy'
psi4_wavefunction.to_file(f_name)


psi4.set_options({"guess":"read"})
# E, wfn =  psi4.prop(f'{esp_settings.method}/{esp_settings.basis}', 
#                     properties=[
#                     "MULLIKEN_CHARGES", 
#                     "LOWDIN_CHARGES", 
#                     "DIPOLE", 
#                     "QUADRUPOLE", 
#                     "MBIS_CHARGES"], 
#                     molecule = psi4_molecule,
#                     return_wfn = True )
                
psi4.oeprop(psi4_wavefunction,"DIPOLE",
                        "QUADRUPOLE", 
                        "MULLIKEN_CHARGES",
                        "LOWDIN_CHARGES",
                        "MBIS_CHARGES",
                        "MBIS_DIPOLE",
                        "MBIS_QUADRUPOLE")
end_time_2 = time.time()

print(f'all WF variables for {esp_settings.method}')
print(psi4_wavefunction.variables())
print('memory use before wfn interaction')

variables_dictionary = dict()
#psi4 computes charges in a.u., elementary charge
variables_dictionary["MULLIKEN_CHARGES"] = psi4_wavefunction.variable("MULLIKEN_CHARGES") * unit.e
variables_dictionary["LOWDIN_CHARGES"] = psi4_wavefunction.variable("LOWDIN_CHARGES") * unit.e
variables_dictionary["MBIS CHARGES"] = psi4_wavefunction.variable("MBIS CHARGES") * unit.e
#psi4 grab the MBIS multipoles
variables_dictionary["MBIS DIPOLE"] = psi4_wavefunction.variable("MBIS DIPOLES") * unit.e * unit.bohr_radius
variables_dictionary["MBIS QUADRUPOLE"] = psi4_wavefunction.variable("MBIS QUADRUPOLES") * unit.e * unit.bohr_radius**2
variables_dictionary["MBIS OCTOPOLE"] = psi4_wavefunction.variable("MBIS OCTUPOLES") * unit.e * unit.bohr_radius**3
variables_dictionary["DIPOLE"] = psi4_wavefunction.variable(f"{esp_settings.method.upper()} DIPOLE") * unit.e * unit.bohr_radius
variables_dictionary["QUADRUPOLE"] = psi4_wavefunction.variable(f"{esp_settings.method.upper()} QUADRUPOLE") * unit.e * unit.bohr_radius**2
variables_dictionary["ALPHA_DENSITY"] = psi4_wavefunction.Da().to_array()
variables_dictionary["BETA_DENSITY"] = psi4_wavefunction.Db().to_array()

print('memory use after wfn interaction')
psi4.core.clean()

print('variable dict from prop')
print(variables_dictionary)

print(f"Portion 2 took {end_time_2 - start_time_2:.4f} seconds")
