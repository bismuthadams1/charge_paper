from qcportal import PortalClient
from openff.toolkit.topology import Molecule
from chargecraft.storage.qcarchive_transfer import QCArchiveToLocalDB
from openff.recharge.grids import LatticeGridSettings
from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore

client = PortalClient("http://127.0.0.1:7777")  
# Define the grid that the electrostatic properties will be trained on and the
# level of theory to compute the properties at.
grid_settings = LatticeGridSettings(
    type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0)

db = MoleculePropStore(database_path='./conformers.db')

db_run =  QCArchiveToLocalDB(qc_archive=client, prop_data_store=db, grid_settings=grid_settings)
db_run.build_db(dataset_id=13)