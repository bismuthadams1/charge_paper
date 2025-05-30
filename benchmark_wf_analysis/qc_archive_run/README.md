This subdirectory contains all necessary scripts to prepare a local QCArchive database using various DFT functionals and basis sets.

### Scripts in this subdirectory

- **`make_geoms.py`**  
  Optimizes molecular geometries to be used for subsequent single-point energy calculations.

- **`make_functionals.py`**  
  Generates a database of molecules with electronic properties calculated using different DFT functionals and basis sets, via single-point calculations.

- **`prepare_data.ipynb`**  
  Jupyter notebook for monitoring the progress and status of local QCArchive calculations.

- **`produce_db.py`**  
  Extracts results from the local QCArchive instance and stores them in a local SQL database.

### Subdirectories

- **`make_workers/`**  
  Contains scripts to launch worker processes for running QCArchive calculations.  
  To use this setup, you must first create a compatible Conda environment using the configuration provided here:  
  [qcarchive-worker-openff-psi4.yaml](https://github.com/openforcefield/qca-dataset-submission/blob/master/devtools/prod-envs/qcarchive-worker-openff-psi4.yaml)
