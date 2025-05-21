This subfolder contains all the necassary scripts to prepare a qc_archive database of DFT functionals and basis sets.
The scripts here are:

1. *make_functionals.py*: make a database of molecules with electronic properties calculated with different DFT functionals
and basis sets as a set of singlepoint calculations. 

2. *make_geoms.py*: this should be run before `make_functionals.py` file to produce the optimized geometries for singlepoint calculations. 

3. *prepare_data.ipynb*: use this jupyter notebook to monitor how the local qcarchive calculations are proceeding. 

4. *produce_db.py*: produce a local sql database from the local qc_archive database.

The following subdirectories are present here:

1. *make_workers*: this subdirectory contains all the scripts to launch a set of workers to run calculations in the qcarchive database. 
To run the qcarchive calculations you must create a separate environment here:
 https://github.com/openforcefield/qca-dataset-submission/blob/master/devtools/prod-envs/qcarchive-worker-openff-psi4.yaml

 