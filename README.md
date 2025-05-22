# Charge Paper Code

The subdirectories pertain to the following studies:

* **producing_esp_from_mbis**: a test to see what the errors are when rebuilding the ESPs from the mbis multipole expansion
up to the quadrupole
* **comparison_of_charge_models**: code which tests a number of charge models by comparing partial charges, dipoles and esps. The directory contains scripts to produce the QM ground truth, produce the charge model data, and analyse the data. 
* **benchmarking_new_charge_models**: code for creating test/train/val loss.

* **benchmark_wf_analysis** code to run the DFT and CCSD calculations and notebooks to analyse and plot the data. 

* **compare_fda_drugs** comparison of the half-polarised ML charges and AM1-BCC calculations. 

* **timing_test** time the charge model against AM1-BCC for an increasingly lengthening carbon chain

* **medchem_explorations**: exploration of the partial charges with two biological targets with the new charge model.

* **exploring_the_dataset**: code to explore the chemical features across the dataset used to train the new model. 

# Installing the required environments

## Charge Model Env

For most scripts here a `charge_model_env` has been created. This can be installed:

`conda env create -n charge_model_env --file charge_model_env_minimal.yml`

Activate the conda environment, then finally install the nagl-mbis models:

`pip install git+https://github.com/bismuthadams1/nagl-mbis --no-build-isolation`

## QCArchive Environments

For generating the QM data here, QCFractal (`https://github.com/MolSSI/QCFractal`) tools were used. 
Two separate environments are required here. 

1. **database environment**: first we require an environment to launch a database. All details of how to set this database up and which environment to
install can be found here: `https://docs.qcarchive.molssi.org/admin_guide/index.html`.

2. **psi4 environment**: we also require an environment to run psi4 calculations and send to our database setup in step 1. This environment can be found here:
`https://github.com/openforcefield/qca-dataset-submission/blob/master/devtools/prod-envs/qcarchive-worker-openff-psi4-ddx.yaml`.

## FeGrow Environments

For notebooks producing the docked ligands, a separate environment for using the FeGrow software will be required.
Details on how to build this environment can be found here:

`https://github.com/cole-group/FEgrow`