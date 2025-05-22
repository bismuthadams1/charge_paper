# Code for: A graph neural network charge model targeting accurate electrostatic properties of organic molecules


**Authors: C. Adams, J.T. Horton, L. Wang, D.L. Mobley, D. W. Wright, D. J. Cole**

## Subdirectories Overview

Each subdirectory contains code and data related to a specific study:

- **benchmark_wf_analysis**  
  Scripts to run DFT and CCSD calculations, along with notebooks for data analysis and plotting.

- **benchmarking_new_charge_models**  
  Code for generating training, validation, and test loss metrics for new charge models.

- **comparison_of_charge_models**  
  Tools to compare various charge models by evaluating partial charges, dipoles, and electrostatic potentials (ESPs). Includes scripts to generate QM ground truth data, run charge models, and perform analyses.

- **compare_fda_drugs**  
  Comparison of half-polarised machine learning charges and AM1-BCC charges on FDA-approved drugs.

- **exploring_the_dataset**  
  Scripts to examine the chemical features of the dataset used to train the new charge model.

- **medchem_explorations**  
  Exploratory analysis of partial charges from the new charge model in the context of two biological targets.

- **producing_esp_from_mbis**  
  Test for evaluating the error introduced when reconstructing ESPs using the MBIS multipole expansion up to the quadrupole level.

- **timing_test**  
  Benchmarking the runtime of the new charge model against AM1-BCC on carbon chains of increasing length.


## Installing the required environments

### Charge Model Env

For most scripts here a `charge_model_env` has been created. This can be installed:

`conda env create -n charge_model_env --file charge_model_env_minimal.yml`

Activate the conda environment, then finally install the nagl-mbis models:

`pip install git+https://github.com/bismuthadams1/nagl-mbis --no-build-isolation`

### QCArchive Environments

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