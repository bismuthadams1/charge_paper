# Charge Paper Code

The subdirectories pertain to the following studies:

* **producing_esp_from_mbis**: a test to see what the errors are when rebuilding the ESPs from the mbis multipole expansion
up to the quadrupole
* **comparison_of_charge_models**: code which tests a number of charge models by comparing partial charges, dipoles and esps. The directory contains scripts to produce the QM ground truth, produce the charge model data, and analyse the data. 
* **benchmarking_new_charge_models**: code for creating test/train/val loss.

* **benchmark_wf_analysis** code to run the DFT and CCSD calculations and notebooks to analyse and plot the data. 

* **compare_fda_drugs** comparison of the half-polarised ML charges and AM1-BCC calculations. 

* **timing_test** time the charge model against AM1-BCC for an increasingly lengthening carbon chain

* ** **

# Installing the required environments

## Charge Model Env

For most scripts here a `charge_model_env` has been created. This can be installed:

`conda env create -n charge_model_env --file charge_model_env_minimal.yml`

Activate the conda environment, then finally install the nagl-mbis models:

`pip install git+https://github.com/bismuthadams1/nagl-mbis --no-build-isolation`

## QCArchive Environments

## FeGrow Environments