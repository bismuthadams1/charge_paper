# Code for: A Graph Neural Network Charge Model Targeting Accurate Electrostatic Properties of Organic Molecules

**Authors:** C. Adams, J.T. Horton, L. Wang, D.L. Mobley, D. W. Wright, D. J. Cole

---

## 📁 Subdirectories Overview

Each subdirectory contains code and data related to a specific study:

- **`benchmark_wf_analysis/`**  
  Scripts to run DFT and CCSD calculations, along with notebooks for data analysis and plotting.

- **`benchmarking_new_charge_models/`**  
  Code for generating training, validation, and test loss metrics for the new charge models.

- **`comparison_of_charge_models/`**  
  Tools to compare various charge models by evaluating partial charges, dipoles, and electrostatic potentials (ESPs). Includes scripts to generate QM ground truth data, apply charge models, and analyze results.

- **`compare_fda_drugs/`**  
  Comparison of half-polarised machine learning charges with AM1-BCC charges across FDA-approved drugs.

- **`conformer_test/`**  
  Code to evaluate how existing and new charge models capture changes in electronic properties across molecular conformations.

- **`exploring_the_dataset/`**  
  Scripts to explore chemical features in the dataset used for training, validation, and testing.

- **`medchem_explorations/`**  
  Exploratory analysis of partial charges from the new charge model for two biological targets.

- **`producing_esp_from_mbis/`**  
  Evaluation of the error introduced when reconstructing ESPs using the MBIS multipole expansion up to the quadrupole level.

- **`timing_test/`**  
  Benchmarking the runtime of the new charge model against AM1-BCC on carbon chains of increasing length.

---

## ⚙️ Installing the Required Environments

### Charge Model Environment

Most scripts require a Conda environment called `charge_model_env`. You can install it using:

```bash
conda env create -n charge_model_env --file charge_model_env_minimal.yml
```

Then activate the environment and install the **nagl-mbis** model:

```bash
pip install git+https://github.com/bismuthadams1/nagl-mbis --no-build-isolation
```

---

### QCArchive Environments

For generating quantum mechanical data, we use [QCFractal](https://github.com/MolSSI/QCFractal). Two environments are needed:

1. **Database Environment**  
   This environment is required to launch the QCArchive database. Setup instructions and the environment YAML can be found here:  
   [QCArchive Admin Guide](https://docs.qcarchive.molssi.org/admin_guide/index.html)

2. **Psi4 Worker Environment**  
   Needed to run Psi4 calculations and submit them to the database. The environment YAML is available at:  
   [qcarchive-worker-openff-psi4-ddx.yaml](https://github.com/openforcefield/qca-dataset-submission/blob/master/devtools/prod-envs/qcarchive-worker-openff-psi4-ddx.yaml)

---

## 🧬 FeGrow Environment (For Ligand Docking)

Notebooks involving ligand docking require the [FeGrow](https://github.com/cole-group/FEgrow) software. Please follow the setup instructions provided in their repository to create a suitable environment.
