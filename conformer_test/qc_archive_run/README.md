This subdirectory contains the setup and analysis scripts for a local QCArchive instance. The instance is used to generate quantum mechanical (QM) data using various methods and to compare electrostatic potentials (ESPs) produced by the new charge models and AM1-BCC.

A key point in this analysis is that we average AM1-BCC charges across the conformer set. This is because the new charge model is **conformer-agnostic**, while AM1-BCC charges are typically assigned across an ensemble of conformers. 

Additionally, the QM reference data used to evaluate the new charge model is computed at its training level of theory (**ωB97X-D/def2-TZVPP**). In contrast, the AM1-BCC charges are compared to QM ESPs calculated at the **HF/6-31G*** level, to reflect the level of theory AM1-BCC was originally parameterized against. This ensures a fair comparison based on the QM targets that each method aims to reproduce.

To generate charge model data, use the `build_charge_models.py` script.

#### Files in this subdirectory:

1. **`build_charge_models.py`**  
   Adds charge model information to geometries stored in the local QCArchive instance.

2. **`submit_functionals.py`**  
   Submits QCFractal jobs to generate conformer data at the ωB97X-D/def2-TZVPP level.

3. **`explore_parquet.ipynb`**  
   Analyzes the generated Parquet files and produces ESP comparison plots across conformers.

### Datasets

The dataset relevant for running the notebooks in this subfolder can be found here: https://zenodo.org/records/15796721