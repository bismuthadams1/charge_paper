This subdirectory focuses on comparing a range of charge models across multiple conformers.

### Files in this subdirectory

- **`build_charge_models.py`**  
  Builds a local Parquet dataset of electrostatic properties computed across different charge models. The dataset is generated using molecules and conformers stored in the local QCArchive instance.

- **`build_esps.py`**  
  Extracts and processes electrostatic potentials (ESPs) from the local QCArchive database.

- **`drug_filter.ipynb`**  
  Jupyter notebook for selecting FDA-approved drugs (from [fda_prop.xls](https://github.com/ericminikel/cnsdrugs)) based on properties such as the number of rotatable bonds.

- **`combined.sdf`**  
  Structure file used to visually inspect whether geometries (e.g., AIMNet2-optimized) are chemically reasonable.

### Subdirectories

- **`qc_archive_run/`**  
  Contains scripts for generating a dataset that compares ESPs across different conformers. ESPs are calculated using the same DFT method used during charge model training. This directory also includes tools to compute and compare ESPs produced by the new charge models against QM data.

### Datasets

The dataset relevant to running the notebooks in this subfolder can be found here: https://zenodo.org/records/14925942
