Here we are comparing a number of charge models across a range of conformers. 
The following files are in this folder:

1. *build_charge_models.py*: here we build the local parquet dataset which calculates elecrostatic properties 
across different charge models. This is build from the local qc_archive dataset.

2. *build_esps.py*: here we the ESPs from the local qcarchive instance.

3. *drug_filter.ipynb*: here we prepare which drugs will be selected from the *fda_prop.xls* file (https://github.com/ericminikel/cnsdrugs). These are filtered by
rotatable bonds.

The following subdirectories are in this folder:

1. *qc_archive_run*: this directory specifically supports the production of a dataset which compares the ESPs of different conofrmers
using the DFT method used in training the charge models. Additionally, ESPs of the new charge models are build here and compared with the QM data. 

2. **