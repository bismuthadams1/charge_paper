In this subdirectory we build the local qcarchive instance which builds the data from different QM methods and compares
the ESPs generated with the new charge models and AM1BCC. An important thing to note here is here we take the average AM1BCC charges across a conformers set, this
is because the charge model we built is conformer agnostic and the AM1BCC charges are usually assigned over an ensemble of conformers. 
To generate the data for this model, use the same *build_charge_models.py*
The following files are in this subdirectory: 

1. *build_charge_models.py*: add the charge model info to the geometries from the local qcarchive instance. 

2. *explore_parquet.ipynb*: