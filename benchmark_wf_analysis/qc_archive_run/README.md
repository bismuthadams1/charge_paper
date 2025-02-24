This subfolder contains all the necassary scripts to replicate the CCSD benchmark. 
The following files are present in the subfolder:

1. *benchmark_data.csv* -  csv file containing all the necassary benchmark data including the geometries and properties associated with
the calculations.

2. *benchmark_review.ipynb* - jupyternotebook containing all the scripts to produce the graphs in the benchmarking section.

The following subfolders are contained in this directory:

1. *make_esp_db* - script for making the ESP database from the qcarchive info

2. *producing_ccsd_data* - chargecraft (package linked in main README) script for running the CCSD calculations from the geometries 
in the CCSD database. 