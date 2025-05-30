This subdirectory contains all the necessary scripts and data to replicate the CCSD benchmark study.

### Files in this subdirectory

- **`benchmark_data.csv`**  
  CSV file containing benchmark data, including molecular geometries and the associated electronic properties.

- **`benchmark_review.ipynb`**  
  Jupyter notebook used to generate the figures and analysis presented in the benchmarking section.

### Subdirectories

- **`make_esp_db/`**  
  Scripts for generating an ESP database from QCArchive outputs.

- **`producing_ccsd_data/`**  
  Contains scripts for running CCSD-level calculations using the ChargeCraft package (linked in the main README). These scripts operate on geometries stored in the CCSD database.

- **`qc_archive_run/`**  
  Example scripts demonstrating how to perform DFT calculations using the QCArchive infrastructure.
