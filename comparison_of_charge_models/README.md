This subdirectory contains scripts to benchmark a range of charge models on a dataset of 50,000 compounds. The analysis uses the **ChargeAPI** module, developed by the authors of the following repository:  
[https://github.com/bismuthadams1/ChargeAPI](https://github.com/bismuthadams1/ChargeAPI)

### Files in this subdirectory

- **`build_charge_models.py`**  
  Generates a Parquet file containing molecules and conformers with electrostatic properties computed using various charge models.

- **`build_esps.py`**  
  Retrieves 50,000 compounds from QCArchive and reconstructs their electrostatic potentials (ESPs) from wavefunction data.

- **`plot_charge_results.ipynb`**  
  Jupyter notebook to visualize and compare charge distributions from the different charge models.

- **`plot_dipole_and_esp_results.ipynb`**  
  Jupyter notebook to compare dipole moments and ESPs across the charge models.

- **`add_additional_models.py`**
  Add the additional ELF10 and AshGC models

- **`plot_additional_models.ipynb`**
  Jupyter notebook to plot the additional charge models