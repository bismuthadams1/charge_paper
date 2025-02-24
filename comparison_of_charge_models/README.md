Here we comparse a number of charge models over 50K compounds. This section uses the ChargeAPI module built by the authors 
of this paper https://github.com/bismuthadams1/ChargeAPI. 
The files in this paper do the following:

1. *build_charge_models.py*: this builds a parquet file of molecules+conformers with electrostatic properties calculated across
the different charge models.

2. *build_esps.py*: here we pull the 50K compounds from qcarchive and build the ESPs from the wavefunction information.

3. *plot_charge_results.ipynb*: here we plot a comparison of the charge models from the calculated parquet set.

4. *plot_dipole_and_esp_results.ipynb*: here we plot the dipole and ESP comparison across different charge models.