Here we dock ligand structures to fXa provided in hte `pdb5c7a.pdb` file. The `ligandXIAP.pdb` file represents the starting ligand.
Within the `Docking.ipynb` notebook we can modify this ligand and then optimize it in the context of the protein pocket.
Then, `break_into_multiple.py` script will break up the `optimised_molecules.sdf` file. We can the plot the partial charges
with the new charge models in the `plot_partial_charges.ipynb`. Other sdf and pdb files provided are intermediates to this process. 