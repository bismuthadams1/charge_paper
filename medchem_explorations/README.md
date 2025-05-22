This subdirectory attempts to replicate the results produce by the ESP_DNN model reported in [this paper](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b01129). 

The study involves two biological targets: **factor Xa (fXa)** and **XIAP**. In both cases, a set of ligands is optimized within the corresponding protein binding pocket, and partial charges are then assigned to the optimized geometries.

To perform ligand optimization, the external package [FEgrow](https://github.com/cole-group/FEgrow) must be installed.

### Files in this subdirectory

1. **`fXa/`**  
   Contains docking and optimization workflows for ligands targeting fXa. Produces `.sdf` files and assigns partial charges to the resulting structures.

2. **`XIAP_opt/`**  
   Contains docking and optimization workflows for ligands targeting XIAP. Similar to the fXa directory, this produces `.sdf` files and assigns partial charges.
