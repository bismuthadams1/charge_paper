Here we try and replicate the results present in [this]{https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b01129} paper. 
There are two biological targets here: fXa and XIAP. In both cases we optimize a set of given ligands in the context
of the target pockets. Then we assign partial charges based on these optimized geometries. In order for the ligands to be optimized, a separate package should be installed called [FEGrow]{https://github.com/cole-group/FEgrow}.

### Files in this subdirectory

1. **fXa**: here we dock the ligands to fXa, produce a series of sdfs, and then assign the partial charges to these molecules.
2. **XIAP_opt**: here we dock the ligands to XIAP, produce a series of sdfs, and then assign the partial charges to these molecules.