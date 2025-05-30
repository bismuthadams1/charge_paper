QCArchive does not support the calculation of electronic properties for post-Hartree–Fock methods such as CCSD. To address this, we use our bespoke program, **ChargeCraft**, to perform these calculations.

### Files in this subdirectory

- **`properties_example.py`**  
  Example script demonstrating how to run a ChargeCraft calculation on a small set of compounds to generate CCSD-level electronic properties. The script reads input molecules from the `set.smi` file.

- **`set.smi`**  
  SMILES file containing the molecules to be processed.
