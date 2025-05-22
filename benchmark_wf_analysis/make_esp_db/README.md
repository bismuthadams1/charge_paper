This subdirectory contains scripts for rebuilding electrostatic potentials (ESPs) from a local QCArchive database.

### Files in this subdirectory

- **`produce_db.py`**  
  Script to generate a local database of ESPs from an existing QCArchive instance.

**Note:** A van der Waals radius of 2.00 Å is used for boron in this setup. This requires a manual modification in the `openff-recharge` package.
