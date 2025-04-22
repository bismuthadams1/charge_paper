from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def main():

    # Specify the input SDF file
    input_sdf = "optimised_molecules.sdf"

    # Read the molecules from the SDF file
    supplier = Chem.SDMolSupplier(input_sdf,removeHs=False)

    # Loop through each molecule and write it to a separate SDF file
    for i, mol in enumerate(supplier):
        if mol is None:
            continue  # Skip any molecules that failed to parse

        # Extract the SMILES string from the molecule properties
        smiles = mol.GetProp("Smiles") if mol.HasProp("Smiles") else None
        if not smiles:
            print(f"Skipping molecule {i+1} due to missing SMILES")
            continue

        # Generate molecular formula from the SMILES string
        smiles_mol = Chem.MolFromSmiles(smiles)
        if not smiles_mol:
            print(f"Failed to parse SMILES for molecule {i+1}")
            continue

        formula = rdMolDescriptors.CalcMolFormula(smiles_mol)

        # Create a new SDF writer with molecular formula in the filename
        output_sdf = f"structure_{i+1}_{formula}.sdf"
        writer = Chem.SDWriter(output_sdf)
        writer.write(mol)
        writer.close()

    print("SDF file has been split into separate files with molecular formula in filenames.")


if __name__ == "__main__":
    main()