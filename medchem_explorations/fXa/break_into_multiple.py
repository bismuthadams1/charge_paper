from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def main():

    input_sdf = "optimised_molecules.sdf"

    supplier = Chem.SDMolSupplier(input_sdf,removeHs=False)

    for i, mol in enumerate(supplier):
        if mol is None:
            continue  
        smiles = mol.GetProp("Smiles") if mol.HasProp("Smiles") else None
        if not smiles:
            print(f"Skipping molecule {i+1} due to missing SMILES")
            continue

        smiles_mol = Chem.MolFromSmiles(smiles)
        if not smiles_mol:
            print(f"Failed to parse SMILES for molecule {i+1}")
            continue

        formula = rdMolDescriptors.CalcMolFormula(smiles_mol)
        output_sdf = f"structure_{i+1}_{formula}.sdf"
        writer = Chem.SDWriter(output_sdf)
        writer.write(mol)
        writer.close()

    print("SDF file has been split into separate files with molecular formula in filenames.")


if __name__ == "__main__":
    main()