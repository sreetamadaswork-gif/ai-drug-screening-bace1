from rdkit import Chem
from rdkit.Chem import Descriptors

# A simple molecule: ethanol
smiles = "CCO"

# Convert SMILES into an RDKit molecule
mol = Chem.MolFromSmiles(smiles)

print("SMILES:", smiles)
print("Molecule created:", mol is not None)

# Calculate molecular weight
molecular_weight = Descriptors.MolWt(mol)

print("Molecular Weight:", molecular_weight)