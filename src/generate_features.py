import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/bace1_clean.csv"
OUTPUT_FILE = "data/processed/bace1_features.csv"

# Morgan fingerprint settings
RADIUS = 2
N_BITS = 2048


# ============================================================
# LOAD CLEAN DATASET
# ============================================================

print("=" * 60)
print("LOADING CLEAN DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "canonical_smiles" not in df.columns:
    raise ValueError(
        "ERROR: 'canonical_smiles' column not found in dataset."
    )


# ============================================================
# CONVERT SMILES → RDKit MOLECULES
# ============================================================

print("\nConverting SMILES to RDKit molecules...")

molecules = []

for smiles in df["canonical_smiles"]:

    mol = Chem.MolFromSmiles(smiles)

    molecules.append(mol)


# ============================================================
# CHECK INVALID MOLECULES
# ============================================================

invalid_count = sum(
    mol is None
    for mol in molecules
)

print("Invalid molecules:", invalid_count)


# ============================================================
# REMOVE INVALID MOLECULES
# ============================================================

valid_mask = [
    mol is not None
    for mol in molecules
]

df = df[valid_mask].reset_index(drop=True)

molecules = [
    mol
    for mol in molecules
    if mol is not None
]

print("Valid molecules:", len(molecules))


# ============================================================
# CREATE MORGAN FINGERPRINT GENERATOR
# ============================================================

print("\nCreating Morgan fingerprint generator...")

morgan_generator = (
    rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS
    )
)


# ============================================================
# GENERATE MORGAN FINGERPRINTS
# ============================================================

print("Generating Morgan fingerprints...")

fingerprints = []

for mol in molecules:

    fingerprint = (
        morgan_generator.GetFingerprint(mol)
    )

    fingerprints.append(
        np.array(fingerprint)
    )


# ============================================================
# CONVERT FINGERPRINTS TO NUMPY ARRAY
# ============================================================

X = np.array(fingerprints)

print("\nFingerprint matrix shape:", X.shape)


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

feature_columns = [
    f"FP_{i}"
    for i in range(X.shape[1])
]

X_df = pd.DataFrame(
    X,
    columns=feature_columns
)


# ============================================================
# COMBINE MOLECULE INFORMATION + FEATURES
# ============================================================

result = pd.concat(
    [
        df[
            [
                "molecule_chembl_id",
                "canonical_smiles",
                "standard_value"
            ]
        ].reset_index(drop=True),

        X_df.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE GENERATION COMPLETE")
print("=" * 60)

print("Molecules:", len(result))
print("Features per molecule:", X.shape[1])
print("Final dataset shape:", result.shape)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")

print(
    result[
        [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value"
        ]
    ].head()
)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)