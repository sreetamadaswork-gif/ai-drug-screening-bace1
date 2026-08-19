import pandas as pd

# Load the raw ChEMBL dataset
file_path = "data/raw/bace1_activity_raw.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("BACE1 DATASET INSPECTION")
print("=" * 60)

# Dataset size
print("\n1. Dataset shape:")
print(df.shape)

# Number of rows
print("\n2. Number of compounds/activity records:")
print(len(df))

# Important columns
important_columns = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "target_chembl_id",
    "target_organism"
]

print("\n3. Important columns:")
for column in important_columns:
    if column in df.columns:
        print("✓", column)
    else:
        print("✗", column, "NOT FOUND")

# Activity types
print("\n4. Activity types:")
print(df["standard_type"].value_counts(dropna=False))

# Activity units
print("\n5. Activity units:")
print(df["standard_units"].value_counts(dropna=False))

# Missing SMILES
print("\n6. Missing SMILES:")
print(df["canonical_smiles"].isna().sum())

# Missing activity values
print("\n7. Missing activity values:")
print(df["standard_value"].isna().sum())

# Unique molecules
print("\n8. Unique molecules:")
print(df["molecule_chembl_id"].nunique())

# Target organisms
print("\n9. Target organisms:")
print(df["target_organism"].value_counts(dropna=False).head(10))

# Show first 5 useful records
print("\n10. First 5 records:")
print(
    df[
        [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_type",
            "standard_value",
            "standard_units"
        ]
    ].head()
)

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)