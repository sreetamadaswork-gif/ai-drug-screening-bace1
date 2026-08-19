import pandas as pd
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

INPUT_FILE = "data/raw/bace1_activity_raw.csv"
OUTPUT_FILE = "data/processed/bace1_clean.csv"


# ==========================================
# LOAD DATA
# ==========================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)


# ==========================================
# SELECT IMPORTANT COLUMNS
# ==========================================

columns = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "target_chembl_id",
    "target_organism"
]

df = df[columns]

print("\nSelected columns:")
print(df.columns.tolist())


# ==========================================
# REMOVE MISSING VALUES
# ==========================================

df = df.dropna(
    subset=[
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value"
    ]
)

print("\nAfter removing missing values:", df.shape)


# ==========================================
# KEEP IC50 DATA
# ==========================================

df = df[df["standard_type"].str.upper() == "IC50"]

print("\nAfter keeping IC50:", df.shape)


# ==========================================
# KEEP MOLAR UNITS
# ==========================================

df = df[
    df["standard_units"].str.lower().isin(
        ["nm", "um", "mm", "m"]
    )
]

print("After unit filtering:", df.shape)


# ==========================================
# CONVERT ACTIVITY TO NUMERIC
# ==========================================

df["standard_value"] = pd.to_numeric(
    df["standard_value"],
    errors="coerce"
)

df = df.dropna(subset=["standard_value"])

print("After numeric conversion:", df.shape)


# ==========================================
# REMOVE INVALID VALUES
# ==========================================

df = df[df["standard_value"] > 0]

print("After removing invalid values:", df.shape)


# ==========================================
# REMOVE DUPLICATE MOLECULES
# ==========================================

df = df.drop_duplicates(
    subset=["molecule_chembl_id", "canonical_smiles"]
)

print("After removing duplicates:", df.shape)


# ==========================================
# SAVE CLEAN DATA
# ==========================================

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print("Final dataset shape:", df.shape)
print("Saved to:", OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())