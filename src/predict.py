# ============================================================
# AI DRUG SCREENING - BACE1 PREDICTION
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "models/random_forest_bace1.joblib"
FEATURE_INFO_FILE = "models/feature_info_bace1.joblib"

FP_SIZE = 2048


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AI DRUG SCREENING - BACE1 PREDICTION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}\n"
        "Please run train_model.py first."
    )

if not os.path.exists(FEATURE_INFO_FILE):
    raise FileNotFoundError(
        f"Feature information not found: {FEATURE_INFO_FILE}\n"
        "Please run train_model.py first."
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

print("Random Forest model loaded successfully.")


# ============================================================
# LOAD FEATURE INFORMATION
# ============================================================

feature_info = joblib.load(FEATURE_INFO_FILE)

print("Feature information loaded.")


# ============================================================
# GET EXACT FEATURE NAMES USED DURING TRAINING
# ============================================================

# This is the important fix.
#
# The trained model remembers the exact feature names:
# fp_0, fp_1, ..., fp_2047
#
# We use those exact names instead of creating
# FP_0, FP_1, etc.

if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)

else:
    # Fallback
    feature_names = [f"fp_{i}" for i in range(FP_SIZE)]


print("\nModel expects:", len(feature_names), "features")

if len(feature_names) != FP_SIZE:
    raise ValueError(
        f"Expected {FP_SIZE} fingerprint features, "
        f"but model expects {len(feature_names)}."
    )


print(
    "Feature naming:",
    feature_names[0],
    "...",
    feature_names[-1]
)


# ============================================================
# CREATE MORGAN FINGERPRINT GENERATOR
# ============================================================

print("\nCreating Morgan fingerprint generator...")

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=FP_SIZE
)


# ============================================================
# PREDICTION LOOP
# ============================================================

print("\n" + "=" * 70)
print("ENTER MOLECULE INFORMATION")
print("=" * 70)

print("\nEnter a molecular SMILES.")
print("Example: CC(=O)OC1=CC=CC=C1C(=O)O")
print("Type 'exit' to stop.")


while True:

    smiles = input("\nSMILES: ").strip()

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if smiles.lower() == "exit":
        print("\nExiting prediction system.")
        break

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not smiles:
        print("Please enter a SMILES.")
        continue

    try:

        # ====================================================
        # CONVERT SMILES TO MOLECULE
        # ====================================================

        print("\nProcessing molecule...")

        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            print("\nERROR:")
            print("Invalid SMILES.")
            print("Please enter a valid molecular SMILES.")
            continue


        # ====================================================
        # GENERATE MORGAN FINGERPRINT
        # ====================================================

        fingerprint = morgan_generator.GetFingerprint(molecule)

        # Convert RDKit fingerprint to NumPy array
        fingerprint_array = np.zeros(
            (FP_SIZE,),
            dtype=np.float32
        )

        from rdkit import DataStructs

        DataStructs.ConvertToNumpyArray(
            fingerprint,
            fingerprint_array
        )


        print(
            "Fingerprint shape:",
            (1, FP_SIZE)
        )


        # ====================================================
        # CREATE DATAFRAME WITH EXACT TRAINING COLUMN NAMES
        # ====================================================

        # IMPORTANT:
        # These names are lowercase fp_0 ... fp_2047
        # exactly like the training dataset.

        X_new = pd.DataFrame(
            [fingerprint_array],
            columns=feature_names
        )


        # ====================================================
        # PREDICT BACE1 ACTIVITY
        # ====================================================

        predicted_activity = model.predict(X_new)[0]


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if not np.isfinite(predicted_activity):
            raise ValueError(
                "Model returned an invalid prediction."
            )


        predicted_activity = max(
            0.0,
            float(predicted_activity)
        )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        # BACE1 activity is reported in nM.
        #
        # Lower nM = stronger activity
        # Higher nM = weaker activity

        if predicted_activity <= 100:
            interpretation = "VERY HIGH ACTIVITY"

        elif predicted_activity <= 1000:
            interpretation = "HIGH ACTIVITY"

        elif predicted_activity <= 10000:
            interpretation = "MODERATE ACTIVITY"

        elif predicted_activity <= 100000:
            interpretation = "LOW ACTIVITY"

        else:
            interpretation = "VERY LOW ACTIVITY"


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        print("\n" + "-" * 70)

        print("PREDICTION RESULT")

        print("-" * 70)

        print("SMILES:")
        print(smiles)

        print("\nPredicted BACE1 activity:")
        print(f"{predicted_activity:.4f} nM")

        print("\nScreening interpretation:")
        print(interpretation)

        print("-" * 70)


    except Exception as error:

        print("\nERROR:")
        print(error)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)