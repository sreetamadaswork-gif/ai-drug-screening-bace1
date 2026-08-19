# ============================================================
# AI DRUG SCREENING - BACE1 MODEL TRAINING
# Log10-transformed activity regression
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/bace1_features.csv"

MODEL_DIR = "models"
RESULT_DIR = "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20

N_FINGERPRINTS = 2048


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("AI DRUG SCREENING - BACE1 MODEL TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n\n"
        "Please run:\n"
        "python src/generate_features.py"
    )

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print("\nChecking required columns...")

required_columns = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_value"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}"
    )

print("Required columns found.")


# ============================================================
# CLEAN TARGET VARIABLE
# ============================================================

print("\nCleaning target variable...")

rows_before = len(df)

df["standard_value"] = pd.to_numeric(
    df["standard_value"],
    errors="coerce"
)

# Remove missing, zero and negative activity values
df = df[
    df["standard_value"].notna()
    & (df["standard_value"] > 0)
].copy()

rows_after = len(df)

print("Rows before cleaning:", rows_before)
print("Rows after cleaning :", rows_after)
print("Rows removed        :", rows_before - rows_after)

print(
    "Raw activity range:",
    df["standard_value"].min(),
    "to",
    df["standard_value"].max(),
    "nM"
)


# ============================================================
# LOG10 TRANSFORMATION
# ============================================================

print("\nApplying log10 transformation...")

df["log10_activity"] = np.log10(
    df["standard_value"]
)

print(
    "Log10 activity range:",
    round(df["log10_activity"].min(), 4),
    "to",
    round(df["log10_activity"].max(), 4)
)

print("\nLog10 activity statistics:")
print(df["log10_activity"].describe())


# ============================================================
# SELECT FINGERPRINT FEATURES
# ============================================================

print("\nSelecting molecular fingerprint features...")

fingerprint_columns = [
    f"FP_{i}"
    for i in range(N_FINGERPRINTS)
]

missing_fp = [
    col
    for col in fingerprint_columns
    if col not in df.columns
]

if missing_fp:

    # Try lowercase naming if necessary
    lowercase_columns = [
        f"fp_{i}"
        for i in range(N_FINGERPRINTS)
    ]

    lowercase_missing = [
        col
        for col in lowercase_columns
        if col not in df.columns
    ]

    if not lowercase_missing:
        fingerprint_columns = lowercase_columns
        print("Using lowercase fingerprint columns.")

    else:
        print(
            "\nFingerprint feature error!"
        )

        print(
            f"Expected {N_FINGERPRINTS} fingerprint features."
        )

        print(
            f"Found {len([c for c in df.columns if str(c).lower().startswith('fp_')])}."
        )

        raise ValueError(
            "\nFingerprint columns are missing.\n"
            "Please run:\n"
            "python src/generate_features.py"
        )


print(
    "Fingerprint features detected:",
    len(fingerprint_columns)
)


# ============================================================
# CREATE X AND y
# ============================================================

X = df[fingerprint_columns].copy()

y = df["log10_activity"].copy()


# ============================================================
# FINAL FEATURE VALIDATION
# ============================================================

print("\nValidating feature matrix...")

if X.shape[1] != N_FINGERPRINTS:
    raise ValueError(
        f"\nExpected {N_FINGERPRINTS} fingerprint features, "
        f"but found {X.shape[1]}."
    )

# Convert to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace missing values
X = X.fillna(0)

# Make sure all values are finite
if not np.isfinite(X.to_numpy()).all():
    raise ValueError(
        "\nFingerprint matrix contains NaN or infinite values."
    )

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

print(
    "Feature values:",
    X.values.min(),
    "to",
    X.values.max()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# MODEL 1 - RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions_log = rf_model.predict(
    X_test
)


# ============================================================
# MODEL 2 - SUPPORT VECTOR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SUPPORT VECTOR REGRESSION")
print("=" * 70)

svr_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svr",
            SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.05,
                gamma="scale"
            )
        )
    ]
)

svr_model.fit(
    X_train,
    y_train
)

svr_predictions_log = svr_model.predict(
    X_test
)


# ============================================================
# CONVERT PREDICTIONS BACK TO nM
# ============================================================

rf_predictions_nm = np.power(
    10,
    rf_predictions_log
)

svr_predictions_nm = np.power(
    10,
    svr_predictions_log
)

actual_nm = np.power(
    10,
    y_test
)


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(
    actual_log,
    predicted_log,
    actual_nm,
    predicted_nm
):

    mae_log = mean_absolute_error(
        actual_log,
        predicted_log
    )

    rmse_log = np.sqrt(
        mean_squared_error(
            actual_log,
            predicted_log
        )
    )

    r2_log = r2_score(
        actual_log,
        predicted_log
    )

    mae_nm = mean_absolute_error(
        actual_nm,
        predicted_nm
    )

    rmse_nm = np.sqrt(
        mean_squared_error(
            actual_nm,
            predicted_nm
        )
    )

    return (
        mae_log,
        rmse_log,
        r2_log,
        mae_nm,
        rmse_nm
    )


# ============================================================
# CALCULATE RANDOM FOREST METRICS
# ============================================================

(
    rf_mae_log,
    rf_rmse_log,
    rf_r2,
    rf_mae_nm,
    rf_rmse_nm
) = calculate_metrics(
    y_test,
    rf_predictions_log,
    actual_nm,
    rf_predictions_nm
)


# ============================================================
# CALCULATE SVR METRICS
# ============================================================

(
    svr_mae_log,
    svr_rmse_log,
    svr_r2,
    svr_mae_nm,
    svr_rmse_nm
) = calculate_metrics(
    y_test,
    svr_predictions_log,
    actual_nm,
    svr_predictions_nm
)


# ============================================================
# PRINT MODEL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print("\nRandom Forest")
print("-" * 40)

print(
    "MAE (log10):",
    round(rf_mae_log, 4)
)

print(
    "RMSE (log10):",
    round(rf_rmse_log, 4)
)

print(
    "R2:",
    round(rf_r2, 4)
)

print(
    "MAE (nM):",
    round(rf_mae_nm, 4)
)

print(
    "RMSE (nM):",
    round(rf_rmse_nm, 4)
)


print("\nSupport Vector Regression")
print("-" * 40)

print(
    "MAE (log10):",
    round(svr_mae_log, 4)
)

print(
    "RMSE (log10):",
    round(svr_rmse_log, 4)
)

print(
    "R2:",
    round(svr_r2, 4)
)

print(
    "MAE (nM):",
    round(svr_mae_nm, 4)
)

print(
    "RMSE (nM):",
    round(svr_rmse_nm, 4)
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

if rf_r2 >= svr_r2:

    best_model_name = "Random Forest"
    best_model = rf_model
    best_predictions_log = rf_predictions_log
    best_predictions_nm = rf_predictions_nm

else:

    best_model_name = "Support Vector Regression"
    best_model = svr_model
    best_predictions_log = svr_predictions_log
    best_predictions_nm = svr_predictions_nm


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best model:", best_model_name)


# ============================================================
# SAVE MODELS
# ============================================================

rf_path = os.path.join(
    MODEL_DIR,
    "random_forest_bace1_log.joblib"
)

svr_path = os.path.join(
    MODEL_DIR,
    "svr_bace1_log.joblib"
)

feature_info_path = os.path.join(
    MODEL_DIR,
    "feature_info_bace1_log.joblib"
)


joblib.dump(
    rf_model,
    rf_path
)

joblib.dump(
    svr_model,
    svr_path
)


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_info = {
    "feature_columns": fingerprint_columns,
    "n_features": len(fingerprint_columns),
    "target_column": "standard_value",
    "target_transform": "log10",
    "activity_unit": "nM",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "best_model": best_model_name
}

joblib.dump(
    feature_info,
    feature_info_path
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    [
        {
            "Model": "Random Forest",
            "MAE_log10": rf_mae_log,
            "RMSE_log10": rf_rmse_log,
            "R2": rf_r2,
            "MAE_nM": rf_mae_nm,
            "RMSE_nM": rf_rmse_nm
        },
        {
            "Model": "Support Vector Regression",
            "MAE_log10": svr_mae_log,
            "RMSE_log10": svr_rmse_log,
            "R2": svr_r2,
            "MAE_nM": svr_mae_nm,
            "RMSE_nM": svr_rmse_nm
        }
    ]
)

comparison_path = os.path.join(
    RESULT_DIR,
    "model_comparison_log.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_results = df.loc[
    X_test.index,
    [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value"
    ]
].copy()

prediction_results["actual_log10"] = y_test.values

prediction_results["rf_predicted_log10"] = (
    rf_predictions_log
)

prediction_results["rf_predicted_nM"] = (
    rf_predictions_nm
)

prediction_results["svr_predicted_log10"] = (
    svr_predictions_log
)

prediction_results["svr_predicted_nM"] = (
    svr_predictions_nm
)

prediction_results["best_model"] = (
    best_model_name
)

prediction_results["best_prediction_nM"] = (
    best_predictions_nm
)

prediction_path = os.path.join(
    RESULT_DIR,
    "model_predictions_log.csv"
)

prediction_results.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "best_bace1_model_log.joblib"
)

joblib.dump(
    best_model,
    best_model_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    "\nMolecules:",
    len(df)
)

print(
    "Fingerprint features:",
    X.shape[1]
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "\nTarget transformation:"
)

print(
    "standard_value → log10(standard_value)"
)

print(
    "\nBest model:",
    best_model_name
)

print(
    "\nFiles created:"
)

print(
    "1.",
    rf_path
)

print(
    "2.",
    svr_path
)

print(
    "3.",
    feature_info_path
)

print(
    "4.",
    best_model_path
)

print(
    "5.",
    comparison_path
)

print(
    "6.",
    prediction_path
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)