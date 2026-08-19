import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Drug Screening | BACE1",
    page_icon="🧬",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }

    .activity-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    .low {
        color: #16a34a;
    }

    .medium {
        color: #d97706;
    }

    .high {
        color: #dc2626;
    }

    .info-box {
        padding: 1rem;
        border-radius: 12px;
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_bace1_model_log.joblib"
)

FEATURE_INFO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_info_bace1_log.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


@st.cache_resource
def load_feature_info():

    if not os.path.exists(FEATURE_INFO_PATH):
        raise FileNotFoundError(
            f"Feature information not found:\n{FEATURE_INFO_PATH}"
        )

    feature_info = joblib.load(FEATURE_INFO_PATH)

    return feature_info


# ============================================================
# MORGAN FINGERPRINT
# ============================================================

def generate_fingerprint(smiles):

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None, None

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=2,
        nBits=2048
    )

    fingerprint_array = np.array(fingerprint)

    fingerprint_array = fingerprint_array.reshape(1, -1)

    return molecule, fingerprint_array


# ============================================================
# ACTIVITY INTERPRETATION
# ============================================================

def classify_activity(activity_nm):

    if activity_nm < 100:
        return "HIGH ACTIVITY", "high"

    elif activity_nm < 10000:
        return "MODERATE ACTIVITY", "medium"

    else:
        return "LOW ACTIVITY", "low"


# ============================================================
# MOLECULAR DESCRIPTORS
# ============================================================

def calculate_descriptors(molecule):

    return {
        "Molecular Weight": round(
            Descriptors.MolWt(molecule), 2
        ),

        "LogP": round(
            Descriptors.MolLogP(molecule), 2
        ),

        "H-Bond Donors": Descriptors.NumHDonors(
            molecule
        ),

        "H-Bond Acceptors": Descriptors.NumHAcceptors(
            molecule
        ),

        "Rotatable Bonds": Descriptors.NumRotatableBonds(
            molecule
        ),

        "Heavy Atoms": molecule.GetNumHeavyAtoms(),

        "Ring Count": Descriptors.RingCount(
            molecule
        ),
    }


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🧬 AI Drug Screening</h1>

        <p>
        BACE1 Activity Prediction using
        Molecular Fingerprints and Machine Learning
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔬 About")

    st.write(
        """
        This application predicts BACE1 molecular
        activity from a SMILES representation.

        The machine learning pipeline uses:

        • RDKit molecular processing

        • Morgan molecular fingerprints

        • 2048-bit fingerprint representation

        • Random Forest regression

        • Log10-transformed activity target
        """
    )

    st.divider()

    st.subheader("Model")

    st.write("**Algorithm:** Random Forest")
    st.write("**Target:** BACE1 activity")
    st.write("**Features:** 2048")
    st.write("**Target:** log10(Standard Value)")

    st.divider()

    st.caption(
        "Research prototype — predictions should "
        "not be interpreted as clinical recommendations."
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()
    feature_info = load_feature_info()

except Exception as error:

    st.error("Unable to load the trained model.")

    st.code(str(error))

    st.stop()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🧪 Molecule Screening")

st.write(
    "Enter a molecular SMILES string to predict "
    "its BACE1 activity."
)

smiles = st.text_input(
    "Molecular SMILES",
    placeholder="Example: CC(=O)OC1=CC=CC=C1C(=O)O"
)


predict_button = st.button(
    "🔬 Predict BACE1 Activity",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if not smiles.strip():

        st.warning(
            "Please enter a molecular SMILES."
        )

        st.stop()

    with st.spinner("Analyzing molecule..."):

        molecule, fingerprint = generate_fingerprint(
            smiles.strip()
        )

        if molecule is None:

            st.error(
                "Invalid SMILES. Please check the molecular structure."
            )

            st.stop()

        try:

            # ------------------------------------------------
            # MODEL FEATURE NAMES
            # ------------------------------------------------

            if hasattr(model, "feature_names_in_"):

                feature_names = list(
                    model.feature_names_in_
                )

                fingerprint_df = pd.DataFrame(
                    fingerprint,
                    columns=feature_names
                )

            else:

                fingerprint_df = fingerprint

            # ------------------------------------------------
            # PREDICT LOG ACTIVITY
            # ------------------------------------------------

            predicted_log_activity = float(
                model.predict(fingerprint_df)[0]
            )

            # ------------------------------------------------
            # CONVERT BACK TO nM
            # ------------------------------------------------

            predicted_activity_nm = (
                10 ** predicted_log_activity
            )

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            interpretation, css_class = (
                classify_activity(
                    predicted_activity_nm
                )
            )

            # ------------------------------------------------
            # MOLECULAR DESCRIPTORS
            # ------------------------------------------------

            descriptors = calculate_descriptors(
                molecule
            )

            # =================================================
            # RESULTS
            # =================================================

            st.success(
                "Molecule successfully analyzed!"
            )

            st.divider()

            col1, col2 = st.columns(
                [1, 1]
            )

            # ------------------------------------------------
            # MOLECULE IMAGE
            # ------------------------------------------------

            with col1:

                st.subheader(
                    "🧬 Molecular Structure"
                )

                image = Draw.MolToImage(
                    molecule,
                    size=(500, 400)
                )

                st.image(
                    image,
                    use_container_width=True
                )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            with col2:

                st.subheader(
                    "📊 Prediction Result"
                )

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div>
                    <b>Predicted BACE1 Activity</b>
                    </div>

                    <div class="activity-value">
                    {predicted_activity_nm:,.2f} nM
                    </div>

                    <br>

                    <b>Screening Interpretation</b>

                    <div class="{css_class}"
                         style="font-size:1.4rem;
                                font-weight:700;
                                margin-top:0.5rem;">

                    {interpretation}

                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="info-box">

                    <b>Model:</b> Random Forest<br>
                    <b>Fingerprint:</b> Morgan 2048-bit<br>
                    <b>Predicted log10 activity:</b>
                    {predicted_log_activity:.4f}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =================================================
            # MOLECULAR INFORMATION
            # =================================================

            st.divider()

            st.subheader(
                "🔍 Molecular Properties"
            )

            descriptor_df = pd.DataFrame(
                descriptors.items(),
                columns=[
                    "Property",
                    "Value"
                ]
            )

            st.dataframe(
                descriptor_df,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # INPUT SMILES
            # =================================================

            st.divider()

            st.subheader(
                "🧾 Input Information"
            )

            st.code(
                smiles,
                language="text"
            )

            # =================================================
            # INTERPRETATION
            # =================================================

            st.subheader(
                "💡 Interpretation"
            )

            if predicted_activity_nm < 100:

                st.success(
                    """
                    The model predicts relatively strong
                    BACE1 activity based on the learned
                    relationship between molecular fingerprints
                    and experimental activity values.
                    """
                )

            elif predicted_activity_nm < 10000:

                st.warning(
                    """
                    The predicted activity falls within an
                    intermediate range. Experimental validation
                    would be required before drawing conclusions.
                    """
                )

            else:

                st.info(
                    """
                    The predicted activity is relatively high
                    in nM, corresponding to lower predicted
                    inhibitory activity in this screening model.
                    """
                )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(error)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Drug Screening • BACE1 Machine Learning Research Prototype"
)