import os
import random
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from google.cloud import storage

from src.constants import DEVICE
from src.pipeline.prediction import PredictionPipeline

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Signature Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Custom card styles */
    .custom-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .custom-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .custom-card-subtitle {
        font-size: 0.9rem;
        color: #666;
    }

    /* Hero text */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Upload area */
    .stFileUploader {
        border-radius: 12px;
    }

    /* Result styling */
    .result-container {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CONFIGURATION
# ============================================================

GCP_BUCKET_NAME = "sign-recognition"
DEMO_ZIP_NAME = "demo_samples.zip"
TEMP_DIR = Path(tempfile.gettempdir())
DEMO_ZIP_PATH = TEMP_DIR / DEMO_ZIP_NAME
DEMO_EXTRACT_DIR = TEMP_DIR / "signature_demo_samples"


# ============================================================
# MODEL CACHE
# ============================================================

@st.cache_resource
def load_model(model_path: str, model_version: float):
    """Load the PyTorch model once into memory."""
    model = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )
    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# PIPELINE CACHE
# ============================================================

@st.cache_resource
def get_prediction_pipeline():
    """Create PredictionPipeline once."""
    return PredictionPipeline()


# ============================================================
# INITIALIZE PIPELINE
# ============================================================

prediction_pipeline = get_prediction_pipeline()

# ============================================================
# GET LOCAL MODEL
# ============================================================

model_path = prediction_pipeline.get_model_path()
model_version = os.path.getmtime(model_path)

# ============================================================
# LOAD MODEL INTO MEMORY
# ============================================================

model = load_model(
    model_path=model_path,
    model_version=model_version
)

# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

if "random_samples" not in st.session_state:
    st.session_state.random_samples = None

if "demo_dataset" not in st.session_state:
    st.session_state.demo_dataset = None


# ============================================================
# DEMO DATASET HELPERS
# ============================================================

def find_existing_demo_dataset():
    """Check whether the demo dataset has already been downloaded."""
    if not DEMO_EXTRACT_DIR.exists():
        return None

    original_dir = DEMO_EXTRACT_DIR / "Original"
    forged_dir = DEMO_EXTRACT_DIR / "Forged"

    if original_dir.exists() and forged_dir.exists():
        return DEMO_EXTRACT_DIR

    # Handle ZIPs containing a nested root directory.
    for directory in DEMO_EXTRACT_DIR.iterdir():
        if not directory.is_dir():
            continue

        nested_original = directory / "Original"
        nested_forged = directory / "Forged"

        if nested_original.exists() and nested_forged.exists():
            return directory

    return None


def download_demo_dataset():
    """Download and extract demo_samples.zip from GCP."""
    # First: Check whether it already exists locally.
    existing_dataset = find_existing_demo_dataset()
    if existing_dataset is not None:
        return existing_dataset

    # Create extraction directory.
    DEMO_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # Download ZIP from GCP.
    try:
        client = storage.Client()
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(DEMO_ZIP_NAME)
        blob.download_to_filename(str(DEMO_ZIP_PATH))
    except Exception as e:
        raise RuntimeError(
            f"Could not download demo_samples.zip from GCP bucket '{GCP_BUCKET_NAME}'.\n\n{type(e).__name__}: {e}"
        ) from e

    # Extract ZIP.
    try:
        with zipfile.ZipFile(DEMO_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DEMO_EXTRACT_DIR)
    except Exception as e:
        raise RuntimeError(f"Could not extract demo_samples.zip.\n\n{type(e).__name__}: {e}") from e

    # Validate extracted structure.
    dataset_dir = find_existing_demo_dataset()
    if dataset_dir is None:
        raise RuntimeError(
            "demo_samples.zip was downloaded and extracted, but the expected folder structure was not found.\n\n"
            "Expected structure:\n\n"
            "demo_samples.zip\n"
            "├── Original/\n"
            "└── Forged/"
        )

    return dataset_dir


# ============================================================
# RANDOM SAMPLE GENERATOR
# ============================================================

def get_random_samples(dataset_dir, samples_per_class=5):
    """Select random Original and Forged signatures."""
    if dataset_dir is None:
        return {"Original": [], "Forged": []}

    supported_extensions = {".png", ".jpg", ".jpeg"}
    result = {"Original": [], "Forged": []}

    for class_name in ["Original", "Forged"]:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue

        images = [
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_extensions
        ]

        if not images:
            continue

        number_of_samples = min(samples_per_class, len(images))
        result[class_name] = random.sample(images, number_of_samples)

    return result


# ============================================================
# HEADER
# ============================================================

st.markdown("### ✍️ Signature AI")
st.markdown('<h1 class="hero-title">Signature Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Identify whether a handwritten signature is Original or Forged.</p>',
            unsafe_allow_html=True)

# ============================================================
# DATASET INFO
# ============================================================

with st.expander("ℹ️ About this model"):
    st.write("""
        **Signature Recognition** is an image classification system trained on the CEDAR Signature Dataset.

        The model classifies a signature into two categories:

        - **Original** — genuine signature
        - **Forged** — suspected forged signature

        **Model:** ResNet34  
        **Dataset:** CEDAR Signature Dataset  
        **Classes:** Original / Forged
    """)

# ============================================================
# RANDOM TEST SAMPLES
# ============================================================

st.divider()
st.subheader("Try a sample")
st.caption("Don't have an image? Load demo signatures from the CEDAR dataset and test the model.")

# ============================================================
# GENERATE RANDOM SAMPLES
# ============================================================

generate_samples = st.button("🎲 Generate Random Samples", use_container_width=False)

if generate_samples:
    # Check session state first.
    dataset_dir = st.session_state.demo_dataset

    # If this is the first request, check /tmp.
    if dataset_dir is None:
        dataset_dir = find_existing_demo_dataset()

    # Dataset doesn't exist. Download it NOW.
    if dataset_dir is None:
        try:
            with st.spinner("Downloading demo signatures..."):
                dataset_dir = download_demo_dataset()
        except Exception as e:
            st.error("Could not load demo signatures from GCP.")
            st.exception(e)
            dataset_dir = None

    # Dataset successfully available.
    if dataset_dir is not None:
        st.session_state.demo_dataset = dataset_dir
        st.session_state.random_samples = get_random_samples(dataset_dir, samples_per_class=5)
        st.session_state.uploaded_bytes = None
        st.session_state.selected_sample = None
        st.session_state.prediction_result = None
        st.rerun()

# ============================================================
# DISPLAY RANDOM SAMPLES
# ============================================================

random_samples = st.session_state.random_samples

if random_samples is not None:
    # ORIGINAL
    original_samples = random_samples.get("Original", [])
    if original_samples:
        st.markdown("#### ✅ Original signatures")
        original_columns = st.columns(len(original_samples))

        for index, sample_path in enumerate(original_samples):
            with original_columns[index]:
                try:
                    image = Image.open(sample_path)
                    st.image(image, use_container_width=True)

                    if st.button("Test", key=f"original_{index}_{sample_path.name}", use_container_width=True):
                        with open(sample_path, "rb") as file:
                            st.session_state.uploaded_bytes = file.read()
                        st.session_state.selected_sample = f"Original • {sample_path.name}"
                        st.session_state.prediction_result = None
                        st.rerun()
                except Exception:
                    st.warning("Could not load sample.")

    # FORGED
    forged_samples = random_samples.get("Forged", [])
    if forged_samples:
        st.markdown("#### ⚠️ Forged signatures")
        forged_columns = st.columns(len(forged_samples))

        for index, sample_path in enumerate(forged_samples):
            with forged_columns[index]:
                try:
                    image = Image.open(sample_path)
                    st.image(image, use_container_width=True)

                    if st.button("Test", key=f"forged_{index}_{sample_path.name}", use_container_width=True):
                        with open(sample_path, "rb") as file:
                            st.session_state.uploaded_bytes = file.read()
                        st.session_state.selected_sample = f"Forged • {sample_path.name}"
                        st.session_state.prediction_result = None
                        st.rerun()
                except Exception:
                    st.warning("Could not load sample.")

# ============================================================
# UPLOAD SECTION
# ============================================================

st.divider()

left_column, right_column = st.columns([1, 1], gap="large")

# ============================================================
# LEFT COLUMN — UPLOAD
# ============================================================

with left_column:
    st.markdown(
        """
        <div class="custom-card">
            <div class="custom-card-title">📤 Upload your signature</div>
            <div class="custom-card-subtitle">Upload a JPG, JPEG or PNG image.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.session_state.uploaded_bytes = uploaded_file.getvalue()
        st.session_state.selected_sample = None
        st.session_state.prediction_result = None

# ============================================================
# RIGHT COLUMN — PREVIEW
# ============================================================

with right_column:
    st.markdown(
        """
        <div class="custom-card">
            <div class="custom-card-title">🖼️ Signature Preview</div>
            <div class="custom-card-subtitle">Your selected signature will appear here.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.uploaded_bytes is not None:
        st.image(
            st.session_state.uploaded_bytes,
            use_container_width=True,
            caption=st.session_state.selected_sample if st.session_state.selected_sample else "Uploaded Signature"
        )
    else:
        st.info("Upload a signature or select a sample above.")

# ============================================================
# ACTION SECTION
# ============================================================

if st.session_state.uploaded_bytes is not None:
    st.divider()
    predict_column, clear_column = st.columns([3, 1])

    # PREDICT
    with predict_column:
        predict_button = st.button("🚀 Predict Signature", type="primary", use_container_width=True)

    # CLEAR
    with clear_column:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    # CLEAR ACTION
    if clear_button:
        st.session_state.uploaded_bytes = None
        st.session_state.prediction_result = None
        st.session_state.selected_sample = None
        st.rerun()

    # PREDICTION
    if predict_button:
        with st.spinner("🔍 Analyzing signature..."):
            result = prediction_pipeline.run_pipeline(
                image_bytes=st.session_state.uploaded_bytes,
                model=model
            )
        st.session_state.prediction_result = result

# ============================================================
# RESULT
# ============================================================

result = st.session_state.prediction_result

if result is not None:
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]

    st.divider()
    st.subheader("Prediction Result")

    # UNKNOWN
    if predicted_class == "Unknown":
        st.warning(
            f"⚠️ **Unknown signature**\n\n"
            f"Confidence: **{confidence * 100:.2f}%**\n\n"
            f"The prediction confidence is below the configured threshold."
        )

    # VALID PREDICTION
    else:
        result_column, confidence_column, status_column = st.columns(3)

        # PREDICTION
        with result_column:
            st.metric(label="Prediction", value=predicted_class)

        # CONFIDENCE
        with confidence_column:
            st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

        # STATUS
        with status_column:
            if predicted_class.lower() == "original":
                st.success("✅ Original Signature")
            elif predicted_class.lower() == "forged":
                st.error("⚠️ Forged Signature")
            else:
                st.info(predicted_class)

        st.progress(confidence, text="Model Confidence")

# ============================================================
# FOOTER
# ============================================================

st.divider()
footer_left, footer_center, footer_right = st.columns([1, 2, 1])

with footer_center:
    st.caption("✍️ Signature Recognition • ResNet34 • CEDAR Dataset • Original vs Forged")