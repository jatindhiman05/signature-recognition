import os

import streamlit as st
import torch

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
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        max-width: 1200px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }

    .brand-text {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 1rem !important;
    }

    .hero-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        margin-bottom: 0.5rem !important;
        text-align: center !important;
    }

    .hero-subtitle {
        font-size: 1.15rem !important;
        color: #666 !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
    }

    .card-container {
        border: 1px solid #e0e0e0 !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        background: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }

    .card-title {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.35rem !important;
    }

    .card-subtitle {
        font-size: 0.9rem !important;
        color: #666 !important;
        margin-bottom: 1rem !important;
    }

    .result-container {
        border: 1px solid #e0e0e0 !important;
        border-radius: 18px !important;
        padding: 1.5rem !important;
        margin-top: 1rem !important;
        background: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }

    .result-label {
        font-size: 0.85rem !important;
        color: #666 !important;
        margin-bottom: 0.25rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    .result-value {
        font-size: 2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }

    .confidence-value {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Custom button styling */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #ccc !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        text-align: center !important;
    }

    .stFileUploader:hover {
        border-color: #666 !important;
    }

    /* Info box styling */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }

    /* Center align for columns */
    .centered {
        text-align: center !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CACHE
# ============================================================

@st.cache_resource
def load_model(
        model_path: str,
        model_version: float
):
    """
    Load the PyTorch model once into memory.

    Streamlit keeps this model cached across
    reruns and user sessions.
    """

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
    """
    Create PredictionPipeline once and reuse it.
    """

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
# LOAD MODEL INTO STREAMLIT MEMORY
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

# ============================================================
# HEADER - USING PURE STREAMLIT COMPONENTS
# ============================================================

# Create a container for the header
header_container = st.container()

with header_container:
    # Brand
    st.markdown("### ✍️ Signature AI")

    # Title
    st.markdown(
        "<h1 style='text-align: center; font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;'>Signature Recognition</h1>",
        unsafe_allow_html=True
    )

    # Subtitle
    st.markdown(
        "<p style='text-align: center; font-size: 1.15rem; color: #666; margin-bottom: 2rem;'>Identify whether a handwritten signature is Original or Forged.</p>",
        unsafe_allow_html=True
    )

# ============================================================
# MAIN COLUMNS
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)

# ============================================================
# LEFT COLUMN — UPLOAD
# ============================================================

with left_column:
    st.markdown(
        """
        <div class="card-container">
            <div class="card-title">
                📤 Upload your signature
            </div>
            <div class="card-subtitle">
                Upload a JPG, JPEG or PNG image.
            </div>
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
        st.session_state.prediction_result = None

# ============================================================
# RIGHT COLUMN — PREVIEW
# ============================================================

with right_column:
    st.markdown(
        """
        <div class="card-container">
            <div class="card-title">
                🖼️ Signature Preview
            </div>
            <div class="card-subtitle">
                Your uploaded signature will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.uploaded_bytes is not None:
        st.image(
            st.session_state.uploaded_bytes,
            use_container_width=True,
            caption="Uploaded Signature"
        )
    else:
        st.info(
            "📌 Upload a signature image to get started."
        )

# ============================================================
# ACTION SECTION
# ============================================================

if st.session_state.uploaded_bytes is not None:

    st.divider()

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        predict_button = st.button(
            "🚀 Predict Signature",
            type="primary",
            use_container_width=True
        )

    with col2:
        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    # ========================================================
    # CLEAR
    # ========================================================

    if clear_button:
        st.session_state.uploaded_bytes = None
        st.session_state.prediction_result = None
        st.rerun()

    # ========================================================
    # PREDICTION
    # ========================================================

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

    # ========================================================
    # UNKNOWN
    # ========================================================

    if predicted_class == "Unknown":
        st.warning(
            f"### ⚠️ Unknown signature\n\n"
            f"Model confidence: **{confidence * 100:.2f}%**\n\n"
            f"The confidence is below the configured threshold of the model."
        )

    # ========================================================
    # VALID PREDICTION
    # ========================================================

    else:
        # Display prediction with better formatting
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.metric(
                label="Prediction",
                value=predicted_class,
                delta=None
            )

        with col2:
            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%",
                delta=None
            )

        with col3:
            # Display status based on prediction
            if predicted_class.lower() == "original":
                st.success("✅ Verified Original")
            else:
                st.error("⚠️ Potential Forgery")

        # Additional visual indicator
        st.progress(confidence, text="Confidence Level")

# ============================================================
# FOOTER
# ============================================================

st.divider()

# Simple footer using Streamlit components
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption(
        "✍️ Signature Recognition • ResNet34 • CEDAR Dataset"
    )