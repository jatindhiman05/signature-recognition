FROM google/cloud-sdk:latest

WORKDIR /sign

# ============================================================
# COPY PROJECT
# ============================================================

COPY . /sign


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsm6 \
        libxext6 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTHON VIRTUAL ENVIRONMENT
# ============================================================

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

RUN pip install --upgrade pip

RUN pip install torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt


# ============================================================
# INSTALL PROJECT
# ============================================================

RUN pip install -e .


# ============================================================
# STREAMLIT
# ============================================================

EXPOSE 8501


# ============================================================
# START APPLICATION
# ============================================================

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]