FROM python:3.11-slim

WORKDIR /sign


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && \
    apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        ca-certificates \
        && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# COPY PROJECT
# ============================================================

COPY . /sign


# ============================================================
# INSTALL PROJECT
# ============================================================

RUN pip install --no-cache-dir -e .


# ============================================================
# STREAMLIT
# ============================================================

EXPOSE 8501


# ============================================================
# START APPLICATION
# ============================================================

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]