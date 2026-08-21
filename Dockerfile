FROM google/cloud-sdk:latest

WORKDIR /sign

COPY . /sign

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsm6 \
        libxext6 \
        apt-transport-https \
        ca-certificates \
        gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

RUN pip install torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

RUN pip install -e .

EXPOSE 8000

CMD ["python", "app.py"]