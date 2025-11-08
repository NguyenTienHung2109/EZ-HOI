FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN python -m pip install --upgrade pip

# Install PyTorch and related packages with consistent CUDA version
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy and install requirements before copying source code for better caching
COPY requirements.dev.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.dev.txt

# Copy only necessary files for CLIP and pocket setup
COPY CLIP/ ./CLIP/
COPY pocket/ ./pocket/

# Install CLIP and pocket
RUN cd CLIP && python setup.py develop && cd .. && \
    pip install -e pocket

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH to include local CLIP directory
ENV PYTHONPATH="${PYTHONPATH}:/app/CLIP"

# Default command
CMD ["/bin/bash"]