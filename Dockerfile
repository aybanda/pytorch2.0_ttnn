# Dockerfile for Stable Diffusion 3.5 Medium Testing on Koyeb (Worker Mode)
# Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV CUDA_VERSION=11.8
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Create virtual environment
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies (without TTNN for now)
RUN pip install diffusers==0.32.2 transformers==4.38.0 huggingface_hub[cli] psutil

# Copy the entire project
COPY . .

# Make test scripts executable
RUN chmod +x deploy_koyeb_sd35.sh

# Set the SD3.5 test script as the entrypoint (worker mode - runs once and exits)
ENTRYPOINT ["python3", "test_sd35_only.py"] 