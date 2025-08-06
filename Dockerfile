# Dockerfile for Stable Diffusion 3.5 Medium Testing on Koyeb (Worker Mode)
# Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)
# Using Tenstorrent base image for proper hardware support

FROM ghcr.io/tenstorrent/pytorch2.0_ttnn/ubuntu-22.04-amd64:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install additional dependencies
RUN pip install --upgrade pip && \
    pip install diffusers==0.32.2 transformers==4.38.0 huggingface_hub[cli] psutil

# Copy the entire project
COPY . .

# Make test scripts executable
RUN chmod +x deploy_koyeb_sd35.sh

# Set the SD3.5 test script as the entrypoint (worker mode - runs once and exits)
ENTRYPOINT ["python3", "test_sd35_only.py"] 