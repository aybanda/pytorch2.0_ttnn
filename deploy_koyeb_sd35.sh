#!/bin/bash
# Deployment script for Stable Diffusion 3.5 Medium test on Koyeb
# Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)

set -e

echo "🚀 Setting up Stable Diffusion 3.5 Medium test environment on Koyeb"
echo "Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)"
echo "=" * 60

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git wget

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch and CUDA dependencies
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# Install TTNN (Tenstorrent Neural Network)
echo "🧠 Installing TTNN..."
pip install ttnn @ https://github.com/tenstorrent/tt-metal/releases/download/v0.59.0-rc56/ttnn-0.59.0rc56-cp310-cp310-manylinux_2_34_x86_64.whl

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install diffusers==0.32.2 transformers==4.38.0 huggingface_hub[cli] psutil

# Clone the repository (if not already present)
if [ ! -d "pytorch2.0_ttnn" ]; then
    echo "📥 Cloning pytorch2.0_ttnn repository..."
    git clone https://github.com/tenstorrent/pytorch2.0_ttnn.git
    cd pytorch2.0_ttnn
else
    echo "📁 Repository already exists, updating..."
    cd pytorch2.0_ttnn
    git pull origin main
fi

# Install the pytorch2.0_ttnn package
echo "🔧 Installing pytorch2.0_ttnn package..."
pip install -e .

# Copy the test script
echo "📋 Copying test script..."
cp ../test_sd35_koyeb.py .

# Check Tenstorrent hardware
echo "🔍 Checking Tenstorrent hardware..."
if command -v tt-smi &> /dev/null; then
    echo "✅ tt-smi found, checking hardware status..."
    tt-smi
else
    echo "⚠️ tt-smi not found, checking for Tenstorrent devices..."
    ls /dev/tenstorrent* 2>/dev/null || echo "No Tenstorrent devices found"
fi

# Run the test
echo "🎬 Running Stable Diffusion 3.5 Medium test..."
echo "Target FPS: 0.3"
echo "Current baseline: 0.06 FPS"
echo "=" * 60

python test_sd35_koyeb.py

echo "🏁 Test completed!"
echo "📊 Check the output above for performance results"
echo "📋 Report results back to GitHub issue #1042" 