# Stable Diffusion 3.5 Medium Testing on Koyeb

## Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)

This guide will help you test the Stable Diffusion 3.5 Medium model on Koyeb with Tenstorrent hardware to measure performance and validate TTNN integration.

## 🎯 Performance Targets

- **Target FPS**: 0.3 FPS on batch 1
- **Current Baseline**: 0.06 FPS on batch 1 (from issue #1042)
- **Improvement Needed**: 5x performance improvement

## 🚀 Quick Start on Koyeb

### 1. Set up Koyeb Instance

1. Go to [Koyeb](https://www.koyeb.com/) and create an account
2. Deploy a new app with Tenstorrent hardware
3. Choose the Tenstorrent instance type (Wormhole or Blackhole)
4. Use Ubuntu 22.04 as the base image

### 2. Connect to Your Instance

```bash
ssh root@your-koyeb-instance-ip
```

### 3. Run the Automated Setup

```bash
# Download the deployment script
wget https://raw.githubusercontent.com/your-repo/pytorch2.0_ttnn/main/deploy_koyeb_sd35.sh

# Make it executable
chmod +x deploy_koyeb_sd35.sh

# Run the setup
./deploy_koyeb_sd35.sh
```

### 4. Run the Tests

After setup completes, you can run different test scripts:

#### Basic Performance Test
```bash
python test_sd35_koyeb.py
```

#### Advanced TTNN Test
```bash
python test_sd35_ttnn.py
```

#### Official Test Suite
```bash
cd pytorch2.0_ttnn
python -m pytest tests/models/stable_diffusion/test_stable_diffusion_3_5_medium.py -v
```

## 📊 Expected Results

The tests will provide:

1. **Baseline Performance**: FPS without TTNN compilation
2. **Memory Usage**: RAM consumption during inference
3. **Inference Time**: Time per image generation
4. **TTNN Integration**: Whether TTNN compilation is available
5. **Batch Performance**: Performance with different batch sizes

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# Install TTNN
pip install ttnn @ https://github.com/tenstorrent/tt-metal/releases/download/v0.59.0-rc56/ttnn-0.59.0rc56-cp310-cp310-manylinux_2_34_x86_64.whl

# Install other dependencies
pip install diffusers==0.32.2 transformers==4.38.0 huggingface_hub[cli] psutil

# Clone and install pytorch2.0_ttnn
git clone https://github.com/tenstorrent/pytorch2.0_ttnn.git
cd pytorch2.0_ttnn
pip install -e .
```

## 📋 Test Scripts Overview

### `test_sd35_koyeb.py`
- Basic performance measurement
- Memory usage tracking
- Simple FPS calculation
- Image output generation

### `test_sd35_ttnn.py`
- Advanced TTNN integration testing
- Baseline vs optimized comparison
- Batch size testing
- Comprehensive performance report

### Official Test Suite
- Full integration with pytorch2.0_ttnn
- Proper test framework integration
- Automated metrics collection

## 🎯 Success Criteria

To complete the bounty, you need to demonstrate:

1. ✅ **Model Integration**: Stable Diffusion 3.5 Medium loads and runs
2. ✅ **Performance Measurement**: Accurate FPS measurement
3. ✅ **TTNN Compilation**: Model compiles with TTNN
4. 🎯 **Target Achievement**: 0.3 FPS or better on batch 1
5. 📊 **Documentation**: Results reported to GitHub issue

## 📝 Reporting Results

After running the tests, report your findings to [GitHub Issue #1042](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1042):

1. **Performance Numbers**: FPS, memory usage, inference time
2. **TTNN Status**: Whether compilation worked
3. **Generated Images**: Sample outputs (if any)
4. **Issues Encountered**: Any problems or errors
5. **Recommendations**: Suggestions for optimization

## 🔍 Troubleshooting

### Common Issues

1. **TTNN Import Error**: Ensure you're on Tenstorrent hardware
2. **CUDA Not Available**: Check GPU drivers and PyTorch installation
3. **Memory Issues**: Reduce batch size or use gradient checkpointing
4. **Model Download Issues**: Check internet connection and HuggingFace access

### Debug Commands

```bash
# Check TTNN installation
python -c "import ttnn; print('TTNN available')"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Tenstorrent hardware
tt-smi

# Check system resources
nvidia-smi
free -h
```

## 📚 Additional Resources

- [Tenstorrent Documentation](https://tenstorrent.com/)
- [Koyeb Tenstorrent Guide](https://www.koyeb.com/blog/tenstorrent-cloud-instances)
- [Stable Diffusion 3.5 Model](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
- [GitHub Issue #1042](https://github.com/tenstorrent/pytorch2.0_ttnn/issues/1042)

## 🏆 Bounty Completion

To claim the $1500 bounty:

1. ✅ Implement the test (already done)
2. 🔄 Run tests on Koyeb (in progress)
3. 📊 Achieve 0.3 FPS target
4. 📝 Report results to GitHub issue
5. 🔄 Submit pull request with improvements

Good luck with the testing! 🚀 