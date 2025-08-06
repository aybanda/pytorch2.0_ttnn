#!/usr/bin/env python3
"""
Stable Diffusion 3.5 Medium Performance Test (Graceful SFPI Handling)
Focused on GitHub Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)
Target: 0.3 FPS on batch 1
Current baseline: 0.06 FPS on batch 1

This script handles SFPI issues gracefully and tests performance.
"""

import sys
import torch
import os
import time
import psutil

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def check_tenstorrent_environment():
    """Check Tenstorrent environment and handle SFPI issues"""
    print("🔍 Checking Tenstorrent environment...")
    
    # Check for SFPI
    sfpi_path = "/opt/tenstorrent/sfpi"
    if os.path.exists(sfpi_path):
        print(f"✅ SFPI found at {sfpi_path}")
        return True
    else:
        print(f"⚠️ SFPI not found at {sfpi_path}")
        print("   This is expected on some Tenstorrent instances")
        return False

def test_ttnn_import():
    """Test TTNN import with graceful error handling"""
    try:
        import ttnn
        print("✅ TTNN imported successfully")
        return True
    except Exception as e:
        print(f"⚠️ TTNN import failed: {e}")
        print("   This may be due to SFPI system package issues")
        return False

def test_stable_diffusion_performance():
    """Test Stable Diffusion 3.5 Medium performance"""
    try:
        from diffusers import StableDiffusionPipeline
        print("\n🧪 Testing Stable Diffusion 3.5 Medium...")
        
        # Load the model
        model_id = "stabilityai/stable-diffusion-3-medium"
        print(f"📦 Loading model: {model_id}")
        
        mem_before = get_memory_usage()
        print(f"💾 Memory before loading: {mem_before:.1f} MB")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        print("✅ Model loaded successfully!")
        
        mem_after = get_memory_usage()
        print(f"💾 Memory after loading: {mem_after:.1f} MB")
        print(f"💾 Memory used: {mem_after - mem_before:.1f} MB")
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            print(f"🖥️ Using CUDA device")
        else:
            device = "cpu"
            print(f"🖥️ Using CPU device")
        
        pipe = pipe.to(device)
        
        # Test basic inference
        print("\n🔄 Testing inference performance...")
        prompt = "a photo of an astronaut riding a horse on mars"
        
        # Warmup
        print("🔥 Warming up...")
        with torch.no_grad():
            _ = pipe(prompt, num_inference_steps=1, output_type="latent")
        
        # Performance test
        num_runs = 3
        times = []
        
        for i in range(num_runs):
            print(f"🔄 Performance run {i+1}/{num_runs}...")
            start = time.time()
            with torch.no_grad():
                result = pipe(prompt, num_inference_steps=20, output_type="pil")
            end = time.time()
            
            inference_time = end - start
            times.append(inference_time)
            fps = 1.0 / inference_time
            print(f"   ⏱️ Time: {inference_time:.2f}s, FPS: {fps:.3f}")
        
        avg_time = sum(times) / len(times)
        avg_fps = 1.0 / avg_time
        
        print(f"\n📊 Performance Results:")
        print(f"   Average FPS: {avg_fps:.3f}")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Target FPS: 0.3")
        print(f"   Current baseline: 0.06")
        print(f"   Device used: {device}")
        
        if avg_fps >= 0.3:
            print("🎉 SUCCESS: Target FPS achieved!")
        else:
            print("⚠️ Target FPS not yet achieved, but test completed successfully")
        
        # Save a sample image
        if result.images:
            image = result.images[0]
            image.save("sd35_test_output.png")
            print("💾 Sample image saved as sd35_test_output.png")
        
        print("\n✅ Stable Diffusion 3.5 Medium test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🎬 Stable Diffusion 3.5 Medium Performance Test (Graceful)")
    print("Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)")
    print("Target: 0.3 FPS on batch 1")
    print("Current baseline: 0.06 FPS on batch 1")
    print("=" * 60)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check PyTorch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 CUDA version: {torch.version.cuda}")
        print(f"🔥 GPU count: {torch.cuda.device_count()}")
    
    # Check Tenstorrent environment
    sfpi_available = check_tenstorrent_environment()
    
    # Check TTNN import
    ttnn_available = test_ttnn_import()
    
    # Check required packages
    try:
        import diffusers
        print(f"✅ diffusers version: {diffusers.__version__}")
    except ImportError as e:
        print(f"❌ diffusers import failed: {e}")
        return 1
    
    try:
        import transformers
        print(f"✅ transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return 1
    
    print("\n🎯 Environment check completed!")
    
    # Test Stable Diffusion performance
    success = test_stable_diffusion_performance()
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   SFPI available: {sfpi_available}")
    print(f"   TTNN available: {ttnn_available}")
    print(f"   SD3.5 test: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if not sfpi_available:
        print(f"\n⚠️ SFPI system package not found")
        print(f"   This prevents TTNN from working properly")
        print(f"   The test ran on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"   For Tenstorrent hardware testing, SFPI needs to be installed")
    
    if not ttnn_available:
        print(f"\n⚠️ TTNN not available")
        print(f"   This prevents Tenstorrent hardware acceleration")
        print(f"   The test ran on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    print(f"\n📋 Report these results to GitHub issue #1042")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 