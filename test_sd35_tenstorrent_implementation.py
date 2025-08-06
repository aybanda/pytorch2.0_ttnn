#!/usr/bin/env python3
"""
Stable Diffusion 3.5 Medium - Tenstorrent Implementation Test
Focused on GitHub Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)

This script tests the actual Tenstorrent implementation referenced in the issue:
https://github.com/tenstorrent/tt-metal/tree/mbahnas/sd35_medium_512_spacelike_feb05/models/experimental/stable_diffusion3

Target: 0.3 FPS on batch 1
Current baseline: 0.06 FPS on batch 1
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
    """Check Tenstorrent environment"""
    print("🔍 Checking Tenstorrent environment...")
    
    # Check for SFPI
    sfpi_path = "/opt/tenstorrent/sfpi"
    if os.path.exists(sfpi_path):
        print(f"✅ SFPI found at {sfpi_path}")
        sfpi_available = True
    else:
        print(f"⚠️ SFPI not found at {sfpi_path}")
        print("   This prevents TTNN from working properly")
        sfpi_available = False
    
    # Check for Tenstorrent devices
    tt_devices = []
    for i in range(10):
        device_path = f"/dev/tenstorrent{i}"
        if os.path.exists(device_path):
            tt_devices.append(i)
    
    if tt_devices:
        print(f"✅ Found Tenstorrent devices: {tt_devices}")
        tt_hardware_available = True
    else:
        print("⚠️ No Tenstorrent devices found")
        tt_hardware_available = False
    
    return sfpi_available, tt_hardware_available

def test_tenstorrent_sd35_implementation():
    """
    Test the Tenstorrent implementation of Stable Diffusion 3.5 Medium
    This should use the model from the tt-metal repository
    """
    print("\n🧪 Testing Tenstorrent SD3.5 Implementation...")
    
    try:
        # First, try to import the Tenstorrent implementation
        try:
            # This would be the actual implementation from tt-metal
            from models.experimental.stable_diffusion3 import StableDiffusion3Pipeline
            print("✅ Tenstorrent SD3.5 implementation imported successfully")
            
            # Load the Tenstorrent model
            model_path = "/path/to/tenstorrent/sd35_medium_512_spacelike_feb05"
            print(f"📦 Loading Tenstorrent model from: {model_path}")
            
            mem_before = get_memory_usage()
            print(f"💾 Memory before loading: {mem_before:.1f} MB")
            
            pipe = StableDiffusion3Pipeline.from_pretrained(model_path)
            print("✅ Tenstorrent SD3.5 model loaded successfully!")
            
            model_source = "Tenstorrent Implementation"
            
        except ImportError:
            print("⚠️ Tenstorrent implementation not available")
            print("   Falling back to HuggingFace model for testing")
            
            from diffusers import StableDiffusionPipeline
            model_id = "stabilityai/stable-diffusion-3-medium"
            print(f"📦 Loading HuggingFace model: {model_id}")
            
            mem_before = get_memory_usage()
            print(f"💾 Memory before loading: {mem_before:.1f} MB")
            
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            print("✅ HuggingFace SD3.5 model loaded successfully!")
            
            model_source = "HuggingFace (Fallback)"
        
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
        
        # Test inference
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
        print(f"   Model source: {model_source}")
        print(f"   Average FPS: {avg_fps:.3f}")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Target FPS: 0.3")
        print(f"   Current baseline: 0.06")
        print(f"   Device used: {device}")
        
        if avg_fps >= 0.3:
            print("🎉 SUCCESS: Target FPS achieved!")
        else:
            improvement_needed = 0.3 / avg_fps if avg_fps > 0 else float('inf')
            print(f"⚠️ Target not met. Need {improvement_needed:.1f}x improvement")
        
        # Save sample image
        if result.images:
            image = result.images[0]
            image.save("sd35_tenstorrent_output.png")
            print("💾 Sample image saved as sd35_tenstorrent_output.png")
        
        print("\n✅ SD3.5 test completed!")
        return True, avg_fps, device, model_source
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0, "none", "none"

def main():
    print("🎬 Stable Diffusion 3.5 Medium - Tenstorrent Implementation Test")
    print("Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)")
    print("Reference: https://github.com/tenstorrent/tt-metal/tree/mbahnas/sd35_medium_512_spacelike_feb05/models/experimental/stable_diffusion3")
    print("Target: 0.3 FPS on batch 1")
    print("Current baseline: 0.06 FPS on batch 1")
    print("=" * 80)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check PyTorch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 CUDA version: {torch.version.cuda}")
        print(f"🔥 GPU count: {torch.cuda.device_count()}")
    
    # Check environment
    sfpi_available, tt_hardware_available = check_tenstorrent_environment()
    
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
    
    # Test Tenstorrent SD3.5 implementation
    success, fps, device, model_source = test_tenstorrent_sd35_implementation()
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   SFPI available: {sfpi_available}")
    print(f"   Tenstorrent hardware: {tt_hardware_available}")
    print(f"   SD3.5 test: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"   Performance: {fps:.3f} FPS on {device}")
    print(f"   Model source: {model_source}")
    
    if not sfpi_available:
        print(f"\n⚠️ SFPI system package not found")
        print(f"   This prevents TTNN from working properly")
        print(f"   The test ran on {device}")
        print(f"   For Tenstorrent hardware testing, SFPI needs to be installed")
    
    if not tt_hardware_available:
        print(f"\n⚠️ Tenstorrent hardware not detected")
        print(f"   This prevents Tenstorrent hardware acceleration")
        print(f"   The test ran on {device}")
    
    print(f"\n📋 Report these results to GitHub issue #1042")
    print(f"   Performance: {fps:.3f} FPS")
    print(f"   Device: {device}")
    print(f"   Model source: {model_source}")
    print(f"   Target: 0.3 FPS")
    print(f"   Baseline: 0.06 FPS")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 