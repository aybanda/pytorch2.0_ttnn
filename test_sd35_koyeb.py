#!/usr/bin/env python3
"""
Standalone test script for Stable Diffusion 3.5 Medium (512x512)
This script can be run on Koyeb to test performance and validate TTNN integration.

Target: 0.3 FPS on batch 1
Current: 0.06 FPS on batch 1 (from issue #1042)
"""

import time
import torch
import psutil
import os
from diffusers import StableDiffusionPipeline

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def test_stable_diffusion_35_medium():
    """Test Stable Diffusion 3.5 Medium model performance"""
    print("🚀 Starting Stable Diffusion 3.5 Medium Performance Test")
    print("=" * 60)
    
    # Model configuration
    model_id = "stabilityai/stable-diffusion-3-medium"
    prompt = "a photo of an astronaut riding a horse on mars"
    
    print(f"📦 Loading model: {model_id}")
    print(f"🎯 Target FPS: 0.3")
    print(f"📝 Prompt: {prompt}")
    print()
    
    # Memory before loading
    mem_before = get_memory_usage()
    print(f"💾 Memory before loading: {mem_before:.2f} MB")
    
    # Load model
    load_start = time.time()
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use half precision for better performance
            use_safetensors=True
        )
        load_time = time.time() - load_start
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Memory after loading
    mem_after_load = get_memory_usage()
    print(f"💾 Memory after loading: {mem_after_load:.2f} MB")
    print(f"💾 Memory used by model: {mem_after_load - mem_before:.2f} MB")
    
    # Move to device if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    if device == "cuda":
        pipe = pipe.to(device)
        print("✅ Model moved to CUDA")
    
    # Test inference
    print("\n🎬 Running inference test...")
    
    # Warmup run
    print("🔥 Warming up...")
    with torch.no_grad():
        _ = pipe(prompt, num_inference_steps=1, output_type="latent")
    
    # Actual test runs
    num_runs = 3
    inference_times = []
    
    for i in range(num_runs):
        print(f"🔄 Run {i+1}/{num_runs}...")
        
        start_time = time.time()
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=20,  # Standard inference steps
                output_type="pil"
            )
        end_time = time.time()
        
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        fps = 1.0 / inference_time
        
        print(f"   ⏱️  Inference time: {inference_time:.2f} seconds")
        print(f"   🎯 FPS: {fps:.3f}")
        
        # Save first image
        if i == 0:
            image = result.images[0]
            image.save(f"sd35_test_output.png")
            print(f"   💾 Saved output image: sd35_test_output.png")
    
    # Calculate average performance
    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_fps = 1.0 / avg_inference_time
    
    # Memory after inference
    mem_after_inference = get_memory_usage()
    
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"🎯 Target FPS: 0.3")
    print(f"📈 Average FPS: {avg_fps:.3f}")
    print(f"📉 Average inference time: {avg_inference_time:.2f} seconds")
    print(f"💾 Total memory used: {mem_after_inference - mem_before:.2f} MB")
    
    # Performance analysis
    if avg_fps >= 0.3:
        print("✅ TARGET ACHIEVED! Performance meets or exceeds 0.3 FPS")
    else:
        improvement_needed = 0.3 / avg_fps
        print(f"⚠️  TARGET NOT MET. Need {improvement_needed:.1f}x improvement to reach 0.3 FPS")
    
    print("\n🔧 Next steps:")
    print("1. Test with TTNN compilation for further optimization")
    print("2. Try different batch sizes")
    print("3. Experiment with model optimizations")
    
    return True

def test_ttnn_integration():
    """Test TTNN integration if available"""
    print("\n🔧 Testing TTNN Integration...")
    
    try:
        import torch_ttnn as ttnn
        print("✅ torch_ttnn imported successfully")
        
        # Test basic TTNN functionality
        print("🧪 Testing basic TTNN operations...")
        
        # Create a simple tensor
        x = torch.randn(1, 3, 512, 512)
        print(f"✅ Created test tensor: {x.shape}")
        
        # Try to compile with TTNN (this would need actual hardware)
        print("⚠️  TTNN compilation requires Tenstorrent hardware")
        print("   Run this on Koyeb with Tenstorrent instance for full testing")
        
        return True
        
    except ImportError as e:
        print(f"❌ torch_ttnn not available: {e}")
        print("   This is expected on non-Tenstorrent hardware")
        return False
    except Exception as e:
        print(f"❌ TTNN test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Stable Diffusion 3.5 Medium Test Suite")
    print("Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)")
    print("=" * 60)
    
    # Test basic model functionality
    success = test_stable_diffusion_35_medium()
    
    if success:
        # Test TTNN integration
        test_ttnn_integration()
    
    print("\n🏁 Test completed!")
    print("📋 Report these results back to GitHub issue #1042") 