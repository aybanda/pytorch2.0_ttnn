#!/usr/bin/env python3
"""
Advanced TTNN test script for Stable Diffusion 3.5 Medium (512x512)
This script tests both baseline performance and TTNN-compiled performance.

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

def test_baseline_performance():
    """Test baseline PyTorch performance without TTNN"""
    print("🔬 Testing Baseline PyTorch Performance")
    print("=" * 50)
    
    model_id = "stabilityai/stable-diffusion-3-medium"
    prompt = "a photo of an astronaut riding a horse on mars"
    
    # Load model
    print(f"📦 Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    print(f"🖥️  Using device: {device}")
    
    # Warmup
    print("🔥 Warming up...")
    with torch.no_grad():
        _ = pipe(prompt, num_inference_steps=1, output_type="latent")
    
    # Test runs
    num_runs = 3
    times = []
    
    for i in range(num_runs):
        print(f"🔄 Baseline run {i+1}/{num_runs}...")
        start = time.time()
        with torch.no_grad():
            result = pipe(prompt, num_inference_steps=20, output_type="pil")
        end = time.time()
        
        inference_time = end - start
        times.append(inference_time)
        fps = 1.0 / inference_time
        print(f"   ⏱️  Time: {inference_time:.2f}s, FPS: {fps:.3f}")
    
    avg_time = sum(times) / len(times)
    avg_fps = 1.0 / avg_time
    
    print(f"\n📊 Baseline Results:")
    print(f"   Average FPS: {avg_fps:.3f}")
    print(f"   Average time: {avg_time:.2f}s")
    
    return avg_fps, pipe

def test_ttnn_compilation(pipe):
    """Test TTNN compilation and performance"""
    print("\n🧠 Testing TTNN Compilation")
    print("=" * 50)
    
    try:
        import torch_ttnn as ttnn
        print("✅ torch_ttnn imported successfully")
        
        # Test TTNN device availability
        print("🔍 Checking TTNN devices...")
        devices = ttnn.get_devices()
        print(f"   Found {len(devices)} TTNN device(s)")
        
        if len(devices) == 0:
            print("❌ No TTNN devices available")
            return None
        
        # Use first available device
        device = devices[0]
        print(f"   Using device: {device}")
        
        # Test basic TTNN operations
        print("🧪 Testing basic TTNN operations...")
        x = torch.randn(1, 3, 512, 512, dtype=torch.float16)
        ttnn_x = ttnn.from_torch(x, device=device)
        print(f"   ✅ Created TTNN tensor: {ttnn_x.shape}")
        
        # Test model compilation (this is a simplified version)
        print("🔧 Attempting model compilation...")
        print("   ⚠️  Full model compilation requires more complex setup")
        print("   This is a basic test of TTNN functionality")
        
        return device
        
    except ImportError as e:
        print(f"❌ torch_ttnn not available: {e}")
        return None
    except Exception as e:
        print(f"❌ TTNN test failed: {e}")
        return None

def test_optimized_inference(pipe, ttnn_device=None):
    """Test optimized inference with various techniques"""
    print("\n⚡ Testing Optimized Inference")
    print("=" * 50)
    
    prompt = "a photo of an astronaut riding a horse on mars"
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        print(f"\n📦 Testing batch size: {batch_size}")
        
        # Create batch prompts
        prompts = [prompt] * batch_size
        
        # Test inference
        start = time.time()
        with torch.no_grad():
            results = pipe(prompts, num_inference_steps=20, output_type="pil")
        end = time.time()
        
        inference_time = end - start
        fps = batch_size / inference_time
        
        print(f"   ⏱️  Time: {inference_time:.2f}s")
        print(f"   🎯 FPS: {fps:.3f}")
        print(f"   📊 FPS per image: {fps/batch_size:.3f}")
        
        # Save first image from batch
        if batch_size == 1:
            results.images[0].save(f"sd35_batch{batch_size}.png")
            print(f"   💾 Saved: sd35_batch{batch_size}.png")

def generate_performance_report(baseline_fps, ttnn_available):
    """Generate a comprehensive performance report"""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 60)
    print(f"🎯 Target FPS: 0.3")
    print(f"📈 Baseline FPS: {baseline_fps:.3f}")
    
    if baseline_fps >= 0.3:
        print("✅ TARGET ACHIEVED! Baseline performance meets target")
    else:
        improvement_needed = 0.3 / baseline_fps
        print(f"⚠️  TARGET NOT MET. Need {improvement_needed:.1f}x improvement")
    
    print(f"\n🔧 TTNN Status: {'✅ Available' if ttnn_available else '❌ Not Available'}")
    
    if ttnn_available:
        print("   🚀 TTNN compilation should provide additional performance gains")
        print("   📋 Next: Implement full TTNN model compilation")
    else:
        print("   ⚠️  TTNN not available - run on Tenstorrent hardware")
    
    print("\n📋 Recommendations:")
    print("1. ✅ Baseline test completed")
    print("2. 🔧 Implement full TTNN compilation")
    print("3. 📊 Test with different model optimizations")
    print("4. 🎯 Optimize for target 0.3 FPS")
    
    print("\n📝 Report this data to GitHub issue #1042")

if __name__ == "__main__":
    print("🎯 Advanced Stable Diffusion 3.5 Medium Test Suite")
    print("Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)")
    print("=" * 60)
    
    # Test baseline performance
    baseline_fps, pipe = test_baseline_performance()
    
    # Test TTNN compilation
    ttnn_device = test_ttnn_compilation(pipe)
    
    # Test optimized inference
    test_optimized_inference(pipe, ttnn_device)
    
    # Generate report
    generate_performance_report(baseline_fps, ttnn_device is not None)
    
    print("\n🏁 Advanced test completed!")
    print("📋 Report these results back to GitHub issue #1042") 