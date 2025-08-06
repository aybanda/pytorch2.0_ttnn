#!/usr/bin/env python3
"""
Stable Diffusion 3.5 Medium Performance Test on Tenstorrent Hardware
Focused on GitHub Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)
Target: 0.3 FPS on batch 1
Current baseline: 0.06 FPS on batch 1

This script uses proper Tenstorrent device configuration.
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

def get_dispatch_core_config():
    """Get proper Tenstorrent dispatch core configuration"""
    try:
        import ttnn
        # Use ETH core type for better performance on N300
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH
        # Use ROW axis for Blackhole/Wormhole
        dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
        dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)
        return dispatch_core_config
    except Exception as e:
        print(f"⚠️ Could not create dispatch core config: {e}")
        return None

def setup_tenstorrent_device():
    """Setup Tenstorrent device with proper configuration"""
    try:
        import ttnn
        
        print("🔧 Setting up Tenstorrent device...")
        
        # Get dispatch core configuration
        dispatch_core_config = get_dispatch_core_config()
        if not dispatch_core_config:
            return None
            
        # L1 small size for N300
        l1_small_size = 65536
        
        # Open device with proper configuration
        device = ttnn.open_device(
            device_id=0, 
            dispatch_core_config=dispatch_core_config, 
            l1_small_size=l1_small_size
        )
        
        # Enable program cache for better performance
        device.enable_program_cache()
        
        # Set as default device
        ttnn.SetDefaultDevice(device)
        
        print("✅ Tenstorrent device setup complete")
        return device
        
    except Exception as e:
        print(f"❌ Tenstorrent device setup failed: {e}")
        return None

def main():
    print("🎬 Stable Diffusion 3.5 Medium Performance Test on Tenstorrent")
    print("Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)")
    print("Target: 0.3 FPS on batch 1")
    print("Current baseline: 0.06 FPS on batch 1")
    print("=" * 60)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check PyTorch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    
    # Check TTNN
    try:
        import ttnn
        print("✅ TTNN imported successfully")
    except ImportError as e:
        print(f"❌ TTNN import failed: {e}")
        return 1
    
    # Setup Tenstorrent device
    device = setup_tenstorrent_device()
    if not device:
        print("❌ Failed to setup Tenstorrent device")
        return 1
    
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
    
    # Test Stable Diffusion 3.5 Medium performance
    try:
        from diffusers import StableDiffusionPipeline
        print("\n🧪 Testing Stable Diffusion 3.5 Medium on Tenstorrent...")
        
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
        
        # Move model to Tenstorrent device
        print("🔄 Moving model to Tenstorrent device...")
        pipe = pipe.to(device)
        print(f"🖥️ Model moved to Tenstorrent device")
        
        # Test basic inference
        print("\n🔄 Testing inference performance on Tenstorrent...")
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
        
        print(f"\n📊 Performance Results on Tenstorrent:")
        print(f"   Average FPS: {avg_fps:.3f}")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Target FPS: 0.3")
        print(f"   Current baseline: 0.06")
        
        if avg_fps >= 0.3:
            print("🎉 SUCCESS: Target FPS achieved on Tenstorrent!")
        else:
            print("⚠️ Target FPS not yet achieved, but test completed successfully")
        
        # Save a sample image
        if result.images:
            image = result.images[0]
            image.save("sd35_tenstorrent_output.png")
            print("💾 Sample image saved as sd35_tenstorrent_output.png")
        
        print("\n✅ Stable Diffusion 3.5 Medium test on Tenstorrent completed!")
        
        # Cleanup
        ttnn.synchronize_device(device)
        ttnn.close_device(device)
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n🎉 All tests completed successfully!")
    print("📋 Report these results to GitHub issue #1042")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 