#!/usr/bin/env python3
"""
Simple test script to verify environment setup
"""

import sys
import torch
import os

def main():
    print("🔍 Environment Test Starting...")
    print("=" * 50)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check PyTorch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 CUDA version: {torch.version.cuda}")
        print(f"🔥 GPU count: {torch.cuda.device_count()}")
    
    # Check TTNN
    try:
        import torch_ttnn as ttnn
        print("✅ torch_ttnn imported successfully")
        
        # Check TTNN devices
        devices = ttnn.get_devices()
        print(f"🔍 Found {len(devices)} TTNN device(s)")
        for i, device in enumerate(devices):
            print(f"   Device {i}: {device}")
            
    except ImportError as e:
        print(f"❌ torch_ttnn import failed: {e}")
    except Exception as e:
        print(f"❌ TTNN error: {e}")
    
    # Check diffusers
    try:
        import diffusers
        print(f"✅ diffusers version: {diffusers.__version__}")
    except ImportError as e:
        print(f"❌ diffusers import failed: {e}")
    
    # Check transformers
    try:
        import transformers
        print(f"✅ transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
    
    # Check psutil
    try:
        import psutil
        print(f"✅ psutil version: {psutil.__version__}")
    except ImportError as e:
        print(f"❌ psutil import failed: {e}")
    
    print("\n🎯 Environment test completed!")
    
    # Try to load a small model to test basic functionality
    try:
        from diffusers import StableDiffusionPipeline
        print("\n🧪 Testing model loading...")
        
        # Load a small test model
        model_id = "stabilityai/stable-diffusion-3-medium"
        print(f"📦 Loading model: {model_id}")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        print("✅ Model loaded successfully!")
        
        # Test basic inference
        print("🔄 Testing basic inference...")
        prompt = "a photo of an astronaut riding a horse on mars"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        with torch.no_grad():
            result = pipe(prompt, num_inference_steps=1, output_type="latent")
        
        print("✅ Basic inference test passed!")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main() 