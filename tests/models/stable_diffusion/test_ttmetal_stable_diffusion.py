import sys
import os
import pytest

print("[TT-METAL SD1.4 PERF] Test file loaded.")

# Add vendored demo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../external/tt_metal_stable_diffusion/demo')))

try:
    from demo import run_demo_inference
    import ttnn
    ttnn_available = True
except ImportError as e:
    print(f"[TT-METAL SD1.4 PERF] ImportError: {e}")
    ttnn_available = False

@pytest.mark.skipif(not ttnn_available, reason="ttnn not available or not importable")
def test_ttmetal_stable_diffusion_performance():
    print("[TT-METAL SD1.4 PERF] Starting performance test...")
    # Initialize the actual TTNN device object
    device = ttnn.open_device(0)  # Use device 0; change index if needed
    reset_seeds = True
    input_path = None  # The demo loads prompts internally if None
    num_prompts = 1
    num_inference_steps = 4  # Minimal steps for quick test; adjust for real perf
    image_size = (512, 512)
    try:
        fps = run_demo_inference(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size)
        print(f"[TT-METAL SD1.4 PERF] Measured FPS: {fps:.3f} (target: 0.3)")
        assert fps > 0.05, f"FPS too low: {fps:.3f}"
    except Exception as e:
        print(f"[TT-METAL SD1.4 PERF] Exception during test: {e}")
        raise
    print("[TT-METAL SD1.4 PERF] Test completed.") 