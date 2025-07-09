import sys
import os
import pytest

# Add vendored demo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../external/tt_metal_stable_diffusion/demo')))

from demo import run_demo_inference

@pytest.mark.compilation_xfail(reason="Requires Tenstorrent hardware and ttnn stack")
def test_ttmetal_stable_diffusion_performance(record_property):
    # Minimal setup: use CPU device string as placeholder, update as needed for your env
    device = "cpu"  # Replace with actual device if available
    reset_seeds = True
    input_path = None  # The demo loads prompts internally if None
    num_prompts = 2  # 2 to skip compile time for FPS
    num_inference_steps = 20  # Reasonable default for test
    image_size = (512, 512)

    # Run and capture stdout
    import io
    import contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        run_demo_inference(device, reset_seeds, input_path, num_prompts, num_inference_steps, image_size)
    output = f.getvalue()

    # Parse FPS from output
    import re
    match = re.search(r"FPS: ([0-9.]+)", output)
    fps = float(match.group(1)) if match else None
    print(f"Measured FPS: {fps}")
    record_property("ttmetal_fps", fps)
    assert fps is not None and fps > 0, "FPS should be positive and present in output" 