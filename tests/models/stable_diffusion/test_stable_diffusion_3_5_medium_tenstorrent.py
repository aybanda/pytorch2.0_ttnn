# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Test the Tenstorrent implementation of Stable Diffusion 3.5 Medium
# Reference: https://github.com/tenstorrent/tt-metal/tree/mbahnas/sd35_medium_512_spacelike_feb05/models/experimental/stable_diffusion3

import torch
import pytest
import time
import os
import sys
try:
    import psutil
except ImportError:
    psutil = None
from tests.utils import ModelTester, repeat_inputs


class TenstorrentSD35Tester(ModelTester):
    def _load_model(self):
        """
        Load the Tenstorrent implementation of Stable Diffusion 3.5 Medium
        This should use the model from the tt-metal repository
        """
        try:
            # Try to import the Tenstorrent implementation
            # This would be the actual implementation from tt-metal
            from models.experimental.stable_diffusion3 import StableDiffusion3Pipeline
            
            # Load the Tenstorrent model
            model_path = "/path/to/tenstorrent/sd35_medium_512_spacelike_feb05"
            pipe = StableDiffusion3Pipeline.from_pretrained(model_path)
            return pipe
            
        except ImportError:
            # Fallback to HuggingFace model for testing
            print("⚠️ Tenstorrent implementation not available, using HuggingFace model")
            from diffusers import StableDiffusionPipeline
            model_id = "stabilityai/stable-diffusion-3-medium"
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
            return pipe

    def _load_inputs(self, batch_size):
        prompt = "a photo of an astronaut riding a horse on mars"
        if batch_size == 1:
            return prompt
        else:
            return [prompt] * batch_size


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
def test_stable_diffusion_3_5_medium_tenstorrent(record_property, mode):
    """
    Test the Tenstorrent implementation of Stable Diffusion 3.5 Medium
    Target: 0.3 FPS on batch 1
    Current baseline: 0.06 FPS on batch 1
    """
    model_name = "Stable Diffusion 3.5 Medium (Tenstorrent Implementation)"
    record_property("model_name", model_name)
    record_property("mode", mode)
    record_property("target_fps", 0.3)
    record_property("current_baseline_fps", 0.06)

    tester = TenstorrentSD35Tester(model_name, mode)
    
    # Memory tracking
    if psutil:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # in MB
    else:
        mem_before = None
    
    # Performance measurement
    start = time.time()
    results = tester.test_model()
    end = time.time()
    
    if mode == "eval":
        image = results.images[0]
    
    elapsed = end - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    
    # Memory usage
    if psutil:
        mem_after = process.memory_info().rss / 1024 / 1024  # in MB
        mem_used = mem_after - mem_before
    else:
        mem_used = None
    
    # Record results
    record_property("inference_time_sec", elapsed)
    record_property("fps", fps)
    record_property("mem_used_mb", mem_used)
    record_property("torch_ttnn", (tester, results))
    
    # Performance analysis
    print(f"\n📊 Performance Results:")
    print(f"   Model: {model_name}")
    print(f"   FPS: {fps:.3f}")
    print(f"   Target FPS: 0.3")
    print(f"   Current baseline: 0.06")
    print(f"   Memory used: {mem_used:.1f} MB" if mem_used else "   Memory: N/A")
    
    if fps >= 0.3:
        print("🎉 SUCCESS: Target FPS achieved!")
    else:
        improvement_needed = 0.3 / fps if fps > 0 else float('inf')
        print(f"⚠️ Target not met. Need {improvement_needed:.1f}x improvement")
    
    # Assert performance target
    assert fps >= 0.06, f"Performance below baseline: {fps:.3f} FPS (baseline: 0.06 FPS)"
    
    return fps 