# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/stabilityai/stable-diffusion-3-medium
from diffusers import StableDiffusionPipeline
import torch
import pytest
import time
try:
    import psutil
except ImportError:
    psutil = None
from tests.utils import ModelTester, repeat_inputs


class ThisTester(ModelTester):
    def _load_model(self):
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
def test_stable_diffusion_3_5_medium(record_property, mode):
    model_name = "Stable Diffusion 3.5 Medium"
    record_property("model_name", model_name)
    record_property("mode", mode)

    tester = ThisTester(model_name, mode)
    if psutil:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # in MB
    else:
        mem_before = None
    start = time.time()
    results = tester.test_model()
    end = time.time()
    if mode == "eval":
        image = results.images[0]
    elapsed = end - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    if psutil:
        mem_after = process.memory_info().rss / 1024 / 1024  # in MB
        mem_used = mem_after - mem_before
    else:
        mem_used = None
    record_property("inference_time_sec", elapsed)
    record_property("fps", fps)
    record_property("mem_used_mb", mem_used)
    record_property("torch_ttnn", (tester, results)) 