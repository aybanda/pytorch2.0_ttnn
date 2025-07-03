# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# https://huggingface.co/runwayml/stable-diffusion-v1-5
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
        model_id = "CompVis/stable-diffusion-v1-4"
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
# Remove or comment out the skip marker below to enable the test on supported environments
# @pytest.mark.skip(reason="Dynamo cannot support pipeline.")
def test_stable_diffusion(record_property, mode):
    model_name = "Stable Diffusion"
    record_property("model_name", model_name)
    record_property("mode", mode)

    tester = ThisTester(model_name, mode)
    # Measure memory before
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
    # Measure memory after
    if psutil:
        mem_after = process.memory_info().rss / 1024 / 1024  # in MB
    else:
        mem_after = None
    elapsed = end - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    record_property("inference_time_sec", elapsed)
    record_property("fps", fps)
    if mem_before is not None and mem_after is not None:
        record_property("memory_usage_mb", mem_after - mem_before)
    record_property("torch_ttnn", (tester, results))
