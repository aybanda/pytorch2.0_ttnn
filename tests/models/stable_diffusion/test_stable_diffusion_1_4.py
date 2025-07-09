print("Stable Diffusion 1.4 test is running!")
print("Loading test file...")

def test_simple():
    """Simple test to verify pytest is working"""
    print("Simple test is running!")
    assert True
    print("Simple test passed!")

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import pytest
from tests.utils import ModelTester, repeat_inputs


class ThisTester(ModelTester):
    def _load_model(self):
        # Load the pre-trained model and tokenizer for v1.4
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.bfloat16
        )
        self.scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        return unet

    def _load_inputs(self, batch_size):
        # Prepare the text prompt
        prompt = "A photo of an astronaut riding a horse on Mars"
        text_input = self.tokenizer(prompt, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids)[0]

        # Generate noise
        batch_size = text_embeddings.shape[0]
        height, width = 512, 512  # Output image size
        latents = torch.randn((batch_size, self.model.in_channels, height // 8, width // 8))

        # Set number of diffusion steps
        num_inference_steps = 1
        self.scheduler.set_timesteps(num_inference_steps)

        # Scale the latent noise to match the model's expected input
        latents = latents * self.scheduler.init_noise_sigma

        # Get the model's predicted noise
        latent_model_input = self.scheduler.scale_model_input(latents, 0)
        arguments = {
            "sample": latent_model_input.to(torch.bfloat16),
            "timestep": 0,
            "encoder_hidden_states": text_embeddings.to(torch.bfloat16),
        }
        arguments = repeat_inputs(arguments, batch_size)
        return arguments


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.compilation_xfail
def test_stable_diffusion_1_4(record_property, mode):
    print(f"Starting test_stable_diffusion_1_4 with mode: {mode}")
    model_name = "Stable Diffusion 1.4"
    record_property("model_name", model_name)
    record_property("mode", mode)

    print("Creating tester...")
    tester = ThisTester(model_name, mode)
    print("Running test_model()...")
    results = tester.test_model()
    print("Test completed!")
    if mode == "eval":
        noise_pred = results.sample

    record_property("torch_ttnn", (tester, results)) 


@pytest.mark.parametrize("mode", ["eval"])
@pytest.mark.compilation_xfail
def test_stable_diffusion_1_4_performance(record_property, mode):
    print(f"Starting test_stable_diffusion_1_4_performance with mode: {mode}")
    model_name = "Stable Diffusion 1.4"
    record_property("model_name", model_name)
    record_property("mode", mode)

    print("Creating tester for performance test...")
    tester = ThisTester(model_name, mode)
    try:
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # in MB
        print(f"Memory before: {mem_before:.2f} MB")
    except ImportError:
        mem_before = None
        print("psutil not available")
    import time
    start = time.time()
    print("Starting inference...")
    results = tester.test_model()
    end = time.time()
    print("Inference completed!")
    if mode == "eval":
        noise_pred = results.sample
    try:
        mem_after = process.memory_info().rss / 1024 / 1024  # in MB
        print(f"Memory after: {mem_after:.2f} MB")
    except Exception:
        mem_after = None
    elapsed = end - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    print(f"Inference time: {elapsed:.4f} seconds")
    print(f"FPS: {fps:.4f}")
    record_property("inference_time_sec", elapsed)
    record_property("fps", fps)
    if mem_before is not None and mem_after is not None:
        record_property("memory_usage_mb", mem_after - mem_before)
    record_property("torch_ttnn", (tester, results)) 