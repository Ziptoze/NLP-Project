import os
import torch
from diffusers import StableDiffusionPipeline

MODEL_DIR = "models/stable-diffusion-v2-1"

def download_model():
    if not os.path.exists(MODEL_DIR):
        print("Downloading Stable Diffusion v2.1...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float32,
            use_safetensors=False,
            safety_checker=None,
            requires_safety_checker=False,
        )
        os.makedirs(MODEL_DIR, exist_ok=True)
        pipe.save_pretrained(MODEL_DIR)
    else:
        print("Model already downloaded.")

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device.upper()}...")

    # Always load the model first in float32
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
    )

    # Move to GPU and optionally cast to float16 only if cuda available
    if device == "cuda":
        pipe = pipe.to(torch.device("cuda"))
    else:
        pipe = pipe.to(torch.device("cpu"))
        pipe.unet = pipe.unet.half()  # just the UNet cast to float16 for speed

    return pipe
