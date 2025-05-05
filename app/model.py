import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

MODEL_PATH = "models/Realistic_Vision_V5.1.safetensors"

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype =torch.float32

    print(f"Loading model from {MODEL_PATH} on {device.upper()} with dtype={dtype}...")

    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe = pipe.to(device)
    return pipe


def load_img2img_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading model from {MODEL_PATH} on {device.upper()} with dtype={dtype}...")

    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe = pipe.to(device)
    return pipe