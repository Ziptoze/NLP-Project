# app/utils.py

import torch
from model import load_model
from datetime import datetime
import os
from enhancement import download_models, translate_to_english, enhance_prompt

# Download enhancement models on startup
download_models()

# Load diffusion model once
pipe = load_model()

def generate_image(prompt, context="", num_inference_steps=50, guidance_scale=7.5):
    # Step 1: Translate if needed
    prompt = translate_to_english(prompt)
    context = translate_to_english(context) if context else ""

    # Step 2: Enhance the prompt
    enhanced_prompt = enhance_prompt(prompt)

    # Step 3: Combine enhanced prompt and context
    full_prompt = f"{enhanced_prompt}. {context}" if context else enhanced_prompt

    with torch.no_grad():
        image = pipe(
            prompt=full_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"generated_{timestamp}.png")
    image.save(file_path)

    return file_path

