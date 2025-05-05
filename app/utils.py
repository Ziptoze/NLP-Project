import torch
from model import load_model, load_img2img_model
from datetime import datetime
import os
from PIL import Image
from io import BytesIO
import base64
from enhancement import download_models, translate_to_english, enhance_prompt

download_models()
pipe = load_model()
img2img_pipe = load_img2img_model()

def generate_image(prompt, context="", num_inference_steps=50, guidance_scale=7.5):
    prompt = translate_to_english(prompt)
    context = translate_to_english(context) if context else ""
    full_prompt = enhance_prompt(prompt) + (". " + context if context else "")

    with torch.no_grad():
        image = pipe(prompt=full_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

    return save_image(image)

def stylize_image(init_image_path, style, num_inference_steps=50, strength=0.8):
    prompt_map = {
        "Ghibli": "a magical scene in the style of Studio Ghibli, whimsical and colorful, anime art",
        "Realistic": "ultra-realistic photograph, 8k, detailed lighting, lifelike textures",
        "Colorize": "vibrant colors, bright and vivid tones, high saturation, cinematic lighting",
        "Anime": "anime style, clean lines, cel shading, high detail, character focus",
        "Cyberpunk": "cyberpunk aesthetic, neon lights, dystopian city, high-tech, dark atmosphere",
        "Fantasy": "epic fantasy painting, mystical creatures, dramatic lighting, concept art",
        "Van Gogh": "in the style of Vincent Van Gogh, swirling brushstrokes, post-impressionist",
        "Watercolor": "soft watercolor painting, pastel tones, gentle transitions, paper texture",
        "Oil Painting": "oil painting on canvas, thick brushstrokes, renaissance style",
        "Comic Book": "comic book art, bold ink lines, halftone shading, dynamic poses",
        "3D Render": "3D rendered image, cinematic lighting, unreal engine graphics, hyperrealism",
        "Sketch": "pencil sketch, black and white, cross-hatching, line art",
        "Synthwave": "synthwave retro aesthetic, purple and pink neon, 80s grid, futuristic vibes",
        "Pixel Art": "pixel art style, low resolution, 16-bit video game aesthetic",
        "Surrealism": "surreal artwork, dream-like, Salvador Dali style, melting elements",
        "Steampunk": "steampunk theme, brass and gears, Victorian sci-fi, intricate machinery",
        "Noir": "film noir style, black and white, dramatic shadows, vintage detective vibe"
    }

    prompt = prompt_map.get(style, "highly detailed art")

    image = Image.open(init_image_path).convert("RGB").resize((512, 512))

    with torch.no_grad():
        out_image = img2img_pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps
        ).images[0]

    return save_image(out_image)



def save_image(image):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"stylized_{timestamp}.png")
    image.save(file_path)
    return file_path
