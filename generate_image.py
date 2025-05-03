# generate_image.py

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os

def main():
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model ID
    model_id = "CompVis/stable-diffusion-v1-4"

    # Load pipeline
    print("Loading Stable Diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline = pipeline.to(device)

    # Prepare output directory
    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)

    # Define prompt
    prompt = "A beautiful futuristic city at sunset, ultra realistic, high resolution"

    # Generate image
    print(f"Generating image for prompt: \"{prompt}\"")
    with torch.autocast(device_type=device):
        image = pipeline(prompt, guidance_scale=7.5).images[0]

    # Save the image
    output_path = os.path.join(output_dir, "futuristic_city.png")
    image.save(output_path)
    print(f"✅ Image saved at: {output_path}")

    # Optionally display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
