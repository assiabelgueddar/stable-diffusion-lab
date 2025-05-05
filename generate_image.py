import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os
from huggingface_hub import login

def main():
    # Authenticate with Hugging Face
    login()  # Will prompt you to enter your HF token

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model ID
    model_id = "CompVis/stable-diffusion-v1-4"

    # Load pipeline
    print("Loading Stable Diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline = pipeline.to(device)

    # Output directory
    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)

    # Define prompt
    prompt = "A beautiful futuristic city at sunset, ultra realistic, high resolution"

    # Generate image
    print(f"Generating image for prompt: \"{prompt}\"")
    with torch.autocast(device_type=device):
        image = pipeline(prompt, guidance_scale=7.5).images[0]

    # Save image
    output_path = os.path.join(output_dir, "futuristic_city.png")
    image.save(output_path)
    print(f"✅ Image saved at: {output_path}")

    # Display
    plt.imshow(image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
