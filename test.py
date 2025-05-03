# Import required libraries
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os

# Check device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipeline = pipeline.to(device)

# Create output directory
output_dir = "../assets"
os.makedirs(output_dir, exist_ok=True)

# Your prompt
prompt = "A beautiful futuristic city at sunset, ultra realistic, high resolution"

# Generate image
with torch.autocast(device_type=device):
    image = pipeline(prompt, guidance_scale=7.5).images[0]

# Save image
output_path = os.path.join(output_dir, "futuristic_city.png")
image.save(output_path)
print(f"Image saved at {output_path}")

# Show image
plt.imshow(image)
plt.axis("off")
plt.show()
