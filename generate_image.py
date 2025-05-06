from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("assiabelgueddar/stable-diffusion-custom-cat")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

image = pipe("a fluffy white cat sitting in a garden").images[0]
image.show()
