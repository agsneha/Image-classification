from diffusers import StableDiffusionPipeline
import torch

# Loading pre-trained Stable Diffusion model to create image using noise
model_id = "CompVis/stable-diffusion-v1-4"  # Model ID from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)  # float32 for CPU
pipe = pipe.to("cpu")

# A prompt for image generation
prompt = "A scenic landscape with mountains, trees, small house, and a river during sunset"

# Generating image
image = pipe(prompt).images[0]

# Save
image.save("/Users/snehaagrawal/Documents/SEM 3/Advance ML/Assignments/1/Task-1/First_Image.png")
image.show()



