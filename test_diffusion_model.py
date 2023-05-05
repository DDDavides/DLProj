# !pip install diffusers transformers accelerate scipy safetensors
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Reduce the amaount of memory used by the model
pipe.enable_attention_slicing()

prompt = "A car from the top"
for i, image in enumerate(pipe(prompt, num_images_per_prompt=2).images):
    file_name = prompt.lower().replace(" ", "_")
    image.save(f"{file_name}_{i}.png")
