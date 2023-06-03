from diffusers import DiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "sport car"

# First-time "warmup" pass if PyTorch version is 1.13 (see explanation above)
# _ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
image = pipe(prompt).images[0]
image.save("image/" + prompt + ".png")

prompt = "sport car from the top"
model_id = "runwayml/stable-diffusion-v1-5"
pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
pipe2 = pipe2.to("mps")

image = image.convert("RGB")

images = pipe2(image=image, prompt=prompt, num_inference_steps=100, strength=0.75, guidance_scale=7.5).images
images[0].save("image/" + prompt + "_2.png")