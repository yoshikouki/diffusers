from datetime import datetime
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained(
    "gsdf/Counterfeit-V2.5",
    torch_dtype=torch.float16
    ).to("mps")

# Recommended if your computer has < 64 GB of RAM
# pipe.enable_attention_slicing()

prompt = """
((masterpiece,best quality)), 1girl, food, fruit, solo, skirt, shop, indoors, jacket, shopping, basket, jewelry, shirt, shelf, short hair, black hair, plaid skirt, black jacket, dutch angle, yellow eyes, looking at viewer
Negative prompt: EasyNegative, extra fingers,fewer fingers,
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 10, Size: 864x512, Denoising strength: 0.58, Hires upscale: 1.8, Hires upscaler: Latent
"""

# First-time "warmup" pass (see explanation above)
_ = pipeline(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
images = pipeline(prompt, num_inference_steps=50).images
now = datetime.now()
for idx, image in enumerate(images):
    file_name = str(now.year) \
      + str(now.month).zfill(2) \
      + str(now.day).zfill(2) \
      + "-" + str(now.hour).zfill(2) \
      + str(now.minute).zfill(2) \
      + "-" + str(idx)
    image.save(f"images/{file_name}.png")
