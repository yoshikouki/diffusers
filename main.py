from datetime import datetime
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

prompt = ",".join([
  "1girl"
])
negative_prompt = ",".join([
  "worst quality",
  "low quality",
  "jpeg artifacts",
  "deformed",
  "bad anatomy",
  "disfigured",
  "mutation",
  "mutated",
  "extra limbs",
  "ugly",
  "fat",
  "missing limb",
  "floating limbs",
  "disconnected limbs",
  "long neck",
  "long body",
  "part of the head",
  "mutated hands and fingers",
  "intricate human hands fingers",
  "poorly drawn hands",
  "malformed hands",
  "poorly drawn face",
  "poorly drawn asymmetrical eyes",
  "low contrast",
  "cmyk",
  "greyscale",
  "flat color",
  "comic",
  "manga",
  "split screen",
  "monochrome",
  "dusty sunbeams",
  "text",
  "title",
  "logo",
  "signature",
  "animal",
  "furry",
])

model_name = "gsdf/Counterfeit-V2.5"
outputs_directory = "./images"
# 実行回数
number_of_attempts = 100
device = "mps"
num_images_per_prompt = 1
height = 864
width = 512
num_inference_steps = 50
guidance_scale = 15

pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
# pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

# Recommended if your computer has < 64 GB of RAM
# pipe.enable_attention_slicing()

# First-time "warmup" pass (see explanation above)
_ = pipeline(prompt, num_inference_steps=1)

i = 1
for _ in range(number_of_attempts):
    print(f"ℹ️ Creating image... ({i}/{number_of_attempts})")

    # Results match those from the CPU device after the warmup pass.
    images = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    ).images

    for _, image in enumerate(images):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = f"{outputs_directory}/{timestamp}-{i}.png"
        image.save(file_path)
        i += 1

    print(f"🎉 Created {file_path}")
