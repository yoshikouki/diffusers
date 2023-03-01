from datetime import datetime
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from PIL import Image

prompt = ",".join([
    "extremely detailed 8k",
    "1girl",
    "high contrast",
    "colorful",
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
    "bloom",
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

device = "mps"
model_name = "gsdf/Counterfeit-V2.5"
num_images_per_prompt = 2
number_of_attempts = 100

# load the pipeline
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

init_file_path = "./refine_targets/20230301-091946-1.jpg"
init_image = Image.open(init_file_path)

i = 1
for _ in range(number_of_attempts):
    print(f"ℹ️ Refining image... ({i}/{number_of_attempts})")
    images = pipeline(
        prompt,
        image=init_image,
        negative_prompt=negative_prompt,
        strength=0.7,
        num_inference_steps=100,
        guidance_scale=7.5,
    ).images

    for _, image in enumerate(images):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = f"./refines/20230301-091946-1-{timestamp}-{i}.jpg"
        image.save(file_path)
        i += 1
        print(f"🎉 Refined {file_path}")
