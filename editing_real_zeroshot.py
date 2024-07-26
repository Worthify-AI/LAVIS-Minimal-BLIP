"""Workflow for editing real images with zero-shot subject swapping."""

import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

if not torch.cuda.is_available():
    raise RuntimeError("Could not access CUDA device.")

model, vis_preprocess, txt_preprocess = load_model_and_preprocess(
    "blip_diffusion",
    "base",
    device="cuda",
    is_eval=True,
)

cond_subject = "cheeseburger"
src_subject = "boat"
tgt_subject = "cheeseburger"

text_prompt = "floating on water"

src_subject = txt_preprocess["eval"](src_subject)
tgt_subject = txt_preprocess["eval"](tgt_subject)
cond_subject = txt_preprocess["eval"](cond_subject)
text_prompt = [txt_preprocess["eval"](text_prompt)]

cond_image = Image.open("/home/austin/Downloads/hamburger.jpg").convert("RGB")

cond_image = vis_preprocess["eval"](cond_image).unsqueeze(0).cuda()

src_image = Image.open("/home/austin/Downloads/drink.JPG").convert("RGB")

samples = {
    "cond_images": cond_image,
    "cond_subject": cond_subject,
    "src_subject": src_subject,
    "tgt_subject": tgt_subject,
    "prompt": text_prompt,
    "raw_image": src_image,
}
iter_seed = 88871
guidance_scale = 7.5
num_inference_steps = 50
num_inversion_steps = 50  # increase to improve DDIM inversion quality
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

output = model.edit(
    samples,
    seed=iter_seed,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    num_inversion_steps=num_inversion_steps,
    neg_prompt=negative_prompt,
)

print("=" * 30)
print("Before editing:")
output[0].show()

print("After editing:")
output[1].show()
