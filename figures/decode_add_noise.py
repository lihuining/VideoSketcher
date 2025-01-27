import os
from PIL import Image
import torch
from torchvision import transforms
from diffusers import DDIMScheduler, StableDiffusionPipeline
def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs, percent):
    if step_index == int(pipe.num_timesteps * percent):
        prompt_embeds = callback_kwargs['prompt_embeds']
        prompt_embeds = prompt_embeds.chunk(2)[-1]
        pipe._guidance_scale = 0.0
        callback_kwargs['prompt_embeds'] = prompt_embeds
    return callback_kwargs
pipeline = StableDiffusionPipeline.from_pretrained("/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

generator = torch.Generator(device="cuda").manual_seed(1)
out = pipeline(
    prompt,
    generator=generator,
    callback_on_step_end=callback_dynamic_cfg,
    callback_on_step_end_tensor_inputs=['prompt_embeds']
)

out.images[0].save("out_custom_cfg.png")


def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60, -5, 15, -50),
        (60, 10, -5, -35)
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(
        -1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)

    return Image.fromarray(image_array)


def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]

    image = latents_to_rgb(latents)
    image.save(f"{step}.png")

    return callback_kwargs
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

image = pipeline(
    prompt = "A croissant shaped like a cute bear.",
    negative_prompt = "Deformed, ugly, bad anatomy",
    callback_on_step_end=decode_tensors,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]
