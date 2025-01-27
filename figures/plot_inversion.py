import torch
from tensorboard.compat.tensorflow_stub.dtypes import float32

from diffusers import StableDiffusionPipeline
from utils import load_video, prepare_depth, save_frames, control_preprocess

model_id = "/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stable-diffusion-2-1-base"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None).to(device)
vae = pipe.vae
@torch.no_grad()
def decode_latents(latents):
    with torch.autocast(device_type=device, dtype=torch.float32):
        latents = 1 / 0.18215 * latents
        imgs = vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs


for i in range(991,1,-10):
    latent_path = f"/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/surf/latents/stable-diffusion-v1-5/noisy_latents_{i}.pt"

    latent = torch.load(latent_path)
    recon_save_path = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/figures/noisy_inversion"
    recon_frames = decode_latents(latent)
    save_frames(recon_frames, recon_save_path)