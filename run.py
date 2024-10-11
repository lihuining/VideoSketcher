import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed

sys.path.append(".")
sys.path.append("..")

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig, Range
from cross_image_utils import latent_utils
from cross_image_utils.latent_utils import load_latents_or_invert_images


@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)


def run(cfg: RunConfig) -> List[Image.Image]:
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)
    model = AppearanceTransferModel(cfg)
    latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=cfg)
    # latents_app:(101,4,64,64) noise_app:(100,4,64,64)
    model.set_latents(latents_app, latents_struct)
    model.set_noise(noise_app, noise_struct)
    print("Running appearance transfer...")
    images = run_appearance_transfer(model=model, cfg=cfg)
    # # prompt为列表形式
    # show_cross_attention(model.pipe.tokenizer,[cfg.prompt],os.path.join(cfg.cross_attention,"cross.png"),model.controller, res=16, from_where=("up", "down"))
    print("Done.")
    return images


def run_appearance_transfer(model: AppearanceTransferModel, cfg: RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg) # [out,app,struct]
    # init_latents:init_latents = torch.stack([model.latents_struct, model.latents_app, model.latents_struct]) torch.Size([3, 4, 64, 64])
    # init_zs: list:3,torch.Size([68, 4, 64, 64])
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start) # 10
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end) # 90
    images = model.pipe(
        prompt=[cfg.prompt] * 3,
        latents=init_latents,
        guidance_scale=1.0,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        generator=torch.Generator('cuda').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
        zs=init_zs,
    ).images # 注意这里guidance_scale = 1
    # Save images
    images[0].save(cfg.output_path / f"out_transfer_seed_{cfg.seed}.png")
    images[1].save(cfg.output_path / f"out_style_seed_{cfg.seed}.png")
    images[2].save(cfg.output_path / f"out_struct_seed_{cfg.seed}.png")
    joined_images = np.concatenate(images[::-1], axis=1)
    Image.fromarray(joined_images).save(cfg.output_path / f"out_joined_seed_{cfg.seed}.png")
    return images


if __name__ == '__main__':
    main()
'''
--app_image_path
/media/allenyljiang/564AFA804AFA5BE51/Codes/mixsa/data/style_sketch/ref_sample3.jpg
--struct_image_path
/media/allenyljiang/564AFA804AFA5BE51/Codes/mixsa/data/ref2sketch_dataset/Em8CEBGVcAEGlt6.jfif
--domain_name
animal
--use_masked_adain
True
--contrast_strength
1.67
--swap_guidance_scale
1.0
--gamma
0.75
'''