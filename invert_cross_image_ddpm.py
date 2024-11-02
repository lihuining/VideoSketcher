import time
from pathlib import Path
from cross_image_utils.latent_utils import load_latents, load_noise
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from transformers import logging
from dataclasses import dataclass
from typing import Optional

from cross_image_utils.styleid_utils import encode_latent
from utils import load_config, save_config
from utils import get_controlnet_kwargs, get_latents_dir, init_model, seed_everything
from utils import load_video, prepare_depth, save_frames, control_preprocess
# import sys
# from typing import List
#
# import numpy as np
# import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig, Range
from cross_image_utils import latent_utils
from cross_image_utils.latent_utils import load_latents_or_invert_images

from cross_image_utils.model_utils import get_stable_diffusion_model
# suppress partial model loading warning
logging.set_verbosity_error()
from cross_image_utils.ddpm_inversion import get_variance

class Inverter(nn.Module):
    def __init__(self, pipe, scheduler, config,cross_image_config):
        super().__init__()
        ### cross_image_config
        self.cross_image_config = cross_image_config
        self.device = config.device
        self.model_key = config.model_key
        self.config = config

        inv_config = config.inversion
        self.num_inference_steps = inv_config.steps  # 总的timestep
        float_precision = inv_config.float_precision if "float_precision" in inv_config else config.float_precision
        if float_precision == "fp16":
            self.dtype = torch.float16
            print("[INFO] float precision fp16. Use torch.float16.")
        else:
            self.dtype = torch.float32
            print("[INFO] float precision fp32. Use torch.float32.")

        self.pipe = pipe
        self.pipe.scheduler.set_timesteps(self.num_inference_steps)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        if config.enable_xformers_memory_efficient_attention:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except ModuleNotFoundError:
                print("[WARNING] xformers not found. Disable xformers attention.")

        scheduler.set_timesteps(self.num_inference_steps)
        self.timesteps_to_save = scheduler.timesteps
        # scheduler.set_timesteps(inv_config.steps)

        self.scheduler = scheduler

        self.skip_steps = inv_config.skip_steps
        self.timesteps = self.pipe.scheduler.timesteps[self.skip_steps:] # forwrad step
        self.t_to_idx = {int(v): k for k, v in enumerate(self.timesteps)}  # key:t ,value:idx {1:67,11:66}

        self.prompt=inv_config.prompt
        self.recon=inv_config.recon

        self.save_latents=inv_config.save_intermediate
        self.batch_size = inv_config.batch_size
        self.force = inv_config.force

        self.n_frames = inv_config.n_frames

        self.frame_height, self.frame_width = config.height, config.width
        self.work_dir = config.work_dir


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt=None, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        if negative_prompt is not None:
            uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                          return_tensors='pt')
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    def load_latent(self,latent_path,choice="content"):
        '''
        加载latents以及noise
        type = latent or ddpm
        choice = content or style
        noise需要每个时刻，
        latents只需要初始值,加载latents的时候先把全部的值加载起来再说
        '''
        if choice == "style":
            noises = []  # 用于存储每个timestep的噪声
            save_path = os.path.join(latent_path,os.path.basename(self.config.app_image_path).split('.')[0])
            save_path = get_latents_dir(save_path, self.model_key)
            # timestep:[1,4,64,64]
            style_name = f'noisy_latents_{self.timesteps[0].item()}.pt' # tensor数值
            style_latent_path = os.path.join(save_path,style_name)
            style_latents = torch.load(style_latent_path)
            for t in self.timesteps:
                noise_name = f'noisy_ddpm_{t}.pt'
                noise_path = os.path.join(save_path, noise_name)
                assert os.path.exists(noise_path), f"Latent at timestep {t} not found in {save_path}."
                noise = torch.load(noise_path)
                noises.append(noise)  # 把每个噪声张量加到列表中
            # 在第一个维度上拼接噪声张量
            style_noises = torch.cat(noises, dim=0)
            return style_latents,style_noises # [1,4,64,64] [68,4,64,64]
        else:
            content_name = f'noisy_latents_{self.timesteps[0].item()}.pt' # tensor数值
            save_path = get_latents_dir(latent_path, self.model_key)
            content_latent_path = os.path.join(save_path,content_name)
            content_latents = torch.load(content_latent_path) # [10,4,64,64]
            noises = []  # 用于存储每个timestep的噪声
            for t in self.timesteps:
                noise_name = f'noisy_ddpm_{t}.pt'
                noise_path = os.path.join(save_path, noise_name)
                assert os.path.exists(noise_path), f"Latent at timestep {t} not found in {save_path}."
                noise = torch.load(noise_path)
                noises.append(noise)  # 把每个噪声张量加到列表中

            # 在第一个维度上拼接噪声张量
            content_noises = torch.stack(noises, dim=1)

            return content_latents,content_noises # [10,4,64,64] [10,68,4,64,64]
    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def decode_latents_batch(self, latents):
        imgs = []
        batch_latents = latents.split(self.batch_size, dim = 0)
        for latent in batch_latents:
            imgs += [self.decode_latents(latent)]
        imgs = torch.cat(imgs)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def encode_imgs_batch(self, imgs):
        latents = []
        batch_imgs = imgs.split(self.batch_size, dim = 0)
        for img in batch_imgs:
            latents += [self.encode_imgs(img)] # img:(8,3,512,512) 按照self.batch_size对img进行encode,直接encode 导致现存爆炸
        latents = torch.cat(latents)
        return latents # （64，4，64，64）


    def encode_text(self,model, prompts):
        text_input = model.tokenizer(
            prompts,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
        return text_encoding

    def forward_step(self,model, model_output, timestep, sample):
        next_timestep = min(self.scheduler.config.num_train_timesteps - 2,
                            timestep + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)

        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        next_sample = self.scheduler.add_noise(pred_original_sample,
                                                model_output,
                                                torch.LongTensor([next_timestep]))
        return next_sample
    def sample_xts_from_x0(self,model, x0, num_inference_steps=50):
        """
        Samples from P(x_1:T|x_0) 和原始顺序不同
        """
        # torch.manual_seed(43256465436)
        alpha_bar = model.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        alphas = model.scheduler.alphas
        betas = 1 - alphas
        variance_noise_shape = (
            num_inference_steps,
            model.unet.in_channels,
            model.unet.sample_size,
            model.unet.sample_size)
        # （100，4，64，64）
        timesteps = model.scheduler.timesteps.to(model.device)  # （991~1）
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}  # key:time value:index {1:99,11:98}
        xts = torch.zeros(variance_noise_shape).to(x0.device)  # （100，4，64，64）
        for t in reversed(timesteps):  # 从1~991
            idx = t_to_idx[int(t)]
            xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        xts = torch.cat([xts, x0], dim=0)  # idx:99 t=1

        return xts  # [101,4,64,64]
    def sample_xts_from_x0_batch(self,model, x0, num_inference_steps=50):
        """
        Samples from P(x_1:T|x_0) 和原始顺序不同
        """
        # torch.manual_seed(43256465436)

        batch = len(x0)
        alpha_bar = model.scheduler.alphas_cumprod # tensor:1000
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        alphas = model.scheduler.alphas
        betas = 1 - alphas
        variance_noise_shape = (
            batch,
            num_inference_steps,
            model.unet.in_channels,
            model.unet.sample_size,
            model.unet.sample_size)
        # （10,100，4，64，64）
        timesteps = model.scheduler.timesteps.to(model.device)  # （991~1）
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}  # key:time value:index {1:99,11:98}
        xts = torch.zeros(variance_noise_shape).to(x0.device)  # （100，4，64，64）
        for t in reversed(timesteps):  # 从1~991
            idx = t_to_idx[int(t)]
            xts[:,idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        xts = torch.cat([xts, x0.unsqueeze(1)], dim=1)  # idx:99 t=1

        return xts  # [101,4,64,64] --> [2,101,4,64,64]
    def inversion_forward_process_batch(self, x0,save_path,
                                  etas=None,
                                  prog_bar=False,
                                  prompt="",
                                  cfg_scale=3.5,
                                  num_inference_steps=50, eps=None
                                  ):
        cur_batch = len(x0)
        model = self.pipe
        num_inference_steps = self.num_inference_steps
        text_embeddings, uncond_embedding = self.prepare_cond(prompt, len(x0))
        timesteps = model.scheduler.timesteps.to(model.device)  # 991~1,100个
        variance_noise_shape = (
            cur_batch,
            num_inference_steps,
            model.unet.in_channels,
            model.unet.sample_size,
            model.unet.sample_size)  # （10，100，4，64，64）
        if etas is None or (type(etas) in [int, float] and etas == 0):
            eta_is_zero = True
            zs = None
        else:
            eta_is_zero = False
            if type(etas) in [int, float]: etas = [etas] * model.scheduler.num_inference_steps  # list:100
            xts = self.sample_xts_from_x0_batch(model, x0, num_inference_steps=num_inference_steps) # (15,101,4,64,64)
            alpha_bar = model.scheduler.alphas_cumprod
            zs = torch.zeros(size=variance_noise_shape, device=model.device)  # (100,4,64,64)

        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}  # key:time value:index {1:99,11:98}
        idx_to_t = {k:int(v)  for k, v in enumerate(timesteps)}
        xt = x0
        op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)  # 100step
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for t in op:  # t从1开始
                idx = t_to_idx[int(t)]
                # 1. predict noise residual
                if not eta_is_zero:
                    xt = xts[:,idx] # [10，4，64，64】

                x_index = torch.arange(len(x0))
                batches = x_index.split(self.batch_size, dim=0)
                noises = []
                for batch in batches:
                    with torch.no_grad():
                        out = model.unet.forward(xt[batch], timestep=t, encoder_hidden_states=uncond_embedding[batch])
                        if not prompt == "":
                            cond_out = model.unet.forward(xt[batch], timestep=t, encoder_hidden_states=text_embeddings[batch])

                        if not prompt == "":
                            ## classifier free guidance
                            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
                        else:
                            noise_pred = out.sample
                        noises += [noise_pred]
                noise_pred = torch.cat(noises) # (10,4,64,64)

                if eta_is_zero:
                    # 2. compute more noisy image and set x_t -> x_t+1
                    xt = self.forward_step(model, noise_pred, t, xt)

                else: # True
                    xtm1 = xts[:,idx + 1] # xt-1     (10,101,4,64,64）-> (10,4,64,64)
                    # pred of x0
                    pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5

                    # direction to xt
                    prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
                    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
                        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

                    variance = get_variance(model,t)
                    pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (0.5) * noise_pred

                    mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                    # z？？？
                    z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
                    zs[:,idx] = z

                    # correction to avoid error accumulation
                    xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z  # 对xts进行修正？？？
                    xts[:,idx + 1] = xtm1 # idx = 31
                    if self.save_latents and t.cpu() in self.timesteps_to_save:
                        if idx+ 1 < len(idx_to_t):
                            torch.save(xtm1, os.path.join(save_path, f'noisy_latents_{idx_to_t[idx+1]}.pt'))
                        pth = os.path.join(save_path, f'noisy_ddpm_{t.item()}.pt')
                        if idx != len(timesteps)-1:
                            torch.save(z, pth)
                        else:
                            torch.save(torch.zeros_like(z), pth)
                        print(f"[INFO] inverted latent saved to: {pth}")

        torch.save(xts[:,idx], os.path.join(save_path, f'noisy_latents_{t.item()}.pt'))
        if not zs is None:
            zs[:,-1] = torch.zeros_like(zs[:,-1])

        return xt, zs, xts
    def inversion_forward_process(self,frame_id, x0,save_path,
                                  etas=None,
                                  prog_bar=False,
                                  prompt="",
                                  cfg_scale=3.5,
                                  num_inference_steps=50, eps=None
                                  ):
        model = self.pipe
        num_inference_steps = self.num_inference_steps
        text_embeddings,uncond_embedding = self.prepare_cond(prompt,len(x0))
        timesteps = model.scheduler.timesteps.to(model.device)  # 991~1,100个
        variance_noise_shape = (
            num_inference_steps,
            model.unet.in_channels,
            model.unet.sample_size,
            model.unet.sample_size)  # （100，4，64，64）
        if etas is None or (type(etas) in [int, float] and etas == 0):
            eta_is_zero = True
            zs = None
        else:
            eta_is_zero = False
            if type(etas) in [int, float]: etas = [etas] * model.scheduler.num_inference_steps  # list:100
            xts = self.sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
            alpha_bar = model.scheduler.alphas_cumprod
            zs = torch.zeros(size=variance_noise_shape, device=model.device)  # (100,4,64,64)

        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}  # key:time value:index {1:99,11:98}
        idx_to_t = {k: int(v) for k, v in enumerate(timesteps)}
        xt = x0
        op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)  # 100step
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for t in op:  # t从1开始
                idx = t_to_idx[int(t)]
                # 1. predict noise residual
                if not eta_is_zero:
                    xt = xts[idx][None]

                with torch.no_grad():
                    out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
                    if not prompt == "":
                        cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)

                if not prompt == "":
                    ## classifier free guidance
                    noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
                else:
                    noise_pred = out.sample

                if eta_is_zero:
                    # 2. compute more noisy image and set x_t -> x_t+1
                    xt = self.forward_step(model, noise_pred, t, xt)

                else:
                    xtm1 = xts[idx + 1][None] # [1,4,64,64]
                    # pred of x0
                    pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5

                    # direction to xt
                    prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
                    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
                        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

                    variance = get_variance(model,t)
                    pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (0.5) * noise_pred

                    mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                    # z？？？
                    z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
                    zs[idx] = z

                    # correction to avoid error accumulation
                    xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z  # 对xts进行修正？？？
                    xts[idx + 1] = xtm1
                    if self.save_latents and t.cpu() in self.timesteps_to_save:
                        if idx+ 1 < len(idx_to_t):
                            torch.save(xtm1, os.path.join(save_path, f'noisy_latents_{idx_to_t[idx+1]}.pt'))
                        latents_path = os.path.join(save_path, f'noisy_latents_{t}.pt')
                        # if not os.path.isfile(latents_path):
                        torch.save(xtm1, latents_path)
                        pth = os.path.join(save_path, f'noisy_ddpm_{t}.pt')
                        # if not os.path.isfile(pth):
                        # torch.save(z, pth)
                        if idx != len(timesteps)-1:
                            torch.save(z, pth)
                        else: # last step
                            torch.save(torch.zeros_like(z), pth)
                        print(f"[INFO] inverted latent saved to: {pth}")

        torch.save(xts[idx], os.path.join(save_path, f'noisy_latents_{t.item()}.pt'))
        if not zs is None:
            zs[-1] = torch.zeros_like(zs[-1]) # the last timestep is zero
        # pth = os.path.join(save_path, f'noisy_ddpm_{frame_id}.pt')
        # torch.save(zs, pth)
        # print(f"[INFO] inverted latent saved to: {pth}")
        return xt, zs, xts
    def inversion_reverse_process_batch(self,
                                  xT, # [B,C,H,W]
                                  etas=0, # 1
                                  prompts="", # [self.prompt]
                                  cfg_scales=None,
                                  prog_bar=False,
                                  zs=None, # [B,T,C,H,W]
                                  controller=None,
                                  asyrp=False):
        model = self.pipe
        batch_size = len(xT) # torch.Size([5, 4, 64, 64])

        cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device) # cfgs必须是sequence

        # text_embeddings1 = self.encode_text(model, prompts* batch_size)
        # uncond_embedding1 = self.encode_text(model, [""] * batch_size)
        text_embeddings, uncond_embedding = self.prepare_cond(prompts[0], len(xT)) # torch.Size([5, 77, 768])

        if etas is None: etas = 0
        if type(etas) in [int, float]: etas = [etas] * self.scheduler.num_inference_steps
        assert len(etas) == self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps.to(model.device)

        xt = xT
        op = tqdm(timesteps[-zs.shape[1]:]) if prog_bar else timesteps[-zs.shape[1]:]

        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[1]:])}
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for t in op:
                # idx = self.scheduler.num_inference_steps - t_to_idx[int(t)] - (
                #             self.scheduler.num_inference_steps - zs.shape[0] + 1)
                idx = t_to_idx[int(t)] # 671->0
                x_index = torch.arange(len(xT))
                batches = x_index.split(self.batch_size, dim=0)
                noises = []
                for batch in batches:

                    ## Unconditional embedding
                    with torch.no_grad():
                        uncond_out = model.unet.forward(xt[batch], timestep=t,
                                                        encoder_hidden_states=uncond_embedding[batch])

                        ## Conditional embedding
                    if prompts:
                        with torch.no_grad():
                            cond_out = model.unet.forward(xt[batch], timestep=t,
                                                          encoder_hidden_states=text_embeddings[batch])

                    # z = zs[idx] if not zs is None else None
                    # z = z.expand(batch_size, -1, -1, -1)
                    if prompts:
                        ## classifier free guidance
                        noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
                    else:
                        noise_pred = uncond_out.sample
                    noises += [noise_pred]

                noise_pred = torch.cat(noises)
                # 2. compute less noisy image and set x_t -> x_t-1
                # xt = self.reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)
                xt = self.perform_ddpm_step(zs[:,idx], xt, t, noise_pred, etas[idx])
                # if controller is not None:
                #     xt = controller.step_callback(xt)

        return xt, zs # xt:为最终latent结果
    def inversion_reverse_process(self,
                                  xT,
                                  etas=0,
                                  prompts="",
                                  cfg_scales=None,
                                  prog_bar=False,
                                  zs=None,
                                  controller=None,
                                  asyrp=False):
        model = self.pipe
        batch_size = len(prompts)

        cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device)

        # text_embeddings = self.encode_text(model, prompts)
        # uncond_embedding = self.encode_text(model, [""] * batch_size)
        text_embeddings, uncond_embedding = self.prepare_cond(prompts[0], len(xT))

        if etas is None: etas = 0
        if type(etas) in [int, float]: etas = [etas] * self.scheduler.num_inference_steps
        assert len(etas) == self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps.to(model.device)

        xt = xT.expand(batch_size, -1, -1, -1)
        op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]

        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for t in op:
                # idx = self.scheduler.num_inference_steps - t_to_idx[int(t)] - (
                #             self.scheduler.num_inference_steps - zs.shape[0] + 1)
                idx = t_to_idx[int(t)]
                ## Unconditional embedding
                with torch.no_grad():
                    uncond_out = model.unet.forward(xt, timestep=t,
                                                    encoder_hidden_states=uncond_embedding)

                    ## Conditional embedding
                if prompts:
                    with torch.no_grad():
                        cond_out = model.unet.forward(xt, timestep=t,
                                                      encoder_hidden_states=text_embeddings)

                # z = zs[idx] if not zs is None else None
                # z = z.expand(batch_size, -1, -1, -1)
                if prompts:
                    ## classifier free guidance
                    noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
                else:
                    noise_pred = uncond_out.sample
                # 2. compute less noisy image and set x_t -> x_t-1
                # xt = self.reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)
                xt = self.perform_ddpm_step(zs[idx], xt, t, noise_pred, etas[idx])
                # if controller is not None:
                #     xt = controller.step_callback(xt)

        return xt, zs # xt:为最终latent结果

    def perform_ddpm_step(self, z, latents, t, noise_pred, eta):
        '''
        论文：DDIM
        '''
        # idx = t_to_idx[int(t)] # time:671 idx:0
        # z = zs[idx] if not zs is None else None
        # 1. get previous step value (=t-1)
        prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5) # (10,4,64,64)
        # 5. compute variance: "sigma_t(η)" -> see formula (16) DDIM
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = get_variance(self.pipe,t)
        std_dev_t = eta * variance ** (0.5)
        # Take care of asymetric reverse process (asyrp)
        model_output_direction = noise_pred
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        # 8. Add noice if eta > 0
        if eta > 0:
            if z is None:
                z = torch.randn(noise_pred.shape, device=self.device)
            sigma_z = eta * variance ** (0.5) * z
            prev_sample = prev_sample + sigma_z
        return prev_sample
    @torch.no_grad()
    def ddim_sample_batch(self, x, conds):
        print("[INFO] reconstructing frames...")

        zs_len = self.num_inference_steps - self.skip_steps # 68
        timesteps = self.scheduler.timesteps
        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs_len:])}  # key:t ,value:idx {1:67,11:66}
        timesteps = timesteps[-zs_len:]

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for i, t in enumerate(tqdm(timesteps)):
                noises = []
                x_index = torch.arange(len(x))
                batches = x_index.split(self.batch_size, dim = 0)
                for batch in batches:
                    noise = self.pred_noise(
                        x[batch], conds[batch], t, batch_idx=batch)
                    noises += [noise]
                noises = torch.cat(noises)
                x = self.pred_next_x(x, noises, t, i, inversion=False)
        return x



    @torch.no_grad()
    def pred_noise(self, x, cond, t, batch_idx=None,flag = 1):
        '''
        0: image
        1:video
        '''
        # For sd-depth model
        if self.use_depth and flag:
            depth = self.depths
            if batch_idx is not None:
                depth = depth[batch_idx]
            x = torch.cat([x, depth.to(x)], dim=1)

        kwargs = dict()
        # Compute controlnet outputs
        if self.control != "none" and flag:
            if batch_idx is None:
                controlnet_cond = self.controlnet_images
            else:
                controlnet_cond = self.controlnet_images[batch_idx]
            controlnet_kwargs = get_controlnet_kwargs(self.controlnet, x, cond, t, controlnet_cond, self.controlnet_scale)
            kwargs.update(controlnet_kwargs)
 
        eps = self.unet(x, t, encoder_hidden_states=cond, **kwargs).sample
        return eps

    @torch.no_grad()
    def pred_next_x(self, x, eps, t, i, inversion=False):
        if inversion:
            timesteps = reversed(self.scheduler.timesteps)
        else:
            timesteps = self.scheduler.timesteps
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        if inversion:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )
        else:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else self.scheduler.final_alpha_cumprod
            )
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        if inversion:
            pred_x0 = (x - sigma_prev * eps) / mu_prev
            x = mu * pred_x0 + sigma * eps
        else:
            pred_x0 = (x - sigma * eps) / mu
            x = mu_prev * pred_x0 + sigma_prev * eps

        return x

    @torch.no_grad()
    def prepare_cond(self, prompts, n_frames):
        if isinstance(prompts, str):
            # prompts = [prompts] * n_frames
            cond = self.encode_text(self.pipe,prompts)
            conds = torch.cat([cond] * n_frames)
            uncond = self.encode_text(self.pipe,"")
            unconds = torch.cat([uncond] * n_frames)
        elif isinstance(prompts, list):
            cond_ls = []
            for prompt in prompts:
                cond = self.get_text_embeds(prompt)
                cond_ls += [cond]
            conds = torch.cat(cond_ls)
            uncond = self.encode_text(self.pipe,"")
            unconds = torch.cat([uncond] * n_frames)
        return conds, unconds


    
    def check_latent_exists(self, save_path,type="style"):
        save_timesteps = [self.scheduler.timesteps[0]]
        if self.save_latents: # 保留中间结果
            save_timesteps += self.timesteps_to_save
        for ts in save_timesteps:
            latent_path = os.path.join(save_path, f'noisy_latents_{ts}.pt')
            noisy_path = os.path.join(save_path, f'noisy_ddpm_{ts}.pt')
            if (not os.path.exists(latent_path)) or (not os.path.exists(noisy_path)):
                print(f"[INFO] {type} latent or noise not found, please check the path: {latent_path} or {noisy_path}")
                return False
        # ddpm inversion,每帧需要额外保存部分噪声
        # for fs in self.n_frames:
        #     noisy_path = os.path.join(save_path, f'noisy_ddpm_{fs}.pt')
        #     if not os.path.exists(noisy_path):
        #         return False
        return True

    def set_latents(self, latents_app: torch.Tensor, latents_struct: torch.Tensor):
        self.latents_app = latents_app
        self.latents_struct = latents_struct

    def set_noise(self, zs_app: torch.Tensor, zs_struct: torch.Tensor):
        self.zs_app = zs_app
        self.zs_struct = zs_struct
    def load_single_image_init(self):
        print("Loading existing latents...")
        self.cross_image_config.app_latent_save_path = Path("/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/output/animal/app=4sketch_style1---struct=000000/latents/4sketch_style1.pt")
        self.cross_image_config.struct_latent_save_path = Path("/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/output/animal/app=4sketch_style1---struct=000000/latents/000000.pt")
        latents_app, latents_struct = load_latents(self.cross_image_config.app_latent_save_path, self.cross_image_config.struct_latent_save_path)
        noise_app, noise_struct = load_noise(self.cross_image_config.app_latent_save_path, self.cross_image_config.struct_latent_save_path)
        self.set_latents(latents_app, latents_struct)
        self.set_noise(noise_app, noise_struct)
        # init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
        if self.latents_struct.dim() == 4 and self.latents_app.dim() == 4 and self.latents_app.shape[0] > 1:
            self.latents_struct = self.latents_struct[self.cross_image_config.skip_steps] # torch.equal(self.content_init[0],self.latents_struct)
            self.latents_app = self.latents_app[self.cross_image_config.skip_steps] # torch.equal(self.latents_app.unsqueeze(0),self.style_init)
        self.init_latents = torch.stack(
            [self.latents_struct, self.latents_app, self.latents_struct])  # torch.Size([3, 4, 64, 64])
        self.init_zs = [self.zs_struct[self.cross_image_config.skip_steps:].unsqueeze(0),
                        self.zs_app[self.cross_image_config.skip_steps:].unsqueeze(0),
                        self.zs_struct[self.cross_image_config.skip_steps:].unsqueeze(0)]  # list:3,torch.Size([1,68, 4, 64, 64])

    @torch.no_grad()
    def __call__(self, data_path, save_path):
        '''
        '''
        # self.scheduler.set_timesteps(self.steps)
        data_path = str(data_path)
        save_path = get_latents_dir(save_path, self.model_key)
        os.makedirs(save_path, exist_ok = True)
        image_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        if os.path.isfile(data_path) and (data_path.split('.')[-1] in image_extensions):
            frames = load_video(self.config.app_image_path, self.frame_height, self.frame_width,device=self.device)  # tensor:(64,3,52,52)
            save_path = os.path.join(self.config.app_image_save_path,os.path.basename(self.config.app_image_path).split('.')[0])
            save_path = get_latents_dir(save_path, self.model_key)
            if self.check_latent_exists(save_path) and not self.force: # _981
                print(f"[INFO] inverted latents exist at: {save_path}. Skip inversion! Set 'inversion.force: True' to invert again.")
                # return
            os.makedirs(save_path, exist_ok=True)
            # conds, prompts = self.prepare_cond(self.prompt, 1)
            latents = self.encode_imgs(frames)
            torch.cuda.empty_cache()
            print(f"[INFO] clean latents shape: {latents.shape}")

            print("load image",self.config.app_image_path,"save_path",save_path)
            wt, zs, wts = self.inversion_forward_process(x0=latents,save_path= save_path,frame_id=0,etas=1,prog_bar=True,prompt=self.prompt,cfg_scale=3.5) # 单帧frame_id = 0
            # if self.recon:
            #     latent_reconstruction, _ = self.inversion_reverse_process(xT=wts[self.skip_steps], etas=1,prompts=[self.prompt] ,cfg_scales=[3.5], prog_bar=True,
            #                                       zs=zs[self.skip_steps:])
            #     torch.cuda.empty_cache()
            #     recon_frames = self.decode_latents(latent_reconstruction)
            #     recon_save_path = os.path.join(save_path, 'recon_frames')
            #     save_frames(recon_frames, recon_save_path)

            self.content_init, self.content_noise = self.load_latent(self.config.app_image_save_path, choice="style")
            latent_reconstruction, _ = self.inversion_reverse_process(xT= self.content_init, etas=1,
                                                                      prompts=[self.prompt], cfg_scales=[3.5],
                                                                      prog_bar=True,
                                                                      zs=self.content_noise)
            torch.cuda.empty_cache()
            recon_frames = self.decode_latents(latent_reconstruction)
            recon_save_path = os.path.join(save_path, 'recon_frames')
            save_frames(recon_frames, recon_save_path)
        else:
            recon_frames_list = []
            # 先进行video加载
            frames = load_video(data_path, self.frame_height, self.frame_width,device=self.device)  # tensor:(64,3,52,52)
            frame_ids = list(range(len(frames)))
            if self.n_frames is not None:
                frame_ids = frame_ids[:self.n_frames]
            frames = frames[frame_ids]
            self.n_frames = len(frames)
            # 根据video数量判断是否数据处理完
            if self.check_latent_exists(save_path) and not self.force:
                print(f"[INFO] inverted latents exist at: {save_path}. Skip inversion! Set 'inversion.force: True' to invert again.")
                return

            latents = self.encode_imgs_batch(frames)
            torch.cuda.empty_cache()
            print(f"[INFO] clean latents shape: {latents.shape}")
            wt, zs, wts = self.inversion_forward_process_batch(x0=latents, save_path=save_path, etas=1,
                                                         prog_bar=True, prompt=self.prompt,
                                                         cfg_scale=3.5)  # 单帧frame_id = 0
            # # # wt:[10,4,64,64],wts: [10,101,4,64,64] torch.equal(wts[:,self.skip_steps],self.content_init)
            # # # zs:[10,100,4,64,64)
            if self.recon:
                latent_reconstruction, _ = self.inversion_reverse_process_batch(xT=wts[:,self.skip_steps], etas=1,
                                                                          prompts=[self.prompt], cfg_scales=[3.5],
                                                                          prog_bar=True,
                                                                          zs=zs[:,self.skip_steps:])
                torch.cuda.empty_cache()
                recon_frames = self.decode_latents_batch(latent_reconstruction)
                recon_save_path = os.path.join(save_path, 'recon_frames_batch')
                save_frames(recon_frames, recon_save_path)
            # torch.equal(zs[:,self.skip_steps:],self.content_noise) 最后一个frame不一样
            ## load saved pt file
            # self.content_init,self.content_noise = self.load_latent(self.config.inversion.save_path,choice="content") # (5,4,64,64) (5,68,4,64,64)
            # latent_reconstruction, _ = self.inversion_reverse_process_batch(xT=self.content_init, etas=1,
            #                                                                 prompts=[self.prompt], cfg_scales=[3.5], # 3.5
            #                                                                 prog_bar=True,
            #                                                                 zs=self.content_noise)
            # torch.cuda.empty_cache()
            # recon_frames = self.decode_latents_batch(latent_reconstruction)
            # recon_save_path = os.path.join(save_path, 'recon_frames_batch_load_cfg1.0')
            # save_frames(recon_frames, recon_save_path)

            ## cross-image-attention single frame load
            # self.load_single_image_init()
            # latent_reconstruction, _ = self.inversion_reverse_process_batch(xT=self.latents_struct.unsqueeze(0), etas=1,
            #                                                                 prompts=[self.prompt], cfg_scales=[3.5],
            #                                                                 prog_bar=True,
            #                                                                 zs=self.zs_struct[self.skip_steps:].unsqueeze(0))
            # torch.cuda.empty_cache()
            # recon_frames = self.decode_latents_batch(latent_reconstruction)
            # recon_save_path = os.path.join(save_path, 'recon_frames_cross_image_1st_frame_cfg3.5')
            # save_frames(recon_frames, recon_save_path)
            # zs_batch = []
            # wts_batch = []
            # latent_reconstruction_list = []
            # for idx,fs in enumerate(frame_ids):
            #     latents = self.encode_imgs(frames[idx].unsqueeze(0))
            #     torch.cuda.empty_cache()
            #     print(f"[INFO] clean latents shape: {latents.shape}")
            #     wt, zs, wts = self.inversion_forward_process(x0=latents,save_path= save_path, frame_id=fs, etas=1, prog_bar=True,prompt=self.prompt, cfg_scale=3.5)  # 单帧frame_id = 0
            #     zs_batch.append(zs)
            #     wts_batch.append(wts)
            #     if self.recon:
            #         latent_reconstruction, _ = self.inversion_reverse_process(xT=wts[self.skip_steps], etas=1,
            #                                                                   prompts=[self.prompt], cfg_scales=[3.5],
            #                                                                   prog_bar=True,
            #                                                                   zs=zs[self.skip_steps:])
            #
            #         latent_reconstruction_list.append(latent_reconstruction)
            #         torch.cuda.empty_cache()
            #         recon_frames = self.decode_latents(latent_reconstruction)
            #         recon_frames_list.append(recon_frames)
            #
            #
            # zs_all = torch.stack(zs_batch,dim=0)
            # wts_all = torch.stack(wts_batch,dim=0)
            # latent_reconstruction_all = torch.stack(latent_reconstruction_list,dim=0) # [5,1,4,6,4]
            # debug_dir = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug/single_variables"
            # torch.save(zs_all,os.path.join(debug_dir,'zs.pth'))
            # torch.save(wts_all,os.path.join(debug_dir,'wts.pth'))
            # torch.save(latent_reconstruction_all, os.path.join(debug_dir, 'latents_recon.pth'))
            #
            # if recon_frames_list:
            #     recon_save_path = os.path.join(save_path, 'recon_frames_single')
            #     save_frames(torch.cat(recon_frames_list,dim=0), recon_save_path)


if __name__ == "__main__":
    config,cross_image_config = load_config()
    pipe,model_key = get_stable_diffusion_model(sd_version=config.sd_version) # ???

    # pipe, scheduler, model_key = init_model(
    #     config.device, config.sd_version, config.model_key, config.inversion.control, config.float_precision)
    # config.model_key = model_key

    start_time = time.time()
    seed_everything(config.seed)
    inversion = Inverter(pipe, pipe.scheduler, config,cross_image_config)
    inversion(config.input_path, config.inversion.save_path)
    # style_dir = "/media/allenyljiang/5234E69834E67DFB/Dataset/Sketch_dataset/ref2sketch_yr/ref"
    # for stylename in os.listdir(style_dir):
    #     inversion.config.app_image_path = os.path.join(style_dir,stylename)
    #     inversion(inversion.config.app_image_path, config.app_image_save_path)
    end_time = time.time()
    print("total cost time is",end_time - start_time)
'''
python invert.py --config configs/tea-pour-debug.yaml
--config configs/default.yaml
recon_frames = self.decode_latents_batch(xts[:,-1,])
recon_save_path="/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug"
save_frames(recon_frames, recon_save_path)
'''


