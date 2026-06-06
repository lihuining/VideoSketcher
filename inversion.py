import gc
import os
from omegaconf import OmegaConf
from cross_image_utils.ddpm_inversion import get_variance

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
import time
from utils import load_video, save_frames
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import load_config, seed_everything
from utils import get_frame_ids, get_latents_dir, load_latent
from cross_image_utils.model_utils import get_stable_diffusion_model
from cross_image_utils.CLIP_model import init_clip_model, get_clip_model


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class InversionReconModel:

    def __init__(self, config, pipe=None):
        self.config = config
        self.work_dir = os.path.join(self.config.work_dir, self.config.input_path.split('/')[-2])
        self.config.inversion.save_path = os.path.join(self.work_dir, "latents")
        self.frame_ids = get_frame_ids(self.config.generation.frame_range, self.config.generation.frame_ids)
        self.sd_version = self.config.sd_version
        self.pipe, self.model_key = get_stable_diffusion_model(self.sd_version, model_id=self.config.get("model_id")) if pipe is None else pipe
        self.config.model_key = self.model_key
        self.pipe.scheduler.set_timesteps(self.config.num_timesteps)

        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder

        if self.config.enable_xformers_memory_efficient_attention:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except ModuleNotFoundError:
                print("[WARNING] xformers not found. Disable xformers attention.")

        self.skip_steps = config.inversion.skip_steps
        self.timesteps = self.scheduler.timesteps[self.skip_steps:]

        self.n_frames = self.config.inversion.n_frames
        self.prompt = self.config.inversion.prompt
        self.struct_data_path = self.config.input_path
        self.struct_save_path = get_latents_dir(self.config.inversion.save_path, self.model_key)
        self.style_data_path = self.config.app_image_path
        self.style_save_path = get_latents_dir(os.path.join(self.config.app_image_save_path,
                                                             os.path.basename(self.config.app_image_path).split('.')[0]),
                                                self.model_key)

        self.save_path = os.path.join(self.work_dir, os.path.basename(self.config.app_image_path).split('.')[0])
        os.makedirs(self.save_path, exist_ok=True)

        # device
        self.device = self.config.device
        float_precision = self.config.get("float_precision", "fp32")
        if float_precision == "fp16":
            self.dtype = torch.float16
            print("[INFO] float precision fp16. Use torch.float16.")
        else:
            self.dtype = torch.float32
            print("[INFO] float precision fp32. Use torch.float32.")
        self.batch_size = self.config.inversion.get("batch_size", 4)
        self.frame_height, self.frame_width = self.config.height, self.config.width

        self.t_to_idx = {int(v): k for k, v in enumerate(self.timesteps)}
        self.idx_to_t = {k: int(v) for k, v in enumerate(self.timesteps)}

        # CLIP (used for eval, not required for recon)
        init_clip_model(
            vit_path=self.config.get("csd_clip_vit_path"),
            checkpoint_path=self.config.get("csd_clip_checkpoint_path"),
        )
        self.clip_model = get_clip_model()
        set_requires_grad(self.clip_model, False)

    # ------------------------------------------------------------------ #
    #  Utility methods
    # ------------------------------------------------------------------ #
    def check_latent_exists(self, save_path):
        for ts in self.timesteps:
            noisy_path = os.path.join(save_path, f'noisy_ddpm_{ts}.pt')
            if not os.path.exists(noisy_path):
                return False
        latent_path = os.path.join(save_path, f'noisy_latents_{self.timesteps[0].item()}.pt')
        if not os.path.exists(latent_path):
            return False
        return True

    def perform_ddpm_step(self, z, latents, t, noise_pred, eta):
        prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        variance = get_variance(self.pipe, t)
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * noise_pred
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        if eta > 0:
            if z is None:
                z = torch.randn(noise_pred.shape, device=self.device)
            sigma_z = eta * variance ** (0.5) * z
            prev_sample = prev_sample + sigma_z
        return prev_sample

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
        batch_latents = latents.split(self.batch_size, dim=0)
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
        batch_imgs = imgs.split(self.batch_size, dim=0)
        for img in batch_imgs:
            latents += [self.encode_imgs(img)]
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def prepare_cond(self, prompts, n_frames):
        if isinstance(prompts, str):
            cond = self.encode_text(prompts)
            conds = torch.cat([cond] * n_frames)
            uncond = self.encode_text("")
            unconds = torch.cat([uncond] * n_frames)
        else:
            raise ValueError("prompts must be a string")
        return conds, unconds

    def encode_text(self, prompt):
        text_input = self.pipe.tokenizer(
            prompt, padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def sample_xts_from_x0_batch(self, model, x0, num_inference_steps=50):
        batch = len(x0)
        alpha_bar = model.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        variance_noise_shape = (batch, num_inference_steps, model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)
        timesteps = model.scheduler.timesteps.to(model.device)
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(variance_noise_shape).to(x0.device)
        for t in reversed(timesteps):
            idx = t_to_idx[int(t)]
            xts[:, idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        xts = torch.cat([xts, x0.unsqueeze(1)], dim=1)
        return xts

    def forward_step(self, model_output, timestep, sample):
        next_timestep = min(self.scheduler.config.num_train_timesteps - 2,
                            timestep + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        next_sample = self.scheduler.add_noise(pred_original_sample, model_output,
                                                torch.LongTensor([next_timestep]))
        return next_sample

    # ------------------------------------------------------------------ #
    #  DDIM Inversion — Forward (image -> noisy latent + noise maps)
    # ------------------------------------------------------------------ #
    def invert_single_image(self, pil_image, save_path, prompt="", cfg_scale=3.5):
        """Invert a single PIL image. Returns (init_latent, noise_maps, wts)."""
        input_tensor = torch.from_numpy(np.array(pil_image)).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        input_tensor = input_tensor / 127.5 - 1.0  # [0,255] -> [-1,1]
        return self.invert_tensor(input_tensor, save_path, prompt, cfg_scale)

    def invert_tensor(self, x0, save_path, prompt="", cfg_scale=3.5, save_latents=True):
        """Invert a batched image tensor x0: [B, C, H, W] in [-1,1] range.
           If B == 1 (single image), the single-image code path is used.
           Returns (init_latent, noise_maps, wts).
        """
        num_inference_steps = self.config.inversion.steps
        text_embeddings, uncond_embedding = self.prepare_cond(prompt, len(x0))
        timesteps = self.scheduler.timesteps.to(self.device)
        model = self.pipe

        variance_noise_shape = (len(x0), num_inference_steps, model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)
        xts = self.sample_xts_from_x0_batch(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=self.device)

        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        idx_to_t = {k: int(v) for k, v in enumerate(timesteps)}
        xt = x0
        op = tqdm(reversed(timesteps), desc="Inversion forward")
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for t in op:
                idx = t_to_idx[int(t)]
                xt = xts[:, idx]

                batches = torch.arange(len(xt)).split(self.batch_size, dim=0)
                noises = []
                for batch in batches:
                    with torch.no_grad():
                        out = model.unet.forward(xt[batch], timestep=t, encoder_hidden_states=uncond_embedding[batch])
                        if prompt != "":
                            cond_out = model.unet.forward(xt[batch], timestep=t, encoder_hidden_states=text_embeddings[batch])
                        if prompt != "":
                            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
                        else:
                            noise_pred = out.sample
                        noises += [noise_pred]
                noise_pred = torch.cat(noises)

                xtm1 = xts[:, idx + 1]
                pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
                prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
                alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
                variance = get_variance(model, t)
                pred_sample_direction = (1 - alpha_prod_t_prev - variance) ** (0.5) * noise_pred  # eta=1
                mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                z = (xtm1 - mu_xt) / (variance ** 0.5)
                zs[:, idx] = z
                xtm1 = mu_xt + (variance ** 0.5) * z
                xts[:, idx + 1] = xtm1

                if save_latents:
                    if idx + 1 < len(idx_to_t):
                        if idx_to_t[idx + 1] == self.timesteps[0].item():
                            torch.save(xtm1, os.path.join(save_path, f'noisy_latents_{idx_to_t[idx + 1]}.pt'))
                    pth = os.path.join(save_path, f'noisy_ddpm_{t.item()}.pt')
                    if not (os.path.exists(pth) and not self.config.inversion.force):
                        if idx != len(timesteps) - 1:
                            torch.save(z, pth)
                        else:
                            torch.save(torch.zeros_like(z), pth)

        zs[:, -1] = torch.zeros_like(zs[:, -1])
        return xt, zs, xts

    # ------------------------------------------------------------------ #
    #  DDIM Reverse (noisy latent -> predicted x0)
    # ------------------------------------------------------------------ #
    def reverse(self, xT, zs, prompts, cfg_scales, etas=1, save_path=None):
        """Reverse from xT using noise maps zs. Returns reconstructed latent."""
        model = self.pipe
        text_embeddings, uncond_embedding = self.prepare_cond(prompts[0], len(xT))
        cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(self.device)
        if type(etas) in [int, float]:
            etas = [etas] * self.scheduler.num_inference_steps

        timesteps = self.scheduler.timesteps.to(self.device)
        xt = xT
        op = tqdm(timesteps[-zs.shape[1]:], desc="Reverse")
        for t in op:
            idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[1]:])}[int(t)]
            batches = torch.arange(len(xt)).split(self.batch_size, dim=0)
            noises = []
            for batch in batches:
                with torch.no_grad():
                    uncond_out = model.unet.forward(xt[batch], timestep=t, encoder_hidden_states=uncond_embedding[batch])
                if prompts:
                    with torch.no_grad():
                        cond_out = model.unet.forward(xt[batch], timestep=t, encoder_hidden_states=text_embeddings[batch])
                    noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
                else:
                    noise_pred = uncond_out.sample
                noises += [noise_pred]
            noise_pred = torch.cat(noises)
            xt = self.perform_ddpm_step(zs[:, idx], xt, t, noise_pred, etas[idx])
        return xt

    # ------------------------------------------------------------------ #
    #  High-level APIs
    # ------------------------------------------------------------------ #
    def invert_and_recon_style(self):
        """Invert and reconstruct the style reference image."""
        save_path = self.style_save_path
        os.makedirs(save_path, exist_ok=True)
        if self.check_latent_exists(save_path) and not self.config.inversion.force:
            print("[Style] Latents already exist, skipping inversion.")
            return

        style_pil = Image.open(self.style_data_path).convert("RGB").resize((self.frame_width, self.frame_height))
        x0 = torch.from_numpy(np.array(style_pil)).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        x0 = x0 / 127.5 - 1.0

        print("[Style] Inverting...")
        wt, zs, wts = self.invert_tensor(x0, save_path, prompt=self.config.inversion.prompt,
                                          cfg_scale=self.config.cfg_inversion_style)

        print("[Style] Reconstructing...")
        latent_recon = self.reverse(xT=wts[:, self.skip_steps], zs=zs[:, self.skip_steps:],
                                     prompts=[self.config.inversion.prompt],
                                     cfg_scales=[self.config.cfg_inversion_style])
        recon_img = self.decode_latents_batch(latent_recon)
        recon_save_path = os.path.join(save_path, 'recon_frames')
        os.makedirs(recon_save_path, exist_ok=True)
        save_frames(recon_img, recon_save_path, start_index=0)
        print(f"[Style] Recon saved to {recon_save_path}")

    def invert_and_recon_video(self):
        """Invert and reconstruct all video frames (batched)."""
        cur_batch_size = 10
        frames = load_video(self.struct_data_path, self.frame_height, self.frame_width, device=self.device)
        recon_save_path = os.path.join(self.config.inversion.save_path, 'recon_frames_batch')
        if self.config.inversion.n_frames:
            self.n_frames = self.config.inversion.n_frames
        else:
            self.n_frames = len(frames)
        print(f"[Video] Total frames: {self.n_frames}")

        for i in range(0, self.n_frames, cur_batch_size):
            end = min(i + cur_batch_size, self.n_frames)
            cur_batch_save = os.path.join(self.struct_save_path, f'batch_frames{i}_{end}')
            os.makedirs(cur_batch_save, exist_ok=True)
            if self.check_latent_exists(cur_batch_save) and not self.config.inversion.force:
                print(f"[Video] Batch {i}-{end} already exists, skip.")
                continue

            latents = self.encode_imgs_batch(frames[i:end])
            wt, zs, wts = self.invert_tensor(latents, cur_batch_save, prompt=self.config.inversion.prompt,
                                              cfg_scale=self.config.cfg_inversion_style)
            latent_recon = self.reverse(xT=wts[:, self.skip_steps], zs=zs[:, self.skip_steps:],
                                         prompts=[self.config.inversion.prompt],
                                         cfg_scales=[self.config.cfg_inversion_style])
            torch.cuda.empty_cache()
            recon_frames = self.decode_latents_batch(latent_recon)
            save_frames(recon_frames, recon_save_path, start_index=i)
            print(f"[Video] Batch {i}-{end} done.")

    def invert_and_recon_all(self):
        """Run inversion + reconstruction for both style image and video."""
        self.invert_and_recon_style()
        self.invert_and_recon_video()


if __name__ == "__main__":

    start_time = time.time()
    config = load_config()
    seed_everything(config.seed)

    model = InversionReconModel(config)
    model.invert_and_recon_all()
    # Or run individually:
    # model.invert_and_recon_style()
    # model.invert_and_recon_video()

    end_time = time.time()
    print("total cost time", end_time - start_time)

# python3 inversion.py --config ./configs/iccv/kid-football_video-debug.yaml
