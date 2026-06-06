"""
DDIM Inversion / Reverse 核心算法
被 AppearanceTransferModel 通过多继承 Mixin 方式使用
"""
import os
import torch
from tqdm import tqdm
from cross_image_utils.ddpm_inversion import get_variance


class DDIMInversionMixin:
    """Mixin: 提供 DDIM 正反向反转、VAE 编解码、文本编码等方法
    要求宿主类提供:
        self.pipe, self.scheduler, self.vae, self.device, self.dtype,
        self.batch_size, self.config, self.timesteps
    """

    # ------------------------------------------------------------------
    #  DDPM step
    # ------------------------------------------------------------------
    def perform_ddpm_step(self, z, latents, t, noise_pred, eta):
        prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_t = self.scheduler.alphas_cumprod[t]
        alpha_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
        beta_t = 1 - alpha_t
        pred_x0 = (latents - beta_t ** 0.5 * noise_pred) / alpha_t ** 0.5
        var = get_variance(self.pipe, t)
        direction = (1 - alpha_prev - eta * var) ** 0.5 * noise_pred
        prev_sample = alpha_prev ** 0.5 * pred_x0 + direction
        if eta > 0:
            if z is None:
                z = torch.randn(noise_pred.shape, device=self.device)
            prev_sample = prev_sample + eta * var ** 0.5 * z
        return prev_sample

    # ------------------------------------------------------------------
    #  VAE encode / decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            return (imgs / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def decode_latents_batch(self, latents):
        imgs = []
        for latent in latents.split(self.batch_size, dim=0):
            imgs += [self.decode_latents(latent)]
        return torch.cat(imgs)

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            imgs = 2 * imgs - 1
            return self.vae.encode(imgs).latent_dist.mean * 0.18215

    @torch.no_grad()
    def encode_imgs_batch(self, imgs):
        latents = []
        for img in imgs.split(self.batch_size, dim=0):
            latents += [self.encode_imgs(img)]
        return torch.cat(latents)

    # ------------------------------------------------------------------
    #  Text encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prepare_cond(self, prompts, n_frames):
        if isinstance(prompts, str):
            cond = self._encode_text(self.pipe, prompts)
            uncond = self._encode_text(self.pipe, "")
            return torch.cat([cond] * n_frames), torch.cat([uncond] * n_frames)
        conds = torch.cat([self.get_text_embeds(p) for p in prompts])
        uncond = self._encode_text(self.pipe, "")
        return torch.cat([uncond] * n_frames), conds

    @torch.no_grad()
    def _encode_text(self, model, prompt):
        tok = model.tokenizer(prompt, padding="max_length",
                              max_length=model.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
        return model.text_encoder(tok.input_ids.to(model.device))[0]

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt=None, device="cuda"):
        tok = self.tokenizer(prompt, padding='max_length',
                             max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors='pt')
        emb = self.text_encoder(tok.input_ids.to(device))[0]
        if negative_prompt is not None:
            un_tok = self.tokenizer(negative_prompt, padding='max_length',
                                    max_length=self.tokenizer.model_max_length, return_tensors='pt')
            emb = torch.cat([self.text_encoder(un_tok.input_ids.to(device))[0], emb])
        return emb

    # ------------------------------------------------------------------
    #  DDIM Forward (Inversion)
    # ------------------------------------------------------------------
    def sample_xts_from_x0_batch(self, model, x0, num_inference_steps=50):
        B = len(x0)
        alpha_bar = model.scheduler.alphas_cumprod
        sqrt_1m = (1 - alpha_bar) ** 0.5
        shape = (B, num_inference_steps, model.unet.in_channels,
                 model.unet.sample_size, model.unet.sample_size)
        timesteps = model.scheduler.timesteps.to(model.device)
        t2idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(shape, device=x0.device)
        for t in reversed(timesteps):
            xts[:, t2idx[int(t)]] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_1m[t]
        return torch.cat([xts, x0.unsqueeze(1)], dim=1)

    def forward_step(self, model_output, timestep, sample):
        next_t = min(self.scheduler.config.num_train_timesteps - 2,
                     timestep + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)
        alpha_t = self.scheduler.alphas_cumprod[timestep]
        beta_t = 1 - alpha_t
        pred_x0 = (sample - beta_t ** 0.5 * model_output) / alpha_t ** 0.5
        return self.scheduler.add_noise(pred_x0, model_output, torch.LongTensor([next_t]))

    def inversion_forward_process_batch(self, x0, save_path, etas=None, prog_bar=False,
                                         prompt="", cfg_scale=3.5):
        B = len(x0)
        model = self.pipe
        steps = self.config.inversion.steps
        text_emb, uncond_emb = self.prepare_cond(prompt, B)
        timesteps = model.scheduler.timesteps.to(model.device)
        shape = (B, steps, model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)

        eta_zero = (etas is None) or (isinstance(etas, (int, float)) and etas == 0)
        if eta_zero:
            zs = None
            xts = None
        else:
            etas = [etas] * model.scheduler.num_inference_steps if isinstance(etas, (int, float)) else etas
            xts = self.sample_xts_from_x0_batch(model, x0, num_inference_steps=steps)
            alpha_bar = model.scheduler.alphas_cumprod
            zs = torch.zeros(size=shape, device=model.device)

        t2idx = {int(v): k for k, v in enumerate(timesteps)}
        idx2t = {k: int(v) for k, v in enumerate(timesteps)}
        xt = x0
        iterator = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for t in iterator:
                idx = t2idx[int(t)]
                if not eta_zero:
                    xt = xts[:, idx]

                noises = []
                for batch in torch.arange(len(xt)).split(self.batch_size):
                    with torch.no_grad():
                        out = model.unet.forward(xt[batch], timestep=t,
                                                  encoder_hidden_states=uncond_emb[batch])
                        if prompt:
                            cond = model.unet.forward(xt[batch], timestep=t,
                                                       encoder_hidden_states=text_emb[batch])
                        npred = out.sample + cfg_scale * (cond.sample - out.sample) if prompt else out.sample
                        noises += [npred]
                npred = torch.cat(noises)

                if eta_zero:
                    xt = self.forward_step(npred, t, xt)
                else:
                    xtm1 = xts[:, idx + 1]
                    pred_x0 = (xt - (1 - alpha_bar[t]) ** 0.5 * npred) / alpha_bar[t] ** 0.5
                    prev_t = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
                    alpha_prev = model.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else model.scheduler.final_alpha_cumprod
                    var = get_variance(model, t)
                    direction = (1 - alpha_prev - etas[idx] * var) ** 0.5 * npred
                    mu_xt = alpha_prev ** 0.5 * pred_x0 + direction
                    z = (xtm1 - mu_xt) / (etas[idx] * var ** 0.5)
                    zs[:, idx] = z
                    xtm1 = mu_xt + (etas[idx] * var ** 0.5) * z
                    xts[:, idx + 1] = xtm1

                    if idx + 1 < len(idx2t) and idx2t[idx + 1] == self.timesteps[0].item():
                        torch.save(xtm1, f'{save_path}/noisy_latents_{idx2t[idx + 1]}.pt')
                    pth = f'{save_path}/noisy_ddpm_{t.item()}.pt'
                    if os.path.exists(pth) and not self.config.inversion.force:
                        continue
                    torch.save(z if idx != len(timesteps) - 1 else torch.zeros_like(z), pth)

        if zs is not None:
            zs[:, -1] = torch.zeros_like(zs[:, -1])
        return xt, zs, xts

    # ------------------------------------------------------------------
    #  DDIM Reverse (Reconstruction)
    # ------------------------------------------------------------------
    def inversion_reverse_process_batch(self, xT, etas=0, prompts="", cfg_scales=None,
                                         prog_bar=False, zs=None):
        model = self.pipe
        cfg_t = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device)
        text_emb, uncond_emb = self.prepare_cond(prompts[0], len(xT))
        etas = [etas] * self.scheduler.num_inference_steps if isinstance(etas, (int, float)) else etas
        timesteps = self.scheduler.timesteps.to(model.device)
        xt = xT
        rev_ts = timesteps[-zs.shape[1]:]
        iterator = tqdm(rev_ts) if prog_bar else rev_ts
        t2idx = {int(v): k for k, v in enumerate(rev_ts)}

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            for t in iterator:
                idx = t2idx[int(t)]
                noises = []
                for batch in torch.arange(len(xt)).split(self.batch_size):
                    with torch.no_grad():
                        u_out = model.unet.forward(xt[batch], timestep=t,
                                                    encoder_hidden_states=uncond_emb[batch])
                    if prompts:
                        with torch.no_grad():
                            c_out = model.unet.forward(xt[batch], timestep=t,
                                                        encoder_hidden_states=text_emb[batch])
                        npred = u_out.sample + cfg_t * (c_out.sample - u_out.sample)
                    else:
                        npred = u_out.sample
                    noises += [npred]
                xt = self.perform_ddpm_step(zs[:, idx], xt, t, torch.cat(noises), etas[idx])
        return xt, zs