from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.schedulers import KarrasDiffusionSchedulers
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from config import Range
from models.unet_2d_condition import FreeUUNet2DConditionModel


class CrossImageAttentionStableDiffusionVideoPipeline(StableDiffusionPipeline):
    """ A modification of the standard StableDiffusionPipeline to incorporate our cross-image attention."""

    def __init__(self, vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: FreeUUNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 requires_safety_checker: bool = True):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker
        )
    @torch.no_grad()
    def __call__(
            self,
            chunk_size: List[int], # add
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            swap_guidance_scale: float = 1.0,
            cross_image_attention_range: Range = Range(10, 90),
            # DDPM addition
            zs: Optional[List[torch.Tensor]] = None,
            perform_cross_frame: bool = True,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt) # 3
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        ) # (3,77,768)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        start_index = zs[0].shape[1]
        timesteps = self.scheduler.timesteps # (68,)
        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-start_index:])} # key:t ,value:idx {1:67,11:66}
        timesteps = timesteps[-start_index:] # 671~1

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        ) # (3,4,64,64)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        op = tqdm(timesteps[-start_index:]) # 68
        n_timesteps = len(timesteps[-start_index:]) # 68

        count = 0 # 统计到目前为止的step数
        for t in op:
            i = t_to_idx[int(t)] # time:671 i:1

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred_swap = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs={'perform_swap': True,'perform_cross_frame': perform_cross_frame},
                return_dict=False,
            )[0]
            noise_pred_no_swap = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs={'perform_swap': False,'perform_cross_frame': perform_cross_frame},
                return_dict=False,
            )[0]
            # torch.equal(noise_pred_swap,noise_pred_no_swap)
            # perform guidance
            if do_classifier_free_guidance:
                _, noise_swap_pred_text = noise_pred_swap.chunk(2)
                noise_no_swap_pred_uncond, _ = noise_pred_no_swap.chunk(2)
                noise_pred = noise_no_swap_pred_uncond + guidance_scale * (
                        noise_swap_pred_text - noise_no_swap_pred_uncond)
            else: # 范围为cross_attn_32_range和cross_attn_64_range 并集
                is_cross_image_step = cross_image_attention_range.start <= i <= cross_image_attention_range.end # (10,90)
                if swap_guidance_scale > 1.0 and is_cross_image_step:
                    swapping_strengths = np.linspace(swap_guidance_scale,
                                                     max(swap_guidance_scale / 2, 1.0),
                                                     n_timesteps)
                    swapping_strength = swapping_strengths[count]
                    noise_pred = noise_pred_no_swap + swapping_strength * (noise_pred_swap - noise_pred_no_swap)
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_swap, guidance_rescale=guidance_rescale)
                else:
                    noise_pred = noise_pred_swap
            # b,c,h,w = latents.shape
            # latents = latents.reshape(chunk_size,-1,c,h,w) # (4,3,4,64,64)
            # noise_pred = noise_pred.reshape(chunk_size,-1,c,h,w)
            # # latents = torch.stack([
            # #     self.perform_ddpm_step(t_to_idx, zs[latent_idx], latents[latent_idx], t, noise_pred[latent_idx], eta)
            # #     for latent_idx in range(latents.shape[0])
            # # ]) # perform_ddpm_step 需要对三个latents使用不同的zs，因此 循环处理,单次循环输出[4,64,64]
            # latents = torch.cat([
            #     self.perform_ddpm_step(t_to_idx, zs[latent_idx], latents[:,latent_idx], t, noise_pred[:,latent_idx], eta)
            #     for latent_idx in range(latents.shape[1])
            # ],dim=0) # perform_ddpm_step 需要对三个latents使用不同的zs，因此 循环处理,单次循环输出[4,64,64]
            # latents = latents.reshape(b,c,h,w)
            # noise_pred = noise_pred.reshape(b,c,h,w)

            b, c, h, w = latents.shape
            # latents = latents.reshape(-1,chunk_size, c, h, w)  # (3,4,4,64,64)
            latents_list  = torch.split(latents, [chunk_size]*3, dim=0) # content1, style, content2 保持了原来的顺序和数据内容
            noise_pred_list = torch.split(noise_pred, [chunk_size]*3, dim=0)
            latents = torch.cat([
                self.perform_ddpm_step(t_to_idx, zs[latent_idx], latents_list[latent_idx], t, noise_pred_list[latent_idx], eta)
                for latent_idx in range(len(noise_pred_list))
            ],dim=0) # perform_ddpm_step 需要对三个latents使用不同的zs，因此 循环处理,单次循环输出[4,64,64]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                # progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

            count += 1

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0] # (3,3,512,512)
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize) # list:3 pil

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def perform_ddpm_step(self, t_to_idx, zs, latents, t, noise_pred, eta):
        '''
        论文：DDIM
        zs:(b,timesteps,c,h,w) 需要在时间维度上进行索引
        '''
        idx = t_to_idx[int(t)]
        z = zs[:,idx] if not zs is None else None
        # 1. get previous step value (=t-1)
        prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = self.get_variance(t)
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

    def get_variance(self, timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
