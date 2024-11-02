import os.path
import time
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.schedulers import KarrasDiffusionSchedulers
# from requests.packages import target
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from config import Range
from constants import OUT_INDEX
from models.unet_2d_condition import FreeUUNet2DConditionModel
from cross_image_utils.gluestick.get_matching import get_sparse_matching_results
from cross_image_utils.sketch_loss_utils import compute_sketch_matching_loss
from utils import save_frames


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value
def spherical_dist_loss(x, y):
    return -x@y.T

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
        print("before:VAE requires_grad:", any(param.requires_grad for param in self.vae.parameters()))
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        print("after:VAE requires_grad:", any(param.requires_grad for param in self.vae.parameters()))

        # set_requires_grad(self.clip_model, False)

    @torch.enable_grad()
    def cond_fn(self,
                noise_pred_list,
                latents,
                chunk_size,
                t, # 当前时间步
                per_step_weight,
                pred_x0,
                clip_model,
                struct_gt,
                chunk_index,
                matching_save_dir,
                prev_latents_x0_list,
                config,
                ):
        loss = torch.tensor(0.0).to(dtype=latents.dtype, device=latents.device)
        if config.latent_update:
            if all((isinstance(value, (list, torch.Tensor)) and len(value) != 0) or (
                    isinstance(value, torch.Tensor) and value.numel() != 0) for value in
                   prev_latents_x0_list.values()):
                pre_chunk_last = prev_latents_x0_list[t.item()].unsqueeze(0)  # [1,4,64,64]
                cur_chunk = pred_x0[:chunk_size][1:, ]
                target_chunk = torch.cat([pre_chunk_last, cur_chunk], dim=0)

                with torch.enable_grad():
                    # 初始化 loss 为 0
                    # loss = torch.tensor(0.0).to(pred_x0.device)
                    dummy_pred_chunk = pred_x0[:chunk_size].clone().detach()
                    dummy_pred_chunk = dummy_pred_chunk.requires_grad_(requires_grad=True)
                    loss = per_step_weight * torch.nn.functional.mse_loss(target_chunk, dummy_pred_chunk)
                    # for frame_id in range(chunk_size):
                    #     # dummy_pred = pred_x0[:chunk_size][frame_id].clone().detach()
                    #     # dummy_pred = dummy_pred.requires_grad_(requires_grad=True)
                    #     dummy_pred = dummy_pred_chunk[frame_id]
                    #     target = target_chunk[frame_id]
                    #     loss += per_step_weight * torch.nn.functional.mse_loss(target, dummy_pred)
                    loss.backward()  # 保留计算图
                    latents[:chunk_size] = latents[
                                           :chunk_size] + dummy_pred_chunk.grad.clone() * -1.  # max:3.46 0.06 norm:91

        ## with update
        if config.update_with_clip or config.update_with_matching:
            stylized_latents = latents[:chunk_size].detach().requires_grad_(True)
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            # with torch.no_grad():
            stylized_image = self.vae.decode(stylized_latents / self.vae.config.scaling_factor, return_dict=False)[
                0]  # (3,3,512,512) tensor[0,1]
            stylized_image_tensor = (stylized_image / 2 + 0.5).clamp(0, 1)  # torch.Size([2, 3, 1024, 1024]), [0, 1]
        if config.update_with_clip and t.item() >= config.update_with_clip_start_time and t.item() <= config.update_with_clip_end_time:
            # with torch.no_grad():
            _, content_stylized_image, style_stylized_image = clip_model(stylized_image_tensor)
            _, struct_content, struct_style = clip_model(struct_gt)
            for i in range(chunk_size):
                loss += config.update_with_clip_guidance * spherical_dist_loss(content_stylized_image[i],
                                                                               struct_content[i])
        if config.update_with_matching and t.item() >= config.update_with_matching_start_time and t.item() <= config.update_with_matching_end_time: # and chunk_index != 0

            # from PIL import Image
            # vis_image = stylized_image.detach().cpu().permute(0, 2, 3, 1).numpy() # (2, 1024, 1024, 3)
            # vis_image = (vis_image * 255).round().astype("uint8")
            # vis_image = Image.fromarray(vis_image[0])
            # vis_image.save("image.jpg")
            # stylized_image_pil = self.image_processor.postprocess(stylized_image.detach(), output_type=output_type,
            #                                          do_denormalize=[True] * stylized_image.shape[0])  # list:3 pil,detach之后才能转化为numpy
            for i in range(chunk_size - 1):
                sparse_matching_lines, sparse_matching_points = get_sparse_matching_results(stylized_image_tensor[i],
                                                                                            stylized_image_tensor[
                                                                                                i + 1],
                                                                                            timestep=t.item(),
                                                                                            save_dir=matching_save_dir,
                                                                                            chunk_index=chunk_index)
                loss += compute_sketch_matching_loss(stylized_image_tensor[i], stylized_image_tensor[i + 1],
                                                     sparse_matching_lines, sparse_matching_points)[0]
            # loss = loss.to(dtype=latents.dtype)

            # loss = torch.tensor(loss, dtype=latents.dtype).to(device=latents.device)
            # loss = loss
            print("Requires grad (stylized_latents):", stylized_latents.requires_grad)
            print("Loss requires grad:", loss.requires_grad)
            if loss.item() == 0:
                print("Loss is 0, skipping gradient computation.")
            else:
                grads = -torch.autograd.grad(loss * config.update_with_matching_guidance, stylized_latents)[0]
                print("gradient max", torch.max(grads))
                noise_pred_list[OUT_INDEX] = noise_pred_list[OUT_INDEX] - torch.sqrt(beta_prod_t) * grads
        return noise_pred_list
    def encode_text(self, prompts,device):
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = self.text_encoder(text_input.input_ids.to(device))[0]
        return text_encoding
    @torch.no_grad()
    def __call__(
            self,
            chunk_index: int = 50,
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
            # perform_cross_frame: bool = True,
            prev_latents_x0_list = {},
            config = None,
            clip_model = None,
            struct_gt = None,
            # latent_update = False,
            # update_with_matching = False,
            # update_with_matching_guidance: float = 50.0,
            # update_with_matching_start_time: Optional[int] = 1,
            # update_with_matching_end_time: Optional[int] = 1,
            matching_save_dir: Optional[str] = "",
            free_u_args = [],
            enable_edit = False,
    ):
        # config
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

        prompt_embeds = self.encode_text(prompt*len(latents),device)
        uncond_prompt_embeds = self.encode_text([""]*len(latents),device)
        # Todo:为什么这两个embedding结果不一样？？？
        # prompt_embeds = self._encode_prompt(
        #     prompt,
        #     device,
        #     num_images_per_prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     lora_scale=text_encoder_lora_scale,
        # ) # (3,77,768) [】没有cfg则为positive结果
        # uncond_prompt_embeds = self._encode_prompt(
        #     "",
        #     device,
        #     num_images_per_prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     lora_scale=text_encoder_lora_scale,
        # ) # (3,77,768) [】没有cfg则为positive结果
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
        per_step_weight = 100.0 # 每个step的权重
        for t in op: # 671 timestep10 end:1
            i = t_to_idx[int(t)] # time:671 i:1
            if i > 25:
                per_step_weight = 0
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # 输出结果一样
            # with torch.no_grad():
            if do_classifier_free_guidance:
                noise_pred_swap = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs={'perform_swap': True,'time_step':t.item()}, # 'perform_cross_frame': perform_cross_frame ,'free_u_args':free_u_args
                    return_dict=False,
                )[0] # (6,4,64,64)
            noise_pred_swap_uncond = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_prompt_embeds,
                cross_attention_kwargs={'perform_swap': True,'time_step':t.item()}, # 'perform_cross_frame': perform_cross_frame ,'free_u_args':free_u_args
                return_dict=False,
            )[0] # (6,4,64,64)
            if do_classifier_free_guidance:
                noise_pred = noise_pred_swap_uncond + guidance_scale * (noise_pred_swap - noise_pred_swap_uncond)
            else:
                noise_pred = noise_pred_swap_uncond
            #noise_pred_swap
            # if do_classifier_free_guidance:
            #     noise_swap_pred_uncond, noise_swap_pred_text = noise_pred_swap.chunk(2)
            #     noise_pred = noise_swap_pred_uncond + guidance_scale * (
            #             noise_swap_pred_text - noise_swap_pred_uncond)
            # noise_pred = rescale_noise_cfg(noise_pred, noise_swap_pred_text, guidance_rescale=1.0)
            ###
            # noise_pred_no_swap = self.unet(
            #     latent_model_input,
            #     t,
            #     encoder_hidden_states=prompt_embeds,
            #     cross_attention_kwargs={'perform_swap': False,'time_step':t.item()},
            #     return_dict=False,
            # )[0]
            # # torch.equal(noise_pred_swap,noise_pred_no_swap)
            # # perform guidance
            # if do_classifier_free_guidance:
            #     noise_swap_pred_uncond, noise_swap_pred_text = noise_pred_swap.chunk(2)
            #     noise_no_swap_pred_uncond, _ = noise_pred_no_swap.chunk(2)
            #     noise_pred = noise_no_swap_pred_uncond + guidance_scale * (
            #             noise_swap_pred_text - noise_no_swap_pred_uncond)
            # else: # 范围为cross_attn_32_range和cross_attn_64_range 并集
            #     is_cross_image_step = cross_image_attention_range.start <= i <= cross_image_attention_range.end # (10,90)
            #     if swap_guidance_scale > 1.0 and is_cross_image_step:
            #         swapping_strengths = np.linspace(swap_guidance_scale,
            #                                          max(swap_guidance_scale / 2, 1.0),
            #                                          n_timesteps)
            #         swapping_strength = swapping_strengths[count] # 每个step的swapping_strength在逐渐变大
            #         noise_pred = noise_pred_no_swap + swapping_strength * (noise_pred_swap - noise_pred_no_swap)
            #         noise_pred = rescale_noise_cfg(noise_pred, noise_pred_swap, guidance_rescale=guidance_rescale)
            #     else:
            #         noise_pred = noise_pred_swap

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
            if b < 3: # 单张图情况
                chunk_size = 1 # 单张图[1,4096,320]
            else:
                chunk_size = b // 3  
            latents_list  = list(torch.split(latents, [chunk_size]*3, dim=0)) # content1, style, content2 保持了原来的顺序和数据内容
            noise_pred_list = list(torch.split(noise_pred, [chunk_size]*3, dim=0)) # (2,4,64,64)

            #latents,pred_x0 = torch.cat([self.perform_ddpm_step(t_to_idx, zs[latent_idx], latents_list[latent_idx], t, noise_pred_list[latent_idx], eta) for latent_idx in range(len(noise_pred_list))],dim=0) # perform_ddpm_step 需要对三个latents使用不同的zs，因此 循环处理,单次循环输出[4,64,64]
            # 后处理 # latent_update
            # if not all(not value for value in prev_latents_x0_list.values()): # 所有值均不为空,RuntimeError: Boolean value of Tensor with more than one value is ambiguous

            if i > 1 and enable_edit: # i>1确保pred_x0已经计算过
                noise_pred_list = self.cond_fn(noise_pred_list,latents,chunk_size,t,per_step_weight,pred_x0,clip_model,struct_gt,chunk_index,matching_save_dir,prev_latents_x0_list,config)
            # else: # Todo: 检查为什么单独运行这一段输出的是彩色的图像？
            #     stylized_latents = latents[:chunk_size].clone().detach().requires_grad_(True)
            #     print('time step is',t.item(),'torch equal',torch.equal(stylized_latents[0],stylized_latents[1]))
            #     alpha_prod_t = self.scheduler.alphas_cumprod[t]
            #     beta_prod_t = 1 - alpha_prod_t
            #     with torch.no_grad():
            #         stylized_image = self.vae.decode(stylized_latents / self.vae.config.scaling_factor, return_dict=False)[0]  # (3,3,512,512) tensor[0,1]
            #     # loss = torch.tensor(0.0).to(dtype=latents.dtype, device=latents.device)
            #     stylized_image_tensor = (stylized_image / 2 + 0.5).clamp(0,1)  # torch.Size([2, 3, 1024, 1024]), [0, 1]
            #     # vis_image = stylized_image.detach().cpu().permute(0, 2, 3, 1).numpy() # (2, 1024, 1024, 3)
            #     # vis_image = (vis_image * 255).round().astype("uint8")
            #     # vis_image = Image.fromarray(vis_image[0])
            #     # vis_image.save(os.path.join(matching_save_dir,f"image_{t.item()}.jpg"))
            #     stylized_image_pil = self.image_processor.postprocess(stylized_image.detach(), output_type=output_type,
            #                                              do_denormalize=[True] * stylized_image.shape[0])  # list:3 pil,detach之后才能转化为numpy
            #     joined_images = np.concatenate(stylized_image_pil[::-1], axis=1)
            #     Image.fromarray(joined_images).save(os.path.join(matching_save_dir,f"image_{t.item()}_combined.jpg"))
            #     for i in range(chunk_size-1):
            #         sparse_matching_lines,sparse_matching_points = get_sparse_matching_results(stylized_image_tensor[i], stylized_image_tensor[i+1],timestep=t.item(),save_dir=matching_save_dir,chunk_index=chunk_index)
            #         loss = compute_sketch_matching_loss(stylized_image_tensor[i],stylized_image_tensor[i+1],sparse_matching_lines,sparse_matching_points)
            #     print(torch.equal(stylized_latents,latents[:chunk_size]))

            noise_pred_list = tuple(noise_pred_list)
            latents_list = tuple(latents_list)
            latents_list_next = []
            pred_x0_list = []
            for latent_idx in range(len(noise_pred_list)):
                latent,pred_x0 = self.perform_ddpm_step(t_to_idx, zs[latent_idx], latents_list[latent_idx], t,noise_pred_list[latent_idx], eta)
                latents_list_next.append(latent)
                pred_x0_list.append(pred_x0)
            latents,pred_x0 = torch.cat(latents_list_next,dim=0),torch.cat(pred_x0_list,dim=0)
            # call the callback, if provided
            if enable_edit and i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                # progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents,pred_x0)

            count += 1
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        # time.sleep(3)
        if not output_type == "latent":
            # with torch.no_grad():
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
        return prev_sample,pred_original_sample

    def get_variance(self, timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
