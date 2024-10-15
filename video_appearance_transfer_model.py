import time
from typing import List, Optional, Callable

from gradio.processing_utils import extract_base64_data

from utils import load_video, prepare_depth, save_frames, control_preprocess
import torch
import torch.nn.functional as F
from config import RunConfig, Range
from config import RunConfig
from constants import OUT_INDEX, STRUCT_INDEX, STYLE_INDEX
from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from cross_image_utils import attention_utils
from cross_image_utils.adain import masked_adain, adain, masked_adain_batch,adain_batch
from cross_image_utils.model_utils import get_stable_diffusion_model
from cross_image_utils.segmentation_batch import Segmentor
from cross_image_utils.ddpm_inversion import AttentionStore
from cross_image_utils.attention_visualization import show_cross_attention,show_self_attention_comp,visualize_and_save_features_pca
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import os
from transformers import logging
from utils import CONTROLNET_DICT
from utils import load_config, save_config
from utils import get_controlnet_kwargs, get_frame_ids, get_latents_dir, init_model, seed_everything
from utils import prepare_control, load_latent, load_video, prepare_depth, save_video
from utils import register_time, register_attention_control, register_conv_control
import vidtome
from einops import rearrange
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
from cross_image_utils.latent_utils import load_latents, load_noise,invert_images,invert_videos_and_image
from pathlib import Path
from cross_image_utils import image_utils
from PIL import Image
class AppearanceTransferModel:

    def __init__(self, video_config,config: RunConfig, pipe = None):
        ### injected layers ### 只保存自注意力
        self.down_layers = []
        self.middle_layers = []
        self.up_layers = []
        
        self.config = config
        self.pipe,self.model_key = get_stable_diffusion_model() if pipe is None else pipe
        self.pipe.scheduler.set_timesteps(self.config.num_timesteps)

        self.controller = AttentionStore() # add controller for visualization
        self.chunk_size = video_config.generation.chunk_size
        self.register_attention_control()
        self.segmentor = Segmentor(prompt=config.prompt, object_nouns=[config.object_noun],chunk_size=self.chunk_size)
        self.latents_app, self.latents_struct = None, None
        self.zs_app, self.zs_struct = None, None
        self.image_app_mask_32, self.image_app_mask_64 = None, None
        self.image_struct_mask_32, self.image_struct_mask_64 = None, None
        self.enable_edit = False
        self.perform_cross_frame = False
        self.step = 0 # get_adain_callback的时候修改时间步
        ## video process ##
        self.device = video_config.device
        self.video_config = video_config
        gene_config = video_config.generation
        float_precision = gene_config.float_precision if "float_precision" in gene_config else video_config.float_precision
        if float_precision == "fp16":
            self.dtype = torch.float16
            print("[INFO] float precision fp16. Use torch.float16.")
        else:
            self.dtype = torch.float32
            print("[INFO] float precision fp32. Use torch.float32.")
        self.batch_size = 2
        self.frame_height, self.frame_width = video_config.height, video_config.width
        self.skip_steps = config.skip_steps # 32
        self.timesteps = self.pipe.scheduler.timesteps[self.skip_steps:]
        self.t_to_idx = {int(v): k for k, v in enumerate(self.timesteps)}  # key:t ,value:idx {1:67,11:66}
        self.model_key = self.video_config.model_key
        # data path for inversion
        self.n_frames = self.video_config.inversion.n_frames
        self.prompt = self.video_config.inversion.prompt
        self.struct_data_path = self.video_config.input_path
        self.style_data_path = self.config.app_image_path
        self.struct_save_path = get_latents_dir(self.video_config.inversion.save_path,self.model_key)
        self.style_save_path = get_latents_dir(os.path.join(self.video_config.app_image_save_path,
                     os.path.basename(self.config.app_image_path).split('.')[0]),self.model_key)
        # generation_save_path
        self.save_path = self.video_config.generation.output_path
    def check_latent_exists(self, save_path):
        save_timesteps = self.pipe.scheduler.timesteps
        for ts in save_timesteps:
            latent_path = os.path.join(save_path, f'noisy_latents_{ts}.pt')
            noisy_path = os.path.join(save_path, f'noisy_ddpm_{ts}.pt')
            if (not os.path.exists(latent_path)) or (not os.path.exists(noisy_path)):
                print(f"[INFO] latent or noise not found, please check the path: {latent_path} or {noisy_path}")
                return False
        return True
    def load_latents_or_invert_videos(self):
        # [10,4,64,64] [10,68,4,64,64]
        if self.config.load_latents and self.check_latent_exists(self.struct_save_path) and self.check_latent_exists(self.style_save_path):
            print("Loading existing latents...")
            self.prepare_data(self.struct_data_path)
            print("Done.")
        else:
            print("Inverting images...")
            app_image, struct_image = image_utils.load_video_images(struct_dir=self.struct_data_path,app_image_path=self.style_data_path) # struct_image:list
            self.enable_edit = False  # Deactivate the cross-image attention layers
            self.style_init, self.style_noises, self.content_init, self.content_noise = invert_videos_and_image(
                                                                                 sd_model=self.pipe,
                                                                                 app_image=app_image,
                                                                                 struct_image_list=struct_image,
                                                                                 prompt=self.prompt,
                                                                                 style_save_path=self.style_save_path,
                                                                                 struct_save_path=self.struct_save_path,
                                                                                 cfg=self.config,
                                                                                 video_cfg = self.video_config)
            self.enable_edit = True
            print("Done.")

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
            style_noises = torch.cat(noises, dim=0).unsqueeze(0)
            return style_latents,style_noises # [1,4,64,64] [1,68,4,64,64]
        else: # 671
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

    def prepare_data(self,content_video_path):
        '''

        frame_ids: content video num
        '''
        self.frames = load_video(content_video_path, self.frame_height,self.frame_width, frame_ids=self.frame_ids, device=self.device)
        self.style_init,self.style_noises = self.load_latent(self.video_config.app_image_save_path,choice="style")
        self.content_init,self.content_noise = self.load_latent(self.video_config.inversion.save_path,choice="content")



    def load_single_image_init(self):
        print("Loading existing latents...")
        self.config.app_latent_save_path = Path("/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/output/animal/app=4sketch_style1---struct=000000/latents/4sketch_style1.pt")
        self.config.struct_latent_save_path = Path("/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/output/animal/app=4sketch_style1---struct=000000/latents/000000.pt")
        latents_app, latents_struct = load_latents(self.config.app_latent_save_path, self.config.struct_latent_save_path)
        noise_app, noise_struct = load_noise(self.config.app_latent_save_path, self.config.struct_latent_save_path)
        self.set_latents(latents_app, latents_struct)
        self.set_noise(noise_app, noise_struct)
        # init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
        if self.latents_struct.dim() == 4 and self.latents_app.dim() == 4 and self.latents_app.shape[0] > 1:
            self.latents_struct = self.latents_struct[self.config.skip_steps] # torch.equal(self.content_init[0],self.latents_struct)
            self.latents_app = self.latents_app[self.config.skip_steps] # torch.equal(self.latents_app.unsqueeze(0),self.style_init)
        self.init_latents = torch.stack(
            [self.latents_struct, self.latents_app, self.latents_struct])  # torch.Size([3, 4, 64, 64])
        self.init_zs = [self.zs_struct[self.config.skip_steps:].unsqueeze(0),
                        self.zs_app[self.config.skip_steps:].unsqueeze(0),
                        self.zs_struct[self.config.skip_steps:].unsqueeze(0)]  # list:3,torch.Size([1,68, 4, 64, 64])
        chunk_size = 2
        # .repeat(chunk_size, 1, 1, 1, 1)
        self.init_latents = torch.cat(
            [self.latents_struct.repeat(chunk_size, 1, 1, 1), self.latents_app.repeat(chunk_size, 1, 1, 1), self.latents_struct.repeat(chunk_size, 1, 1, 1)],dim=0)  # torch.Size([3, 4, 64, 64])
        self.init_zs = [self.zs_struct[self.config.skip_steps:].unsqueeze(0).repeat(chunk_size, 1, 1, 1, 1),
                        self.zs_app[self.config.skip_steps:].unsqueeze(0).repeat(chunk_size, 1, 1, 1, 1),
                        self.zs_struct[self.config.skip_steps:].unsqueeze(0).repeat(chunk_size, 1, 1, 1, 1)]  # list:3,torch.Size([1,68, 4, 64, 64])
        # self.style_init, self.style_noises, self.content_init, self.content_noise
        self.chunk_size = chunk_size
        start_step = min(self.config.cross_attn_32_range.start, self.config.cross_attn_64_range.start)  # 10
        end_step = max(self.config.cross_attn_32_range.end, self.config.cross_attn_64_range.end)  # 90
        images = self.pipe(
            chunk_size = chunk_size,
            prompt=[self.video_config.inversion.prompt] * 3 *chunk_size,
            latents=self.init_latents,
            guidance_scale=1.0,
            num_inference_steps=self.config.num_timesteps,
            swap_guidance_scale=self.config.swap_guidance_scale,
            callback=self.get_adain_callback(),
            eta=1,
            generator=torch.Generator('cuda').manual_seed(self.config.seed),
            cross_image_attention_range=Range(start=start_step, end=end_step),
            zs=self.init_zs,
        ).images  # 注意这里guidance_scale = 1

        joined_images = np.concatenate(images[::-1], axis=1)
        Image.fromarray(joined_images).save(os.path.join(self.video_config.work_dir, f"out_joined_single_{chunk_size}_copy.png"))

    def inference_chunk(self,frame_ids,chunk_index):
        chunk_size = len(frame_ids) # chunk_size会发生改变
        frame_ids = torch.tensor(frame_ids)
        # self.pipe.scheduler.set_timesteps(self.config.num_timesteps)
        start_step = min(self.config.cross_attn_32_range.start, self.config.cross_attn_64_range.start)  # 10
        end_step = max(self.config.cross_attn_32_range.end, self.config.cross_attn_64_range.end)  # 90
        init_latents = torch.cat([self.content_init[frame_ids], self.style_init.repeat((chunk_size,1,1,1)), self.content_init[frame_ids]],dim=0) # (12,4,64,64)
        init_zs = [self.content_noise[frame_ids], self.style_noises.repeat(chunk_size, 1, 1, 1, 1), self.content_noise[frame_ids]] # list,[4,68,4,64,64]
        # self.chunk_size = chunk_size
        images = self.pipe(
            chunk_size = chunk_size,
            prompt=[self.video_config.inversion.prompt] * 3 *chunk_size, # 'a tea pot pouring tea into a cup.'
            latents=init_latents,
            guidance_scale=1.0,
            num_inference_steps=self.config.num_timesteps, # 100
            swap_guidance_scale=self.config.swap_guidance_scale, # 1.0
            callback=self.get_adain_callback(),
            eta=1,
            generator=torch.Generator('cuda').manual_seed(self.config.seed),
            cross_image_attention_range=Range(start=start_step, end=end_step),
            zs=init_zs,
            # perform_cross_frame=self.video_config.perform_cross_frame,
        ).images  # 注意这里guidance_scale = 1
        # joined_images = np.concatenate(images[::-1], axis=1)
        # Image.fromarray(joined_images).save(os.path.join(self.video_config.work_dir, f"chunk_inference_{chunk_size}_test.png"))

        return images

    def __call__(self, video_data_path, video_latent_path, output_path,frame_ids):
        self.frame_ids = frame_ids
        self.load_latents_or_invert_videos()
        # self.prepare_data(video_data_path)
        # frames = len(self.frames)
        chunk_ids = np.arange(0,self.n_frames,self.chunk_size-1)
        result_stylized = []
        result_style = []
        result_content = []
        frames_counter = 0
        # self.enable_edit = False
        # self.perform_cross_frame = False
        # # self.load_single_image_init() # debug
        # frame_ids=[0]
        # self.inference_chunk(frame_ids, 0)
        self.enable_edit = True
        self.perform_cross_frame = True
        # self.enable_edit = False  # Activate our cross-image attention layers
        post = f'{self.chunk_size}'+ ('_cross_frame' if self.video_config.perform_cross_frame else '_wo_cross_frame')
        save_path = os.path.join(output_path, f'generated_result_{post}')
        save_video = os.path.join(output_path, f'recon_result_{post}')
        style_path = os.path.join(output_path, f'style_result_{post}')
        intermediate_path = os.path.join(output_path, f'intermediate_result_{post}')
        os.makedirs(save_path,exist_ok=True)
        os.makedirs(save_video,exist_ok=True)
        os.makedirs(style_path,exist_ok=True)
        os.makedirs(intermediate_path,exist_ok=True)
        for i in range(len(chunk_ids)):
            ch_start = chunk_ids[i]
            ch_end = self.n_frames if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
            prefix = [0]
            # prefix = []
            frame_ids =  prefix + list(range(ch_start, ch_end))  # 每个chunk会加上第一帧
            total = self.inference_chunk(frame_ids =frame_ids,chunk_index = i)
            cur_chunk_size = len(frame_ids)
            cur_res_generated = total[:cur_chunk_size]
            cur_res_style = total[cur_chunk_size:2*cur_chunk_size]
            cur_res_content = total[2*cur_chunk_size:]
            pre_len = len(prefix)
            result_stylized.extend(cur_res_generated[pre_len:])
            result_style.extend(cur_res_style[pre_len:])
            result_content.extend(cur_res_content[pre_len:])
            frames_counter += len(chunk_ids) - 1
            from PIL import Image
            joined_images = np.concatenate(total[::-1], axis=1)
            Image.fromarray(joined_images).save(os.path.join(intermediate_path, f"out_joined_{i}.png"))

        torch.cuda.empty_cache()

        save_frames(result_stylized, save_path)
        joined_images = np.concatenate(result_stylized[::-1], axis=1)
        Image.fromarray(joined_images).save(os.path.join(save_path, f"combined.png"))
        save_frames(result_content, save_video)
        save_frames(result_style,style_path)


    def set_latents(self, latents_app: torch.Tensor, latents_struct: torch.Tensor):
        self.latents_app = latents_app
        self.latents_struct = latents_struct

    def set_noise(self, zs_app: torch.Tensor, zs_struct: torch.Tensor):
        self.zs_app = zs_app
        self.zs_struct = zs_struct

    def set_masks(self, masks: List[torch.Tensor]):
        self.image_app_mask_32, self.image_struct_mask_32, self.image_app_mask_64, self.image_struct_mask_64 = masks

    def get_adain_callback(self):

        def callback(st: int, timestep: int, latents: torch.FloatTensor) -> Callable:
            self.step = st
            self.t = timestep
            self.segmentor.chunk_size = latents.shape[0] // 3 # chunk_size发生变化
            # Compute the masks using prompt mixing self-segmentation and use the masks for AdaIN operation
            if self.config.use_masked_adain and self.step == self.config.adain_range.start:
                masks = self.segmentor.get_object_masks()
                self.set_masks(masks)
            # Apply AdaIN operation using the computed masks
            if self.config.adain_range.start <= self.step < self.config.adain_range.end:
                if self.config.use_masked_adain:
                    # latents[0] = masked_adain(latents[0], latents[1], self.image_struct_mask_64, self.image_app_mask_64)
                    #print(latents[:self.segmentor.chunk_size].shape, latents[self.segmentor.chunk_size:2*self.segmentor.chunk_size].shape,self.image_struct_mask_64.shape, self.image_app_mask_64.shape)
                    latents[:self.segmentor.chunk_size] = masked_adain_batch(latents[:self.segmentor.chunk_size], latents[self.segmentor.chunk_size:2*self.segmentor.chunk_size], self.image_struct_mask_64, self.image_app_mask_64)
                else:
                    latents[:self.segmentor.chunk_size] = adain_batch(latents[:self.segmentor.chunk_size], latents[self.segmentor.chunk_size:2*self.segmentor.chunk_size])
                    # latents[0] = adain(latents[0], latents[1])

        return callback

        
    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0
    def register_attention_control(self):

        model_self = self # self表示AppearanceTransferModel

        class AttentionProcessor:

            def __init__(self, place_in_unet: str,query_preserve=False,layer_name = ""):
                # self.chunk_size = chunk_size
                self.place_in_unet = place_in_unet
                self.query_preserve = query_preserve
                self.layer_name = layer_name
                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError("AttnProcessor2_0 requires torch 2.0, to use it, please upgrade torch to 2.0.")

            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         perform_swap: bool = False,
                         perform_cross_frame: bool = True,):
                if hidden_states.shape[0] < 3: # 单张图情况
                    chunk_size = 1
                else:
                    chunk_size = hidden_states.shape[0] // 3  # 单张图[1,4096,320]
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                is_cross = encoder_hidden_states is not None
                if not is_cross:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                should_mix = False

                # Potentially apply our cross image attention operation
                # To do so, we need to be in a self-attention layer in the decoder part of the denoising network
                
                vis_flag = False

                # print(f"perform_swap: {perform_swap}, is_cross: {is_cross}")
                # print("Key shape before rearrange:", key.shape)
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit: # model_self.enable_edit：True
                    if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                        should_mix = True
                        key = rearrange(key, "(b f) d c -> b f d c", f=chunk_size)
                        value = rearrange(value, "(b f) d c -> b f d c", f=chunk_size)
                        query = rearrange(query, "(b f) d c -> b f d c", f=chunk_size)
                        if model_self.step % 5 == 0 and model_self.step < 40:
                            # Inject the structure's keys and values
                            key[OUT_INDEX] = key[STRUCT_INDEX].clone() # key长度为3
                            value[OUT_INDEX] = value[STRUCT_INDEX].clone()
                        else:
                            # Inject the appearance's keys and values
                            key[OUT_INDEX] = key[STYLE_INDEX].clone()
                            value[OUT_INDEX] = value[STYLE_INDEX].clone()
                        # add query_preserve
                        if self.query_preserve:
                            vis_flag = True
                            query[OUT_INDEX] = query[STRUCT_INDEX]*model_self.config.gamma + query[OUT_INDEX]*(1-model_self.config.gamma)
                            query[OUT_INDEX] = query[OUT_INDEX]*model_self.config.temperature
                        key = rearrange(key, "b f d c -> (b f) d c")
                        value = rearrange(value, "b f d c -> (b f) d c")
                        query = rearrange(query, "b f d c -> (b f) d c")
                else: 
                    # 进行cross_frame_attention
                    if model_self.perform_cross_frame:
                        former_frame_index = [0]*chunk_size
                        # print(f"former_frame_index: {former_frame_index}, key[OUT_INDEX].shape: {key[OUT_INDEX].shape}")
                        key = rearrange(key, "(b f) d c -> b f d c", f=chunk_size)  # [12,4096,320] -> [3,4,4096,320]
                        key[OUT_INDEX] = key[OUT_INDEX][former_frame_index]  # torch.Size([4, 4096, 320])
                        key = rearrange(key, "b f d c -> (b f) d c")
                        value = rearrange(value, "(b f) d c -> b f d c", f=chunk_size)
                        value[OUT_INDEX] = value[OUT_INDEX][former_frame_index]
                        value = rearrange(value, "b f d c -> (b f) d c")


                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # query.contiguous()
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                # ## visualize
                # save_dir = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs_debug/attentions"
                # visualize_and_save_features_pca(query[OUT_INDEX], int(model_self.step), save_dir, self.place_in_unet, suffix="q_cs")

                # Compute the cross attention and apply our contrasting operation
                hidden_states, attn_weight = attention_utils.compute_scaled_dot_product_attention(
                    query, key, value,
                    edit_map=perform_swap and model_self.enable_edit and should_mix,
                    is_cross=is_cross,
                    contrast_strength=model_self.config.contrast_strength,
                )

                # if model_self.controller is None:
                #     model_self.controller = DummyController()
                # attn_weight = model_self.controller(attn_weight) # Todo: ?? controller放置位置？ attn_weight:(3,8,1024,1024) 写的有点问题
                # TypeError: __call__() missing 2 required positional arguments: 'is_cross' and 'place_in_unet'
                # Update attention map for segmentation
                if model_self.config.use_masked_adain and model_self.step == model_self.config.adain_range.start - 1: # model_self.config.adain_range.start：20
                    model_self.segmentor.update_attention(attn_weight, is_cross)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[OUT_INDEX].dtype)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor


                return hidden_states

        def register_recr(net_,name, count, place_in_unet):
            '''
            在这里实现指定层添加自定义的AttentionProcessor
            '''
            if net_.__class__.__name__ == 'ResnetBlock2D':
                pass
            if net_.__class__.__name__ == 'Attention':
                if name.endswith("attn1"):
                    if place_in_unet == "down": 
                        model_self.down_layers.append(name)
                    elif place_in_unet == "mid":
                        model_self.middle_layers.append(name)
                    elif place_in_unet == "up":
                        model_self.up_layers.append(name)
                if len(model_self.up_layers) >= 4:
                    query_preserve = True
                    net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}",query_preserve))
                else:
                    net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}"))
                return count + 1
            elif hasattr(net_, 'children'):
                for child_name,net__ in net_.named_children():
                    new_full_name = f"{name}.{child_name}" if name else child_name
                    count = register_recr(net__, new_full_name,count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.pipe.unet.named_children()
        for net_name,net in sub_nets:
            if "down" in net_name:
                cross_att_count += register_recr(net,net_name, 0, "down")
            elif "up" in net_name:
                cross_att_count += register_recr(net,net_name, 0, "up")
            elif "mid" in net_name:
                cross_att_count += register_recr(net,net_name, 0, "mid")

if __name__ == "__main__":

    start_time = time.time()
    config,cross_image_config = load_config()
    # pipe, scheduler, model_key = init_model(
    #     config.device, config.sd_version, config.model_key, config.generation.control, config.float_precision)

    # pipe, model_key = get_stable_diffusion_model()
    # scheduler = pipe.scheduler


    config.model_key = "/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stable-diffusion-v1-5"
    seed_everything(config.seed)
    generator = AppearanceTransferModel(config,cross_image_config)
    frame_ids = get_frame_ids(
        config.generation.frame_range, config.generation.frame_ids)
    generator(config.input_path, config.generation.latents_path,
              config.generation.output_path, frame_ids=frame_ids)
    end_time = time.time()
    print("total cost time",end_time - start_time)