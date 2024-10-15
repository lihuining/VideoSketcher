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
from cross_image_utils.adain import masked_adain, adain, masked_adain_batch, adain_batch
from cross_image_utils.model_utils import get_stable_diffusion_model
from cross_image_utils.segmentation_batch import Segmentor
from cross_image_utils.ddpm_inversion import AttentionStore
from cross_image_utils.attention_visualization import show_cross_attention, show_self_attention_comp, \
    visualize_and_save_features_pca
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
from cross_image_utils.latent_utils import load_latents, load_noise, invert_images, invert_videos_and_image
from pathlib import Path
from cross_image_utils import image_utils


class TestModel:

    def __init__(self, video_config, config: RunConfig, pipe=None):
        ### injected layers ### 只保存自注意力
        self.down_layers = []
        self.middle_layers = []
        self.up_layers = []

        self.config = config
        self.controller = AttentionStore()  # add controller for visualization
        self.chunk_size = video_config.generation.chunk_size
        self.segmentor = Segmentor(prompt=config.prompt, object_nouns=[config.object_noun], chunk_size=self.chunk_size)
        self.latents_app, self.latents_struct = None, None
        self.zs_app, self.zs_struct = None, None
        self.image_app_mask_32, self.image_app_mask_64 = None, None
        self.image_struct_mask_32, self.image_struct_mask_64 = None, None
        self.enable_edit = False
        self.step = 0  # get_adain_callback的时候修改时间步
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
        self.skip_steps = config.skip_steps  # 32
        self.timesteps = self.pipe.scheduler.timesteps[self.skip_steps:]
        self.t_to_idx = {int(v): k for k, v in enumerate(self.timesteps)}  # key:t ,value:idx {1:67,11:66}
        self.model_key = self.video_config.model_key
        # data path for inversion
        self.n_frames = self.video_config.inversion.n_frames
        self.prompt = self.video_config.inversion.prompt
        self.struct_data_path = self.video_config.input_path
        self.style_data_path = self.config.app_image_path
        self.struct_save_path = get_latents_dir(self.video_config.inversion.save_path, self.model_key)
        self.style_save_path = get_latents_dir(os.path.join(self.video_config.app_image_save_path,
                                                            os.path.basename(self.config.app_image_path).split('.')[0]),
                                               self.model_key)
        # generation_save_path
        self.save_path = self.video_config.generation.output_path

    def load_single_image_init(self):
        print("Loading existing latents...")
        self.config.app_latent_save_path = Path(
            "/output/animal/app=4sketch_style1---struct=000000/latents/4sketch_style1.pt")
        self.config.struct_latent_save_path = Path(
            "/output/animal/app=4sketch_style1---struct=000000/latents/000000.pt")
        self.latents_app, self.latents_struct = load_latents(self.config.app_latent_save_path, self.config.struct_latent_save_path)
        self.noise_app, self.noise_struct = load_noise(self.config.app_latent_save_path, self.config.struct_latent_save_path)

        print(torch.equal(self.latents_app,self.style_init[0]))
        # self.set_latents(latents_app, latents_struct)
        # self.set_noise(noise_app, noise_struct)
        # init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
        # if self.latents_struct.dim() == 4 and self.latents_app.dim() == 4 and self.latents_app.shape[0] > 1:
        #     self.latents_struct = self.latents_struct[self.config.skip_steps]
        #     self.latents_app = self.latents_app[self.config.skip_steps]
        # self.init_latents = torch.stack(
        #     [self.latents_struct, self.latents_app, self.latents_struct])  # torch.Size([3, 4, 64, 64])
        # self.init_zs = [self.zs_struct[self.config.skip_steps:].unsqueeze(0),
        #                 self.zs_app[self.config.skip_steps:].unsqueeze(0),
        #                 self.zs_struct[self.config.skip_steps:].unsqueeze(0)]  # list:3,torch.Size([1,68, 4, 64, 64])
    def prepare_data(self):
        '''

        frame_ids: content video num
        '''
        self.style_init, self.style_noises = self.load_latent(self.video_config.app_image_save_path, choice="style")
        self.content_init, self.content_noise = self.load_latent(self.video_config.inversion.save_path,
                                                                 choice="content")
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

if __name__ == "__main__":

    start_time = time.time()
    config,cross_image_config = load_config()
    generator = TestModel(config,cross_image_config)
    generator.prepare_data()
    generator.load_single_image_init()