import gc
import os
from omegaconf import OmegaConf
from cross_image_utils.ddpm_inversion import get_variance
# from tests.test_cross_attention import batch_size

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
import time
import pyrallis
from utils import load_video, prepare_depth, save_frames, control_preprocess
import torch.nn.functional as F
from config import RunConfig, Range
from config import RunConfig
from constants import OUT_INDEX, STRUCT_INDEX, STYLE_INDEX
from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from cross_image_utils import attention_utils
from cross_image_utils.adain import masked_adain, adain, masked_adain_batch,adain_batch
from cross_image_utils.model_utils import get_stable_diffusion_model
from cross_image_utils.segmentation_batch_separate import Segmentor
from cross_image_utils.ddpm_inversion import AttentionStore
from cross_image_utils.attention_visualization import show_cross_attention,show_self_attention_comp,visualize_and_save_features_pca,visualize_and_save_mean_features_pca
from cross_image_utils.attention_visualization import *
import torch
import numpy as np
from tqdm import tqdm
import os
from utils import load_config, save_config
from utils import get_controlnet_kwargs, get_frame_ids, get_latents_dir, init_model, seed_everything
from utils import prepare_control, load_latent, load_video, prepare_depth, save_video
from cross_image_utils.video_util import frame_to_video,video_to_frame
from einops import rearrange
from cross_image_utils.latent_utils import load_latents, load_noise,invert_images,invert_videos_and_image
from pathlib import Path
from cross_image_utils import image_utils
from PIL import Image
from cross_image_utils.CSD_Score.model import CSD_CLIP, convert_state_dict

from cross_image_utils.figures_visualization.attention_map_std import *
from cross_image_utils.CLIP_model import tensor_process, spherical_dist_loss, clip_model

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

class AppearanceTransferModel:

    def __init__(self,config, pipe = None):
        ### injected layers ### 只保存自注意力

        self.down_layers = [] # 12
        self.middle_layers = [] # 2
        self.up_layers = [] # 18
        
        self.config = config
        self.work_dir = os.path.join(self.config.work_dir,self.config.input_path.split('/')[-2]) # '/media/allenyljiang/8204e606-a28e-48fa-89f4-560d3c00163e/home/allenyljiang/Desktop/CVPR2025/Struct_latents/breakdance-flare'
        self.config.inversion.save_path = os.path.join(self.work_dir,"latents") # latents save path
        self.frame_ids = get_frame_ids(self.config.generation.frame_range, self.config.generation.frame_ids) # 根据frame_range生成，可以确定加载图片的间隔
        self.sd_version = self.config.sd_version
        self.pipe,self.model_key = get_stable_diffusion_model(self.sd_version) if pipe is None else pipe
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
        self.skip_steps = config.inversion.skip_steps # 32
        self.timesteps = self.scheduler.timesteps[self.skip_steps:] # forward过程中的steps
        # query preservation
        self.up_layers_start_index = self.config.up_layers_start_index # 必须在register_attention_control前
        self.register_attention_control() # 必须在AttentionStore前，因为需要保存layers
        self.controller = AttentionStore(timesteps = self.timesteps,layers = self.down_layers + self.up_layers,cond_layer = self.config.cond_layer) # add controller for visualization
        self.valid_layers = self.controller.get_valid_layers()
        self.chunk_size = self.config.generation.chunk_size

        # data path for inversion
        self.n_frames = self.config.inversion.n_frames
        self.prompt = self.config.inversion.prompt
        self.struct_data_path = self.config.input_path
        self.style_data_path = self.config.app_image_path
        self.struct_save_path = get_latents_dir(self.config.inversion.save_path,self.model_key)
        self.style_save_path = get_latents_dir(os.path.join(self.config.app_image_save_path,os.path.basename(self.config.app_image_path).split('.')[0]),self.model_key)
        if self.config.use_edge:
            struct_name = os.path.basename(self.struct_data_path)
            self.struct_save_path_edge = get_latents_dir(self.config.inversion.save_path+'_edge', self.model_key)

            self.struct_data_path_edge = self.struct_data_path.replace(struct_name, struct_name + '_edge')
            os.makedirs(self.struct_data_path_edge,exist_ok=True)
            os.makedirs(self.struct_save_path_edge,exist_ok=True)

        self.update_with_matching = self.config.update_with_matching
        # generation_save_path
        # if self.config.whether_use_ablation:
        #     if self.config.ablation_key:
        #         self.work_dir = os.path.join(self.work_dir,self.config.ablation_key)
        self.save_path = os.path.join(self.work_dir,os.path.basename(self.config.app_image_path).split('.')[0]) # 不同style分开存放
        os.makedirs(self.save_path,exist_ok=True)

        # 用于loss计算
        self.prev_latents_x0_list = {} # 保留上一个帧所有时间步预测的x0
        for t in self.timesteps:
            self.prev_latents_x0_list[int(t.item())] = []
        ## 跟新config的outputPath

        # self.segmentor = Segmentor(prompt=config.prompt, object_nouns=[config.object_noun],chunk_size=self.chunk_size)
        self.segmentor = Segmentor(config=self.config,tokenizer=self.pipe.tokenizer)
        self.latents_app, self.latents_struct = None, None
        self.zs_app, self.zs_struct = None, None
        self.image_app_mask_32, self.image_app_mask_64 = None, None
        self.image_struct_mask_32, self.image_struct_mask_64 = None, None
        self.enable_edit = False
        self.perform_cross_frame = False
        self.perform_cross_frame_with_prev = False
        self.step = 0 # get_adain_callback的时候修改时间步
        ## video process ##
        self.device = self.config.device
        gene_config = self.config.generation
        float_precision = gene_config.float_precision if "float_precision" in gene_config else self.config.float_precision
        if float_precision == "fp16":
            self.dtype = torch.float16
            print("[INFO] float precision fp16. Use torch.float16.")
        else:
            self.dtype = torch.float32
            print("[INFO] float precision fp32. Use torch.float32.")
        self.batch_size = self.config.inversion.batch_size
        self.frame_height, self.frame_width = self.config.height, self.config.width

        self.t_to_idx = {int(v): k for k, v in enumerate(self.timesteps)}  # key:t ,value:idx {1:67,11:66}
        self.idx_to_t = {k: int(v) for k, v in enumerate(self.timesteps)}

        self.model_key = self.config.model_key
        self.latent_update = self.config.latent_update
        self.start_frame = self.config.start_frame
        ## debug usage variables
        self.frame_ids_cur_chunk = None
        self.app_image, self.struct_image,self.struct_tensor_image = None,None,None # 保存输入appearance以及struct,numpy格式,self.struct_image为list
        self.chunk_index = 0 # 表示当前是第几个chunk，初始化需要使用的不一样
        ## vis
        self.key_injection_layers = set()
        ## add CLIP
        self.clip_model = CSD_CLIP("vit_large", "default",
                              model_path="Codes/cross-image-attention/cross_image_utils/CSD_Score/models/ViT-L-14.pt")
        set_requires_grad(self.clip_model, False)
        model_path = "Codes/cross-image-attention/cross_image_utils/CSD_Score/models/checkpoint.pth"
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        self.clip_model.load_state_dict(state_dict, strict=False)
        self.clip_model = self.clip_model.to(self.device)


    def check_latent_exists(self, save_path):
        # save_timesteps = self.scheduler.timesteps
        for ts in self.timesteps:
            # latent_path = os.path.join(save_path, f'noisy_latents_{ts}.pt')
            noisy_path = os.path.join(save_path, f'noisy_ddpm_{ts}.pt')
            # if (not os.path.exists(latent_path)) or (not os.path.exists(noisy_path)):
            #     print(f"[INFO] latent or noise not found, please check the path: {latent_path} or {noisy_path}")
            #     return False
            if (not os.path.exists(noisy_path)):
                print(f"[INFO] latent or noise not found, please check the path: {noisy_path}")
                return False

        latent_path = os.path.join(save_path, f'noisy_latents_{self.timesteps[0].item()}.pt')
        if not os.path.exists(latent_path):
            return False
        return True
    def load_latents_or_invert_videos(self):
        if not self.config.use_edge:
            self.app_image, self.struct_image = image_utils.load_video_images(self.style_data_path,self.struct_data_path, self.struct_data_path)  # 没有使用edge的时候没有self.struct_data_path_edge这个路径
        else:
            self.app_image, self.struct_image = image_utils.load_video_images(self.style_data_path,self.struct_data_path, self.struct_data_path_edge,edge_method=self.config.edge_method)
            # 加载完成之后再修改
            self.struct_data_path = self.struct_data_path_edge
            self.struct_save_path = self.struct_save_path_edge
        #self.app_image, self.struct_image = image_utils.load_video_images(struct_dir=self.struct_data_path,app_image_path=self.style_data_path)  # struct_image:list
        # [10,4,64,64] [10,68,4,64,64]
        if self.config.load_latents and self.check_latent_exists(self.struct_save_path) and self.check_latent_exists(self.style_save_path):
            print("Loading existing latents...")
            self.prepare_data()
            print("Done.")
        else:
            print("Inverting images...")

            self.enable_edit = False  # Deactivate the cross-image attention layers
            if self.check_latent_exists(self.struct_save_path) and not self.check_latent_exists(self.style_save_path):
                self.content_init, self.content_noise = self.load_latent(self.struct_save_path,choice="content")
                self.style_init, self.style_noises = invert_videos_and_image(sd_model=self.pipe,
                                                                             app_image=self.app_image,
                                                                             struct_image_list=self.struct_image,
                                                                             prompt=self.prompt,
                                                                             style_save_path=self.style_save_path,
                                                                             struct_save_path=self.struct_save_path,
                                                                             cfg=self.config,
                                                                             choice="style")

            elif self.check_latent_exists(self.style_save_path) and not self.check_latent_exists(self.struct_save_path):
                self.style_init, self.style_noises = self.load_latent(self.style_save_path,choice="style")
                self.content_init, self.content_noise = invert_videos_and_image(sd_model=self.pipe,
                                                                                     app_image=self.app_image,
                                                                                     struct_image_list=self.struct_image,
                                                                                     prompt=self.prompt,
                                                                                     style_save_path=self.style_save_path,
                                                                                     struct_save_path=self.struct_save_path,
                                                                                     cfg=self.config,
                                                                                     choice="content")
            else:
                self.content_init, self.content_noise = invert_videos_and_image(sd_model=self.pipe,
                                                                                app_image=self.app_image,
                                                                                struct_image_list=self.struct_image,
                                                                                prompt=self.prompt,
                                                                                style_save_path=self.style_save_path,
                                                                                struct_save_path=self.struct_save_path,
                                                                                cfg=self.config,
                                                                                choice="content")
                self.style_init, self.style_noises = invert_videos_and_image(sd_model=self.pipe,
                                                                                     app_image=self.app_image,
                                                                                     struct_image_list=self.struct_image,
                                                                                     prompt=self.prompt,
                                                                                     style_save_path=self.style_save_path,
                                                                                     struct_save_path=self.struct_save_path,
                                                                                     cfg=self.config,
                                                                                     choice="style")

            self.enable_edit = True # 控制所有操作
            print("Done.")

    def load_latent(self,save_path,choice="content"):
        '''
        save_path:直接给的是上一级目录，不需要额外的处理,不进行start_frame和end_frame的处理
        加载latents以及noise
        type = latent or ddpm
        choice = content or style
        noise需要每个时刻，
        latents只需要初始值,加载latents的时候先把全部的值加载起来再说
        '''
        if choice == "style":

            noises = []  # 用于存储每个timestep的噪声
            # save_path = os.path.join(latent_path,os.path.basename(self.config.app_image_path).split('.')[0])
            # save_path = get_latents_dir(save_path, self.model_key)
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
            # save_path = get_latents_dir(latent_path, self.model_key)
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

    def prepare_data(self):
        '''
        只进行latents加载
        frame_ids: content video num
        self.app_image, self.struct_image = image_utils.load_video_images(struct_dir=self.struct_data_path,app_image_path=self.style_data_path)  # struct_image:list
        '''
        self.style_init,self.style_noises = self.load_latent(self.style_save_path,choice="style")
        self.content_init,self.content_noise = self.load_latent(self.struct_save_path,choice="content")



    # def load_single_image_init(self):
    #     print("Loading existing latents...")
    #     self.config.app_latent_save_path = Path("Codes/cross-image-attention/output/animal/app=4sketch_style1---struct=000000/latents/4sketch_style1.pt")
    #     self.config.struct_latent_save_path = Path("Codes/cross-image-attention/output/animal/app=4sketch_style1---struct=000000/latents/000000.pt")
    #     latents_app, latents_struct = load_latents(self.config.app_latent_save_path, self.config.struct_latent_save_path)
    #     noise_app, noise_struct = load_noise(self.config.app_latent_save_path, self.config.struct_latent_save_path)
    #     self.set_latents(latents_app, latents_struct)
    #     self.set_noise(noise_app, noise_struct)
    #     # init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    #     if self.latents_struct.dim() == 4 and self.latents_app.dim() == 4 and self.latents_app.shape[0] > 1:
    #         self.latents_struct = self.latents_struct[self.config.skip_steps] # torch.equal(self.content_init[0],self.latents_struct)
    #         self.latents_app = self.latents_app[self.config.skip_steps] # torch.equal(self.latents_app.unsqueeze(0),self.style_init)
    #     self.init_latents = torch.stack(
    #         [self.latents_struct, self.latents_app, self.latents_struct])  # torch.Size([3, 4, 64, 64])
    #     self.init_zs = [self.zs_struct[self.config.skip_steps:].unsqueeze(0),
    #                     self.zs_app[self.config.skip_steps:].unsqueeze(0),
    #                     self.zs_struct[self.config.skip_steps:].unsqueeze(0)]  # list:3,torch.Size([1,68, 4, 64, 64])
    #     chunk_size = 2
    #     # .repeat(chunk_size, 1, 1, 1, 1)
    #     self.init_latents = torch.cat(
    #         [self.latents_struct.repeat(chunk_size, 1, 1, 1), self.latents_app.repeat(chunk_size, 1, 1, 1), self.latents_struct.repeat(chunk_size, 1, 1, 1)],dim=0)  # torch.Size([3, 4, 64, 64])
    #     self.init_zs = [self.zs_struct[self.config.skip_steps:].unsqueeze(0).repeat(chunk_size, 1, 1, 1, 1),
    #                     self.zs_app[self.config.skip_steps:].unsqueeze(0).repeat(chunk_size, 1, 1, 1, 1),
    #                     self.zs_struct[self.config.skip_steps:].unsqueeze(0).repeat(chunk_size, 1, 1, 1, 1)]  # list:3,torch.Size([1,68, 4, 64, 64])
    #     # self.style_init, self.style_noises, self.content_init, self.content_noise
    #     self.chunk_size = chunk_size
    #     start_step = min(self.config.cross_attn_32_range[0], self.config.cross_attn_64_range[0])  # 10
    #     end_step = max(self.config.cross_attn_32_range[1], self.config.cross_attn_64_range[1])  # 90
    #     images = self.pipe(
    #         # chunk_size = chunk_size,
    #         prompt=[self.config.inversion.prompt] * 3 *chunk_size,
    #         latents=self.init_latents,
    #         guidance_scale=self.config.cfg_inversion_style,  # 1.0 --> 0.0
    #         num_inference_steps=self.config.num_timesteps,
    #         swap_guidance_scale=self.config.swap_guidance_scale,
    #         callback=self.get_adain_callback(),
    #         eta=1,
    #         generator=torch.Generator('cuda').manual_seed(self.config.seed),
    #         cross_image_attention_range=Range(start=start_step, end=end_step),
    #         zs=self.init_zs,
    #         prev_latents_x0_list=self.prev_latents_x0_list,
    #         latent_update = self.latent_update,
    #     ).images  # 注意这里guidance_scale = 1
    #
    #     joined_images = np.concatenate(images[::-1], axis=1)
    #     Image.fromarray(joined_images).save(os.path.join(self.work_dir, f"out_joined_single_{chunk_size}_copy.png"))
    #
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
    def sample_xts_from_x0_batch(self,model, x0, num_inference_steps=50):
        """
        Samples from P(x_1:T|x_0) 和原始顺序不同
        """
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
    def forward_step(self, model_output, timestep, sample):
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
    def inversion_forward_process_batch(self, x0,save_path,
                                  etas=None,
                                  prog_bar=False,
                                  prompt="",
                                  cfg_scale=3.5,
                                  num_inference_steps=50, eps=None
                                  ):
        cur_batch = len(x0)
        model = self.pipe
        num_inference_steps = self.config.inversion.steps
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
                    xt = self.forward_step(noise_pred, t, xt)

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
                    # if t.cpu() in self.timesteps:
                    if idx+ 1 < len(idx_to_t):
                        #print(idx_to_t[idx + 1], self.timesteps[0].item(),idx_to_t[idx + 1]==self.timesteps[0].item())
                        if idx_to_t[idx+1] == self.timesteps[0].item():
                            torch.save(xtm1, os.path.join(save_path, f'noisy_latents_{idx_to_t[idx+1]}.pt'))
                    pth = os.path.join(save_path, f'noisy_ddpm_{t.item()}.pt')
                    if os.path.exists(pth) and not self.config.inversion.force:
                        continue
                    if idx != len(timesteps)-1:
                        torch.save(z, pth)
                    else:
                        torch.save(torch.zeros_like(z), pth)
                    print(f"[INFO] inverted latent saved to: {pth}")

        # torch.save(xts[:,idx], os.path.join(save_path, f'noisy_latents_{t.item()}.pt'))
        if not zs is None:
            zs[:,-1] = torch.zeros_like(zs[:,-1])

        return xt, zs, xts
    def inversion_reverse_process_batch(self,
                                  xT, # [B,C,H,W]
                                  etas=0,
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
    def inference_chunk(self,frame_ids,chunk_index):
        self.chunk_index = chunk_index
        chunk_size = len(frame_ids) # chunk_size会发生改变
        self.frame_ids_cur_chunk = torch.tensor(frame_ids)
        # self.scheduler.set_timesteps(self.config.num_timesteps)
        start_step = min(self.config.cross_attn_32_range[0], self.config.cross_attn_64_range[0])  # 10
        end_step = max(self.config.cross_attn_32_range[1], self.config.cross_attn_64_range[1])  # 90
        init_latents = torch.cat([self.content_init[self.frame_ids_cur_chunk], self.style_init.repeat((chunk_size,1,1,1)), self.content_init[self.frame_ids_cur_chunk]],dim=0) # (12,4,64,64)
        init_zs = [self.content_noise[self.frame_ids_cur_chunk], self.style_noises.repeat(chunk_size, 1, 1, 1, 1), self.content_noise[self.frame_ids_cur_chunk]] # list,[4,68,4,64,64]
        # self.chunk_size = chunk_size
        if self.perform_cross_frame_with_prev:
            if self.chunk_index == 0:
                self.controller.set_task("initfirst")
            else:
                self.controller.set_task("updatecur")

        images = self.pipe(
            chunk_index = chunk_index,
            # chunk_size = chunk_size,
            prompt=[self.config.inversion.prompt] * 3 *chunk_size, # 'a tea pot pouring tea into a cup.'
            latents=init_latents,
            guidance_scale=self.config.cfg_inversion_style,
            num_inference_steps=self.config.num_timesteps, # 100
            swap_guidance_scale=self.config.swap_guidance_scale, # 1.0
            callback=self.get_adain_callback(),
            eta=1,
            generator=torch.Generator('cuda').manual_seed(self.config.seed),
            cross_image_attention_range=Range(start=start_step, end=end_step),
            zs=init_zs,
            prev_latents_x0_list = self.prev_latents_x0_list,
            matching_save_dir = self.matching_save_dir,
            config = self.config,
            clip_model = self.clip_model,
            struct_gt = self.struct_tensor_image[frame_ids],
            enable_edit = self.enable_edit,
            std_file = self.mean_std_file,
            # perform_cross_frame=self.config.perform_cross_frame,
        ).images  # 注意这里guidance_scale = 1
        # joined_images = np.concatenate(images[::-1], axis=1)
        # Image.fromarray(joined_images).save(os.path.join(self.config.work_dir, f"chunk_inference_{chunk_size}_test.png"))
        if self.perform_cross_frame_with_prev:
            self.controller.set_task("updateprev")
            self.controller()
        return images
    def ensure_unique_save_path(self,save_path):
        original_save_path = save_path
        counter = 1
        while os.path.exists(save_path):
            save_path = f"{original_save_path}_{counter}"
            counter += 1
        return save_path
    def inversion_and_recon(self):
        '''
        一次保存10帧的结果
        '''
        # load or invert
        # self.load_latents_or_invert_videos()
        # self.n_frames = self.n_frames - self.start_frame # [start,end)
        # self.frame_ids = self.frame_ids[:self.n_frames]
        cur_batch_size = 10
        count = 0
        frames = load_video(self.struct_data_path, self.frame_height, self.frame_width, device=self.device) #先读取所有帧
        recon_save_path = os.path.join(self.config.inversion.save_path, 'recon_frames_batch')
        if self.config.inversion.n_frames:
            self.n_frames = self.config.inversion.n_frames
        else:
            self.n_frames = len(frames)
        print("cur video frames",self.n_frames)
        for i in range(0,self.n_frames, cur_batch_size):
            cur_batch_struct_save_path = os.path.join(self.struct_save_path,f'batch_frames{i}_{min(i+cur_batch_size,self.n_frames)}')
            os.makedirs(cur_batch_struct_save_path,exist_ok=True)
            if self.check_latent_exists(cur_batch_struct_save_path) and not self.config.inversion.force:
                continue
            latents = self.encode_imgs_batch(frames[i:min(i+cur_batch_size,self.n_frames)])
            wt, zs, wts = self.inversion_forward_process_batch(x0=latents, save_path=cur_batch_struct_save_path, etas=1,
                                                               prog_bar=True, prompt=self.config.inversion.prompt,
                                                               cfg_scale=self.config.cfg_inversion_style)  # 单帧frame_id = 0
            self.content_init, self.content_noise = self.load_latent(cur_batch_struct_save_path, choice="content")
            # # (5,4,64,64) (5,68,4,64,64)
            # latent_reconstruction, _ = self.inversion_reverse_process_batch(xT=wts[:, self.skip_steps], etas=1,
            #                                                                 prompts=[self.config.inversion.prompt], cfg_scales=[self.config.cfg_inversion_style],
            #                                                                 prog_bar=True,
            #                                                                 zs=zs[:, self.skip_steps:])
            ## load data
            latent_reconstruction, _ = self.inversion_reverse_process_batch(xT=self.content_init, etas=1,
                                                                            prompts=[self.config.inversion.prompt], cfg_scales=[self.config.cfg_inversion_style],
                                                                            prog_bar=True,
                                                                            zs=self.content_noise)

            torch.cuda.empty_cache()
            recon_frames = self.decode_latents_batch(latent_reconstruction)
            save_frames(recon_frames, recon_save_path,start_index=i)

    def debug(self, ablate_variable='', ablate_value=''):
        # load or invert
        result_stylized = []
        result_style = []
        result_content = []
        self.load_latents_or_invert_videos()
        self.content_init, self.content_noise = self.content_init[self.start_frame:self.n_frames], self.content_noise[self.start_frame:self.n_frames]
        self.n_frames = self.n_frames - self.start_frame # [start,end)
        self.frame_ids = self.frame_ids[:self.n_frames]
        frames_counter = 0
        self.enable_edit = self.config.enable_edit
        self.perform_cross_frame = self.config.perform_cross_frame
        self.perform_cross_frame_with_prev = self.config.perform_cross_frame_with_prev

        # print("chunk_ids",chunk_ids,"n_frame",self.n_frames,"frame_ids",self.frame_ids)
        # self.enable_edit = False  # Activate our cross-image attention layers
        post = f'{self.sd_version}_chunk_size{self.chunk_size}' + (
            '_cross_frame' if self.perform_cross_frame else '') + (
                   '_prev_frame' if self.perform_cross_frame_with_prev else '') + (
                   '_masked_adain' if self.config.use_masked_adain else '') + (
                   '_adain' if self.config.use_adain else '') + ('_latent_update' if self.latent_update else '') + \
               ((
                            f'matching_guidance_{self.config.update_with_matching_guidance}' + f'start_time{self.config.update_with_matching_start_time}' + f'end_time{self.config.update_with_matching_end_time}') if self.config.update_with_matching else '')
        if ablate_variable:
            # self.register_attention_control() # 需要再次调用才能生效
            self.cur_save_path = os.path.join(self.save_path, str(ablate_variable))  # self.save_path每次不能改变
            post = post + f'_{ablate_value}'
        else:
            self.cur_save_path = self.save_path
        self.cur_save_path = os.path.join(self.cur_save_path, post)
        self.cur_save_path = self.ensure_unique_save_path(self.cur_save_path)
        os.makedirs(self.cur_save_path, exist_ok=True)
        print("cur saving dir is", self.cur_save_path)
        # pyrallis.dump(self.config, open(os.path.join(self.cur_save_path, 'config.yaml'), 'w'))
        # 保存到文件
        OmegaConf.save(config=config, f=os.path.join(self.cur_save_path, 'config.yaml'))
        self.attention_output_file = os.path.join(self.cur_save_path, "attention_std.txt")
        # 清空文件内容
        with open(self.attention_output_file, "w") as f:
            pass
        # pyrallis.dump(self.config, open(os.path.join(self.cur_save_path, 'self.config.yaml'), 'w'))
        save_path = os.path.join(self.cur_save_path, f'generated_result')
        save_video = os.path.join(self.cur_save_path, f'recon_result')
        style_path = os.path.join(self.cur_save_path, f'style_result')
        intermediate_path = os.path.join(self.cur_save_path, f'intermediate_result')
        self.matching_save_dir = os.path.join(self.cur_save_path, f'sparse_matching_result')
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_video, exist_ok=True)
        os.makedirs(style_path, exist_ok=True)
        os.makedirs(intermediate_path, exist_ok=True)
        os.makedirs(self.matching_save_dir, exist_ok=True)
        self.struct_tensor_image = torch.cat([tensor_process(image) for image in self.struct_image], dim=0)
        ### debug
        mask_debug_save_dir = ['mask_cross_attention', 'mask_self_attention', 'mask_cluster', 'mask_binarization']
        self.mask_debug_directory = [os.path.join(self.cur_save_path, i) for i in mask_debug_save_dir]
        for directory in self.mask_debug_directory:
            os.makedirs(directory, exist_ok=True)
        if self.perform_cross_frame:
            chunk_ids = np.arange(0, self.n_frames, self.chunk_size - 1)
        elif self.perform_cross_frame_with_prev:
            chunk_ids = np.arange(0, self.n_frames, self.chunk_size)
        else:
            chunk_ids = np.arange(0, self.n_frames, self.chunk_size)
        for i in range(len(chunk_ids)):
            ch_start = chunk_ids[i]
            ch_end = self.n_frames if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
            prefix = [0] if self.perform_cross_frame else []
            # prefix = []
            frame_ids = prefix + list(range(ch_start, ch_end))  # 每个chunk会加上第一帧
            total = self.inference_chunk(frame_ids=frame_ids, chunk_index=i)
            cur_chunk_size = len(frame_ids)
            cur_res_generated = total[:cur_chunk_size]
            cur_res_style = total[cur_chunk_size:2 * cur_chunk_size]
            cur_res_content = total[2 * cur_chunk_size:]
            pre_len = len(prefix)
            result_stylized.extend(cur_res_generated[pre_len:])
            result_style.extend(cur_res_style[pre_len:])
            result_content.extend(cur_res_content[pre_len:])
            frames_counter += len(result_style)
            joined_images = np.concatenate(total[::-1], axis=1)
            Image.fromarray(joined_images).save(os.path.join(intermediate_path, f"out_joined_{i}.png"))

        torch.cuda.empty_cache()
        save_frames(result_stylized, save_path)
        frame_to_video(os.path.join(save_path, 'generated.mp4'), save_path)
        joined_images = np.concatenate(result_stylized, axis=1)
        Image.fromarray(joined_images).save(os.path.join(save_path, f"combined.png"))
        save_frames(result_content, save_video)
        save_frames(result_style, style_path)
        attention_std_save_path = os.path.join(self.cur_save_path, 'global_average_attention_plot.png')
        plot_global_average_attention_std(self.attention_output_file, attention_std_save_path)
    def __call__(self,ablate_variable='',ablate_value=''):
        result_stylized = []
        result_style = []
        result_content = []

        self.perform_cross_frame = self.config.perform_cross_frame
        self.perform_cross_frame_with_prev = self.config.perform_cross_frame_with_prev
        post = f'{self.sd_version}_chunk_size{self.chunk_size}'+ ('_cross_frame' if self.perform_cross_frame else '') + ('_prev_frame' if self.perform_cross_frame_with_prev else '')+('_masked_adain' if self.config.use_masked_adain else '') +('_adain' if self.config.use_adain else '') + ('_latent_update' if self.latent_update else '') + \
               ((f'matching_guidance_{self.config.update_with_matching_guidance}'+f'start_time{self.config.update_with_matching_start_time}'+ f'end_time{self.config.update_with_matching_end_time}') if self.config.update_with_matching else '')
        if ablate_variable:
            # self.register_attention_control() # 需要再次调用才能生效
            self.cur_save_path = os.path.join(self.save_path,str(ablate_variable)) # self.save_path每次不能改变
            post = post + f'_{ablate_value}'
        else:
            self.cur_save_path = self.save_path
        self.cur_save_path = os.path.join(self.cur_save_path,post)
        self.cur_save_path = self.ensure_unique_save_path(self.cur_save_path)
        os.makedirs(self.cur_save_path,exist_ok=True)
        print("cur saving dir is",self.cur_save_path)
        # 保存到文件
        OmegaConf.save(config=config, f=os.path.join(self.cur_save_path, 'config.yaml'))
        self.attention_output_file = os.path.join(self.cur_save_path, "attention_std.txt")
        self.mean_std_file = os.path.join(self.cur_save_path, "latent_mean_std.txt")
        self.adaptive_contrast_file = os.path.join(self.cur_save_path, "adaptive_contrast.txt")
        # 清空文件内容
        with open(self.attention_output_file, "w") as f:
            pass
        # 清空文件内容
        with open(self.mean_std_file, "w") as f:
            pass
        with open(self.adaptive_contrast_file, "w") as f:
            pass
        # pyrallis.dump(self.config, open(os.path.join(self.cur_save_path, 'self.config.yaml'), 'w'))
        save_path = os.path.join(self.cur_save_path, f'generated_result')
        save_video = os.path.join(self.cur_save_path, f'recon_result')
        style_path = os.path.join(self.cur_save_path, f'style_result')
        intermediate_path = os.path.join(self.cur_save_path, f'intermediate_result')
        self.matching_save_dir = os.path.join(self.cur_save_path, f'sparse_matching_result')
        os.makedirs(save_path,exist_ok=True)
        os.makedirs(save_video,exist_ok=True)
        os.makedirs(style_path,exist_ok=True)
        os.makedirs(intermediate_path,exist_ok=True)
        os.makedirs(self.matching_save_dir,exist_ok=True)
        ### 初始化style_init ###
        self.app_image, self.struct_image = image_utils.load_video_images(self.style_data_path, self.struct_data_path,self.struct_data_path)
        if not self.check_latent_exists(self.style_save_path):
            self.style_init, self.style_noises = invert_videos_and_image(sd_model=self.pipe,
                                                                         app_image=self.app_image,
                                                                         struct_image_list=self.struct_image,
                                                                         prompt=self.prompt,
                                                                         style_save_path=self.style_save_path,
                                                                         struct_save_path=self.struct_save_path,
                                                                         cfg=self.config,
                                                                         choice="style")
        self.enable_edit = self.config.enable_edit
        self.struct_tensor_image = torch.cat([tensor_process(image) for image in self.struct_image],dim=0)
        ### debug
        mask_debug_save_dir = ['mask_cross_attention','mask_self_attention','mask_cluster','mask_binarization']
        self.mask_debug_directory = [os.path.join(self.cur_save_path,i) for i in mask_debug_save_dir]
        for directory in self.mask_debug_directory:
            os.makedirs(directory,exist_ok=True)
        #### inference_all ####
        frames = load_video(self.struct_data_path, self.frame_height, self.frame_width, device=self.device)  # 先读取所有帧

        gt_frames = len(frames)
        self.style_init,self.style_noises = self.load_latent(self.style_save_path,choice="style")
        cur_batch_size = 10
        frames_counter = 0
        start_batch_index = 0 if not self.config.start_frame  else (self.config.start_frame // cur_batch_size)
        end_batch_index = gt_frames // cur_batch_size if not self.n_frames  else (self.n_frames // cur_batch_size)
        start_batch_remain = self.config.start_frame % cur_batch_size
        if self.n_frames:
            self.n_frames = min(len(frames), self.n_frames)
        else:
            self.n_frames = len(frames)
        end_batch_remain = self.n_frames % cur_batch_size

        cur_index = 0
        chunks_counter = 0
        for i in range(start_batch_index,self.n_frames, cur_batch_size):

            cur_batch_struct_save_path = os.path.join(self.struct_save_path,f'batch_frames{i}_{min(i+cur_batch_size,gt_frames)}') # 这个文件夹名称只和保存的时候真实的帧数有关
            self.content_init,self.content_noise = self.load_latent(cur_batch_struct_save_path,choice="content")
            if cur_index == start_batch_index:
                self.content_init = self.content_init[start_batch_remain:]
                self.content_noise = self.content_noise[start_batch_remain:]
            if cur_index == end_batch_index:
                self.content_init = self.content_init[:end_batch_remain]
                self.content_noise = self.content_noise[:end_batch_remain]
            #### 一个batch的生成 ####
            batch_size_tmp = len(self.content_init) # 当前处理的batch数目
            if self.perform_cross_frame:
                chunk_ids = np.arange(0, batch_size_tmp, self.chunk_size - 1)
            elif self.perform_cross_frame_with_prev:
                chunk_ids = np.arange(0, batch_size_tmp, self.chunk_size)
            else:
                chunk_ids = np.arange(0, batch_size_tmp, self.chunk_size)
            for j in range(len(chunk_ids)):
                ch_start = chunk_ids[j]
                ch_end = batch_size_tmp if j == len(chunk_ids) - 1 else chunk_ids[j + 1]
                prefix = [0] if self.perform_cross_frame else []
                frame_ids =  prefix + list(range(ch_start, ch_end))  # 每个chunk会加上第一帧
                total = self.inference_chunk(frame_ids =frame_ids,chunk_index = j)
                cur_chunk_size = len(frame_ids)
                cur_res_generated = total[:cur_chunk_size]
                cur_res_style = total[cur_chunk_size:2*cur_chunk_size]
                cur_res_content = total[2*cur_chunk_size:]
                pre_len = len(prefix)
                # result_stylized.extend(cur_res_generated[pre_len:])
                # result_style.extend(cur_res_style[pre_len:])
                # result_content.extend(cur_res_content[pre_len:])

                joined_images = np.concatenate(total[::-1], axis=1)
                Image.fromarray(joined_images).save(os.path.join(intermediate_path, f"out_joined_{chunks_counter}.png"))
                save_frames(cur_res_generated[pre_len:], save_path,start_index=frames_counter)
                save_frames(cur_res_content[pre_len:], save_video,start_index=frames_counter)
                save_frames(cur_res_style[pre_len:], style_path,start_index=frames_counter)
                frames_counter += len(cur_res_style[pre_len:])
                chunks_counter += 1
            cur_index += 1

        torch.cuda.empty_cache()
        frame_to_video(os.path.join(save_path,'generated.mp4'), save_path)
        # joined_images = np.concatenate(result_stylized, axis=1)
        # Image.fromarray(joined_images).save(os.path.join(save_path, f"combined.png"))
        attention_std_save_path = os.path.join(self.cur_save_path,'global_average_attention_plot.png')
        latents_std_save_path = os.path.join(self.cur_save_path,'latents_mean_std.png')
        adaptive_contrast_save_path = os.path.join(self.cur_save_path,'adaptive_contrast.png')

        plot_global_average_attention_std(self.attention_output_file,attention_std_save_path)
        plot_mean_std_over_time(self.mean_std_file,latents_std_save_path)
        if self.config.use_adaptive_contrast:
            plot_adaptive_contrast_over_time(self.adaptive_contrast_file,adaptive_contrast_save_path)

    def set_latents(self, latents_app: torch.Tensor, latents_struct: torch.Tensor):
        self.latents_app = latents_app
        self.latents_struct = latents_struct

    def set_noise(self, zs_app: torch.Tensor, zs_struct: torch.Tensor):
        self.zs_app = zs_app
        self.zs_struct = zs_struct

    def set_masks(self, masks: List[torch.Tensor]):
        #self.image_app_mask_32, self.image_struct_mask_32, self.image_app_mask_64, self.image_struct_mask_64 = masks
        self.image_app_mask_64, self.image_struct_mask_64 = masks

    def get_adain_callback(self):

        def callback(st: int, timestep, latents: torch.FloatTensor,pred_x0) -> Callable:
            self.step = st
            self.t = timestep.item() # t.item()

            self.segmentor.chunk_size = latents.shape[0] // 3 # chunk_size发生变化
            if self.latent_update:
                self.prev_latents_x0_list[self.t] = pred_x0[:self.segmentor.chunk_size][-1, :].clone().detach() # (1,4,64,64)
            # Compute the masks using prompt mixing self-segmentation and use the masks for AdaIN operation
            if self.config.use_masked_adain and self.step >= self.config.adain_range[0]: # 应该是大于的每个step都需要segment
                if self.step == self.config.adain_range[0]:
                    self.segmentor.setdirs(self.mask_debug_directory)
                masks = self.segmentor.get_object_masks(self.chunk_index)
                self.set_masks(masks)
            # Apply AdaIN operation using the computed masks
            if self.config.adain_range[0] <= self.step < self.config.adain_range[1]:
                if self.config.use_masked_adain:
                    # latents[0] = masked_adain(latents[0], latents[1], self.image_struct_mask_64, self.image_app_mask_64)
                    #print(latents[:self.segmentor.chunk_size].shape, latents[self.segmentor.chunk_size:2*self.segmentor.chunk_size].shape,self.image_struct_mask_64.shape, self.image_app_mask_64.shape)
                    latents[:self.segmentor.chunk_size] = masked_adain_batch(latents[:self.segmentor.chunk_size], latents[self.segmentor.chunk_size:2*self.segmentor.chunk_size], self.image_struct_mask_64, self.image_app_mask_64)
                elif self.config.use_adain:
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

            def __init__(self, name,place_in_unet: str,query_preserve=False,layer_name = ""):
                # self.chunk_size = chunk_size
                self.name = name
                self.place_in_unet = place_in_unet
                self.query_preserve = query_preserve
                # self.layer_name = layer_name
                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError("AttnProcessor2_0 requires torch 2.0, to use it, please upgrade torch to 2.0.")

            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         time_step = 0, # new add
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         perform_swap: bool = False,
                         perform_cross_frame: bool = True,):
                chunk_flag = False
                use_prev_frame = False
                if hidden_states.shape[0] < 3: # 单张图情况
                    chunk_size = 1
                else:
                    chunk_size = hidden_states.shape[0] // 3  # 单张图[1,4096,320]
                    chunk_flag = True
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

                if model_self.enable_edit and model_self.perform_cross_frame_with_prev and chunk_flag and model_self.controller.check_validation(hidden_states,self.place_in_unet):
                    hidden_states = model_self.controller(hidden_states, self.place_in_unet,time_step)  # Todo: ?? controller放置位置？ attn_weight:(3,8,1024,1024) 写的有点问题

                query = attn.to_q(hidden_states)


                is_cross = encoder_hidden_states is not None
                if not is_cross:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states) # (6,4096,320)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                should_mix = False

                # Potentially apply our cross image attention operation
                # To do so, we need to be in a self-attention layer in the decoder part of the denoising network
                
                vis_flag = False

                # print(f"perform_swap: {perform_swap}, is_cross: {is_cross}")
                # print("Key shape before rearrange:", key.shape)
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit: # decoder的自注意力层才进行这些操作
                    if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                        if self.place_in_unet not in model_self.key_injection_layers:
                            model_self.key_injection_layers.add(self.place_in_unet)
                        should_mix = True
                        key = rearrange(key, "(b f) d c -> b f d c", f=chunk_size)
                        value = rearrange(value, "(b f) d c -> b f d c", f=chunk_size)
                        query = rearrange(query, "(b f) d c -> b f d c", f=chunk_size)
                        if model_self.config.keep_struct and model_self.step % 5 == 0 and model_self.step < model_self.config.keep_struct_end:
                            # Inject the structure's keys and values
                            key[OUT_INDEX] = key[STRUCT_INDEX].clone() # key长度为3
                            value[OUT_INDEX] = value[STRUCT_INDEX].clone()
                        else:
                            # Inject the appearance's keys and values
                            key[OUT_INDEX] = key[STYLE_INDEX].clone()
                            value[OUT_INDEX] = value[STYLE_INDEX].clone()
                        # add query_preserve
                        if self.query_preserve and model_self.step < model_self.config.gamma_end:
                            # print(model_self.config.gamma)
                            vis_flag = True
                            query[OUT_INDEX] = query[STRUCT_INDEX]*model_self.config.gamma + query[OUT_INDEX]*(1-model_self.config.gamma)
                            # query[OUT_INDEX] = query[OUT_INDEX]*model_self.config.temperature
                        key = rearrange(key, "b f d c -> (b f) d c")
                        value = rearrange(value, "b f d c -> (b f) d c")
                        query = rearrange(query, "b f d c -> (b f) d c")

                if not is_cross and "up" in self.place_in_unet and model_self.enable_edit and not should_mix and query.shape[2] < 1281: # edit开关打开并且未进行query交换
                    # 进行cross_frame_attention
                    if model_self.perform_cross_frame and not is_cross:
                        former_frame_index = [0]*chunk_size
                        # print(f"former_frame_index: {former_frame_index}, key[OUT_INDEX].shape: {key[OUT_INDEX].shape}")
                        key = rearrange(key, "(b f) d c -> b f d c", f=chunk_size)  # [12,4096,320] -> [3,4,4096,320]
                        key[OUT_INDEX] = key[OUT_INDEX][former_frame_index]  # torch.Size([4, 4096, 320])
                        key = rearrange(key, "b f d c -> (b f) d c")
                        value = rearrange(value, "(b f) d c -> b f d c", f=chunk_size)
                        value[OUT_INDEX] = value[OUT_INDEX][former_frame_index]
                        value = rearrange(value, "b f d c -> (b f) d c")
                    elif model_self.perform_cross_frame_with_prev and not is_cross: # 第一帧的时候不进行
                        if model_self.chunk_index != 0:
                            # if self.place_in_unet in model_self.valid_layers: # 在替换的层当中
                            if model_self.controller.check_validation(hidden_states,self.place_in_unet):
                                total,d,c = query.shape
                                use_prev_frame = True
                                stylized_query = query[:chunk_size,:] # (1,1024,640)
                                stylized_key = key[:chunk_size,:] # (1,1024,640)
                                stylized_value = value[:chunk_size,:] # (1,1024,640)

                                new_stylized_key = torch.zeros((chunk_size,2*d,c))
                                new_stylized_values = torch.zeros((chunk_size,2*d,c))
                                for i in range(chunk_size):
                                    if i == 0:
                                        start_key,prev_key = model_self.controller.get_current_query(self.place_in_unet,time_step)
                                        new_stylized_key[i] = torch.cat([attn.to_k(start_key),attn.to_k(prev_key)],dim=1)
                                        new_stylized_values[i] = torch.cat([attn.to_v(start_key),attn.to_v(prev_key)],dim=1)
                                    else:
                                        start_key, _ = model_self.controller.get_current_query(self.place_in_unet,time_step)
                                        prev_key = stylized_key[i-1] # (1024,640)
                                        prev_value = stylized_value[i-1]
                                        new_stylized_key[i] = torch.cat([attn.to_k(start_key),prev_key.unsqueeze(0)],dim=1)
                                        new_stylized_values[i] = torch.cat([attn.to_v(start_key), prev_value.unsqueeze(0)], dim=1)

                                stylized_key = new_stylized_key.to(query.device) # (1,2048,640)
                                stylized_value = new_stylized_values.to(query.device) # (1,2048,640)
                                del new_stylized_key,new_stylized_values
                                gc.collect()
                                style_content_query = query[chunk_size:,:]
                                style_content_key = key[chunk_size:,:]
                                style_content_value = value[chunk_size:,:]
                            # 其余层正常的attention计算
                        else:
                            former_frame_index = [0] * chunk_size
                            # print(f"former_frame_index: {former_frame_index}, key[OUT_INDEX].shape: {key[OUT_INDEX].shape}")
                            key = rearrange(key, "(b f) d c -> b f d c",f=chunk_size)  # [12,4096,320] -> [3,4,4096,320]
                            key[OUT_INDEX] = key[OUT_INDEX][former_frame_index]  # torch.Size([4, 4096, 320])
                            key = rearrange(key, "b f d c -> (b f) d c")
                            value = rearrange(value, "(b f) d c -> b f d c", f=chunk_size)
                            value[OUT_INDEX] = value[OUT_INDEX][former_frame_index]
                            value = rearrange(value, "b f d c -> (b f) d c")

                # Compute the cross attention and apply our contrasting operation
                with torch.no_grad():
                    if model_self.enable_edit and use_prev_frame:
                        cur_batch_size = chunk_size
                        stylized_query = stylized_query.view(cur_batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # query.contiguous() (1,8,1024,80)
                        stylized_key = stylized_key.view(cur_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        stylized_value = stylized_value.view(cur_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        stylized_hidden_states, stylized_attn_weight,stylized_pre_attn_map = attention_utils.compute_scaled_dot_product_attention(
                            stylized_query, stylized_key, stylized_value,
                            edit_map=perform_swap and model_self.enable_edit and should_mix,
                            is_cross=is_cross,
                            contrast_strength=model_self.config.contrast_strength,
                            chunk_size = chunk_size,
                        ) # (1,8,1024,80),(1,8,1024,2048)
                        sc_batch_size = 2*chunk_size
                        style_content_query = style_content_query.view(sc_batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # query.contiguous()
                        style_content_key = style_content_key.view(sc_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        style_content_value = style_content_value.view(sc_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        style_content_hidden_states, style_content_attn_weight,style_content_pre_attn_map = attention_utils.compute_scaled_dot_product_attention(
                            style_content_query, style_content_key, style_content_value,
                            edit_map=perform_swap and model_self.enable_edit and should_mix,
                            is_cross=is_cross,
                            contrast_strength=model_self.config.contrast_strength,
                            chunk_size=chunk_size,
                        ) # (2,8,1024,80),(2,8,1024,1024)
                        # 组合 stylized 和 style/content 的结果
                        hidden_states = torch.cat([stylized_hidden_states, style_content_hidden_states], dim=0)
                        attn_weight = stylized_attn_weight
                        #attn_weight = torch.cat([stylized_attn_weight, style_content_attn_weight], dim=0)
                    else:
                        # print(query.shape,batch_size,attn.heads,head_dim,time_step)
                        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # query.contiguous()
                        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # （6，5，4096，64） ？？？
                        hidden_states, attn_weight,pre_attn_map = attention_utils.compute_scaled_dot_product_attention(
                            query, key, value,
                            edit_map=perform_swap and model_self.enable_edit and should_mix,
                            is_cross=is_cross,
                            contrast_strength=model_self.config.contrast_strength,
                            chunk_size=chunk_size,
                            use_sparse_attention=model_self.config.use_sparse_attention,
                        ) # attn_weight:(12,8,1024,1024) pre_attn_map:[6, 8, 4096, 4096]
                        ## visualize
                        idx = len(model_self.up_layers) // 2 - 2  # 倒数第二层
                        if model_self.enable_edit and model_self.chunk_index == 0 and self.place_in_unet == model_self.up_layers[2 * idx] and \
                                model_self.t_to_idx[time_step] % 5 == 0 and perform_swap:
                            query_save_dir = os.path.join(model_self.cur_save_path, "pca_vis")
                            os.makedirs(query_save_dir, exist_ok=True)
                            visualize_and_save_mean_features_pca(query[:chunk_size], int(model_self.step), query_save_dir,
                                                            self.place_in_unet, suffix="q_cs")
                            visualize_and_save_mean_features_pca(query[chunk_size:2*chunk_size], int(model_self.step), query_save_dir,
                                                            self.place_in_unet, suffix="q_s")
                            visualize_and_save_mean_features_pca(query[2 * chunk_size:,], int(model_self.step), query_save_dir,
                                                            self.place_in_unet, suffix="q_c")
                        if model_self.enable_edit and not is_cross and chunk_size > 1 and perform_swap: # Todo: inversion forward 和 backward不一样
                            map_dict = {0: 'stylized', 1: 'style', 2: 'struct'}
                            # if time_step == 1: # only last timestep
                            #     attn_map_mean = torch.mean(attn_weight, dim=1) # (1024,1024)
                            #     attention_map_save_dir = os.path.join(model_self.cur_save_path, "attn_map_vis")
                            #     os.makedirs(attention_map_save_dir,exist_ok=True)
                            #
                            #     for i in range(3):
                            #         save_path = os.path.join(attention_map_save_dir,f"layer_{self.place_in_unet}_timestep_{time_step}_{map_dict[i]}.png")
                            #         visualize_head(attn_map_mean[i*chunk_size], save_path)
                            ## attention std value
                            # output_file = os.path.join(model_self.cur_save_path, "attention_std.txt")
                            cur_std_list = []
                            cur_std_after_softmax = []
                            for i in range(3):
                                # std_dev = pre_attn_map[i*chunk_size].std().item()
                                # cur_chunk_std = []
                                # for j in range(chunk_size):
                                #     std_dev_tmp = pre_attn_map[i*chunk_size].std(dim=-1).mean()
                                #     cur_chunk_std.append(std_dev_tmp.item())
                                # std_dev = sum(cur_chunk_std) / len(cur_chunk_std)
                                # cur_std_list.append(std_dev)
                                # ### 方式一
                                # std_dev = pre_attn_map[i*chunk_size].std().item()
                                ### 方式二
                                std_dev = pre_attn_map[i * chunk_size].std(dim=-1).mean().item()
                                mean_dev = pre_attn_map[i * chunk_size].mean(dim=-1).mean().item()

                                std_dev_after = attn_weight[i * chunk_size].std(dim=-1).mean().item()
                                mean_dev_after = attn_weight[i * chunk_size].mean(dim=-1).mean().item()

                                cur_std_list.append(std_dev)
                                cur_std_after_softmax.append(std_dev_after)
                                #std_dev = pre_attn_map[i*chunk_size].std(dim=-1).mean().item()
                                # 将结果格式化为 "place_in_unet, t, value"
                                result_str = f"{self.place_in_unet},{map_dict[i]},{time_step},{std_dev},{mean_dev}\n"

                                # 将结果追加写入到 .txt 文件
                                with open(model_self.attention_output_file, 'a') as f: # Todo: invert 的时候这个值还不存在
                                    f.write(result_str)
                            # print("cur time step",time_step,'strength',cur_std_list[1] / cur_std_list[0])
                            if model_self.config.use_adaptive_contrast:
                                # ## 方法一:使用style的方差/ 当前的方差
                                # model_self.config.contrast_strength =cur_std_list[1] / cur_std_list[0]
                                ## 方法二: 使用struct自身的方差 / 当前的方差
                                model_self.config.contrast_strength = max(cur_std_list[2] / cur_std_list[0],cur_std_list[1] / cur_std_list[0])
                                after_softmax = max(cur_std_after_softmax[2] / cur_std_after_softmax[0],cur_std_after_softmax[1] / cur_std_after_softmax[0])
                                ### 方法二: 使用softmax之后的方差
                                model_self.config.contrast_strength = after_softmax
                                adaptive_contrast_str = f"{self.place_in_unet}, {time_step}, {model_self.config.contrast_strength},{after_softmax}\n"
                                # 将结果追加写入到 .txt 文件
                                with open(model_self.adaptive_contrast_file, 'a') as f:  # Todo: invert 的时候这个值还不存在
                                    f.write(adaptive_contrast_str)
                            #print("model_self.config.contrast_strength",model_self.config.contrast_strength)
                res = int(attn_weight.shape[2] ** 0.5)
                if model_self.enable_edit and model_self.chunk_index == 0 and not is_cross and model_self.struct_image and res == 64 and time_step == 1: # 经过cross frame attention之后大小会发生变化
                    image = Image.fromarray(model_self.struct_image[model_self.frame_ids_cur_chunk[0].item()])
                    single_query_dir = os.path.join(model_self.cur_save_path,"single_query_vis")
                    os.makedirs(single_query_dir,exist_ok=True)
                    average_att_map = torch.mean(attn_weight[OUT_INDEX].squeeze(0),dim = 0) # (4096,4096)
                    # should_mix
                    #x,y = 296,236 # breackdance 374,189 # blackswan
                    #points_list = [(295,239),(124,260),(469,119),(250,316)]
                    points_list = [(283,287),(374,189),(378,79),(433,138),(161,259)] # blackswan
                    for point in points_list:
                        x,y = point
                        visualize_grid_to_grid_normal(x, y, average_att_map, image, single_query_dir, self.place_in_unet + ('mix' if should_mix else ''), alpha=0.6)
                    # for grid_index in range(0,len(average_att_map),10):
                    #     visualize_grid_to_grid_normal(average_att_map,grid_index,image,single_query_dir,self.place_in_unet)
                # if model_self.controller is None:
                #     model_self.controller = DummyController()

                # TypeError: __call__() missing 2 required positional arguments: 'is_cross' and 'place_in_unet'
                # Update attention map for segmentation
                if model_self.config.use_masked_adain and model_self.step == model_self.config.adain_range[0] - 1: # model_self.config.adain_range[0]：20
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
                post = "self" if name.endswith("attn1") else "cross"
                # if name.endswith("attn1"):
                if place_in_unet == "down":
                    model_self.down_layers.append(place_in_unet + f"_{count + 1}_{post}") # place_in_unet + f"_{count + 1}_{post}"
                elif place_in_unet == "mid":
                    model_self.middle_layers.append(place_in_unet + f"_{count + 1}_{post}")
                elif place_in_unet == "up":
                    model_self.up_layers.append(place_in_unet + f"_{count + 1}_{post}")

                if len(model_self.up_layers) >= model_self.up_layers_start_index:
                    query_preserve = True
                    net_.set_processor(AttentionProcessor(name,place_in_unet + f"_{count + 1}_{post}",query_preserve))
                else:
                    net_.set_processor(AttentionProcessor(name,place_in_unet + f"_{count + 1}_{post}"))
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
                cross_att_count += register_recr(net,net_name, 0, "down") # down,up和mid的count分开计算
            elif "up" in net_name:
                cross_att_count += register_recr(net,net_name, 0, "up")
            elif "mid" in net_name:
                cross_att_count += register_recr(net,net_name, 0, "mid")
        print("total attention layers",cross_att_count)

if __name__ == "__main__":

    start_time = time.time()
    config = load_config()
    seed_everything(config.seed)
    generator = AppearanceTransferModel(config)
    generator.inversion_and_recon()
    end_time = time.time()
    print("total cost time", end_time - start_time)

'''
python3 video_appearance_transfer_model_recon.py --config Codes/cross-image-attention/configs/dog.yaml
'''