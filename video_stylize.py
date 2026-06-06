import os
import json
import time
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from utils import load_video, save_frames, load_config, seed_everything
from utils import get_frame_ids, get_latents_dir
from cross_image_utils.video_util import frame_to_video
from cross_image_utils import image_utils
from cross_image_utils.latent_utils import invert_videos_and_image
from cross_image_utils.CLIP_model import tensor_process, init_clip_model, get_clip_model
from cross_image_utils.adain import masked_adain_batch, adain_batch
from cross_image_utils.segmentation_batch_separate import Segmentor
from cross_image_utils.ddpm_inversion import AttentionStore
from config import RunConfig, Range

from models.ddim_inversion import DDIMInversionMixin
from models.attention_control import AttentionControlMixin


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class AppearanceTransferModel(DDIMInversionMixin, AttentionControlMixin):

    def __init__(self, config, pipe=None):
        self.down_layers = []
        self.middle_layers = []
        self.up_layers = []

        self.config = config
        self.frame_ids = get_frame_ids(self.config.generation.frame_range, self.config.generation.frame_ids)
        self.sd_version = self.config.sd_version
        from cross_image_utils.model_utils import get_stable_diffusion_model
        self.pipe, self.model_key = get_stable_diffusion_model(
            self.sd_version, model_id=self.config.get("model_id")) if pipe is None else pipe
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
        self.up_layers_start_index = self.config.up_layers_start_index

        # 注意力控制（注册 processor 必须在 AttentionStore 之前 -> register + store）
        self.register_attention_control()
        self.controller = AttentionStore(
            timesteps=self.timesteps,
            layers=self.down_layers + self.up_layers,
            cond_layer=self.config.cond_layer)
        self.valid_layers = self.controller.get_valid_layers()
        self.chunk_size = self.config.generation.chunk_size

        self.n_frames = self.config.inversion.n_frames
        self.prompt = self.config.inversion.prompt
        self.update_with_matching = self.config.update_with_matching

        self.prev_latents_x0_list = {}
        for t in self.timesteps:
            self.prev_latents_x0_list[int(t.item())] = []

        # —— 路径 ——
        self.struct_data_path = self.config.input_path
        self.style_data_path = self.config.app_image_path
        self.work_dir = os.path.join(self.config.work_dir,
                                     self.config.input_path.rstrip("/").split('/')[-2])
        self.config.inversion.save_path = os.path.join(self.work_dir, "latents")
        self.struct_save_path = get_latents_dir(self.config.inversion.save_path, self.model_key)
        self.style_save_path = get_latents_dir(
            os.path.join(self.config.app_image_save_path,
                         os.path.basename(self.config.app_image_path).split('.')[0]),
            self.model_key)
        self.save_path = os.path.join(self.work_dir,
                                      os.path.basename(self.config.app_image_path).split('.')[0])
        os.makedirs(self.save_path, exist_ok=True)

        self.segmentor = Segmentor(config=self.config, tokenizer=self.pipe.tokenizer)

        # 实例变量
        self.latents_app = self.latents_struct = None
        self.zs_app = self.zs_struct = None
        self.image_app_mask_32 = self.image_struct_mask_32 = None
        self.image_app_mask_64 = self.image_struct_mask_64 = None
        self.enable_edit = False
        self.perform_cross_frame = False
        self.perform_cross_frame_with_prev = False
        self.step = 0

        self.device = self.config.device
        gene_cfg = self.config.generation
        fp = gene_cfg.float_precision if "float_precision" in gene_cfg else self.config.float_precision
        self.dtype = torch.float16 if fp == "fp16" else torch.float32
        print(f"[INFO] float precision {fp}. Use {self.dtype}.")
        self.batch_size = self.config.inversion.batch_size
        self.frame_height, self.frame_width = self.config.height, self.config.width

        self.t_to_idx = {int(v): k for k, v in enumerate(self.timesteps)}
        self.idx_to_t = {k: int(v) for k, v in enumerate(self.timesteps)}

        self.model_key = self.config.model_key
        self.latent_update = self.config.latent_update
        self.start_frame = self.config.start_frame
        self.frame_ids_cur_chunk = None
        self.app_image = self.struct_image = self.struct_tensor_image = None
        self.chunk_index = 0
        self.key_injection_layers = set()

        init_clip_model(vit_path=self.config.get("csd_clip_vit_path"),
                        checkpoint_path=self.config.get("csd_clip_checkpoint_path"))
        self.clip_model = get_clip_model()
        set_requires_grad(self.clip_model, False)

    # ------------------------------------------------------------------
    #  Latent I/O
    # ------------------------------------------------------------------
    def check_latent_exists(self, save_path):
        for ts in self.timesteps:
            if not os.path.exists(os.path.join(save_path, f'noisy_ddpm_{ts}.pt')):
                return False
        return os.path.exists(os.path.join(save_path, f'noisy_latents_{self.timesteps[0].item()}.pt'))

    def load_latents_or_invert_videos(self):
        if not self.config.use_edge:
            self.app_image, self.struct_image = image_utils.load_video_images(
                self.style_data_path, self.struct_data_path, self.struct_data_path)
        else:
            self.app_image, self.struct_image = image_utils.load_video_images(
                self.style_data_path, self.struct_data_path, self.struct_data_path_edge,
                edge_method=self.config.edge_method)
            self.struct_data_path = self.struct_data_path_edge
            self.struct_save_path = self.struct_save_path_edge

        if self.config.load_latents and self.check_latent_exists(self.struct_save_path) and self.check_latent_exists(self.style_save_path):
            print("Loading existing latents...")
            self.prepare_data()
        else:
            print("Inverting images...")
            self.enable_edit = False
            s_ok = self.check_latent_exists(self.struct_save_path)
            t_ok = self.check_latent_exists(self.style_save_path)
            if s_ok and not t_ok:
                self.content_init, self.content_noise = self.load_latent(self.struct_save_path, "content")
                self.style_init, self.style_noises = invert_videos_and_image(
                    self.pipe, self.app_image, self.struct_image, self.prompt,
                    self.style_save_path, self.struct_save_path, self.config, "style")
            elif t_ok and not s_ok:
                self.style_init, self.style_noises = self.load_latent(self.style_save_path, "style")
                self.content_init, self.content_noise = invert_videos_and_image(
                    self.pipe, self.app_image, self.struct_image, self.prompt,
                    self.style_save_path, self.struct_save_path, self.config, "content")
            else:
                self.content_init, self.content_noise = invert_videos_and_image(
                    self.pipe, self.app_image, self.struct_image, self.prompt,
                    self.style_save_path, self.struct_save_path, self.config, "content")
                self.style_init, self.style_noises = invert_videos_and_image(
                    self.pipe, self.app_image, self.struct_image, self.prompt,
                    self.style_save_path, self.struct_save_path, self.config, "style")
            self.enable_edit = True
        print("Done.")

    def load_latent(self, save_path, choice="content"):
        latent = torch.load(os.path.join(save_path, f'noisy_latents_{self.timesteps[0].item()}.pt'))
        noises = []
        for t in self.timesteps:
            noises.append(torch.load(os.path.join(save_path, f'noisy_ddpm_{t}.pt')))
        if choice == "style":
            return latent, torch.cat(noises, dim=0).unsqueeze(0)
        return latent, torch.stack(noises, dim=1)

    def prepare_data(self):
        self.style_init, self.style_noises = self.load_latent(self.style_save_path, "style")
        self.content_init, self.content_noise = self.load_latent(self.struct_save_path, "content")

    # ------------------------------------------------------------------
    #  Inference chunk
    # ------------------------------------------------------------------
    def inference_chunk(self, frame_ids, chunk_index):
        self.chunk_index = chunk_index
        chunk_size = len(frame_ids)
        self.frame_ids_cur_chunk = torch.tensor(frame_ids)
        ss = min(self.config.cross_attn_32_range[0], self.config.cross_attn_64_range[0])
        es = max(self.config.cross_attn_32_range[1], self.config.cross_attn_64_range[1])

        init_latents = torch.cat([
            self.content_init[self.frame_ids_cur_chunk],
            self.style_init.repeat((chunk_size, 1, 1, 1)),
            self.content_init[self.frame_ids_cur_chunk]
        ], dim=0)
        init_zs = [
            self.content_noise[self.frame_ids_cur_chunk],
            self.style_noises.repeat(chunk_size, 1, 1, 1, 1),
            self.content_noise[self.frame_ids_cur_chunk]
        ]

        if self.perform_cross_frame_with_prev:
            self.controller.set_task("initfirst" if self.chunk_index == 0 else "updatecur")

        images = self.pipe(
            chunk_index=chunk_index,
            prompt=[self.config.inversion.prompt] * 3 * chunk_size,
            latents=init_latents,
            guidance_scale=self.config.cfg_inversion_style,
            num_inference_steps=self.config.num_timesteps,
            swap_guidance_scale=self.config.swap_guidance_scale,
            callback=self.get_adain_callback(),
            eta=1,
            generator=torch.Generator('cuda').manual_seed(self.config.seed),
            cross_image_attention_range=Range(start=ss, end=es),
            zs=init_zs,
            prev_latents_x0_list=self.prev_latents_x0_list,
            matching_save_dir=self.matching_save_dir,
            config=self.config,
            clip_model=self.clip_model,
            struct_gt=self.struct_tensor_image[frame_ids],
            enable_edit=self.enable_edit,
            std_file=self.mean_std_file,
        ).images

        if self.perform_cross_frame_with_prev:
            self.controller.set_task("updateprev")
            self.controller()
        return images

    def ensure_unique_save_path(self, path):
        orig = path
        c = 1
        while os.path.exists(path):
            path = f"{orig}_{c}"
            c += 1
        return path

    # ------------------------------------------------------------------
    #  AdaIN callback
    # ------------------------------------------------------------------
    def get_adain_callback(self):
        def callback(st, timestep, latents, pred_x0):
            self.step = st
            self.t = timestep.item()
            self.segmentor.chunk_size = latents.shape[0] // 3
            if self.latent_update:
                self.prev_latents_x0_list[self.t] = pred_x0[:self.segmentor.chunk_size][-1].clone().detach()
            if self.config.use_masked_adain and self.step >= self.config.adain_range[0]:
                if self.step == self.config.adain_range[0]:
                    self.segmentor.setdirs(self.mask_debug_directory)
                masks = self.segmentor.get_object_masks(self.chunk_index)
                self.image_app_mask_64, self.image_struct_mask_64 = masks
            if self.config.adain_range[0] <= self.step < self.config.adain_range[1]:
                if self.config.use_masked_adain:
                    latents[:self.segmentor.chunk_size] = masked_adain_batch(
                        latents[:self.segmentor.chunk_size],
                        latents[self.segmentor.chunk_size:2 * self.segmentor.chunk_size],
                        self.image_struct_mask_64, self.image_app_mask_64)
                elif self.config.use_adain:
                    latents[:self.segmentor.chunk_size] = adain_batch(
                        latents[:self.segmentor.chunk_size],
                        latents[self.segmentor.chunk_size:2 * self.segmentor.chunk_size])
        return callback

    # ------------------------------------------------------------------
    #  Inversion & Recon (validation helper)
    # ------------------------------------------------------------------
    def inversion_and_recon(self):
        self.dtype = torch.float32
        cur_batch_size = 10
        frames = load_video(self.struct_data_path, self.frame_height, self.frame_width, device=self.device)
        recon_save_path = os.path.join(self.config.inversion.save_path, 'recon_frames_batch')
        self.n_frames = min(self.config.inversion.n_frames, len(frames)) if self.config.inversion.n_frames else len(frames)
        print("cur video frames", self.n_frames)

        for i in range(0, self.n_frames, cur_batch_size):
            end = min(i + cur_batch_size, self.n_frames)
            batch_save = os.path.join(self.struct_save_path, f'batch_frames{i}_{end}')
            os.makedirs(batch_save, exist_ok=True)
            if self.check_latent_exists(batch_save) and not self.config.inversion.force:
                continue
            latents = self.encode_imgs_batch(frames[i:end])
            wt, zs, wts = self.inversion_forward_process_batch(
                x0=latents, save_path=batch_save, etas=1, prog_bar=True,
                prompt=self.config.inversion.prompt, cfg_scale=self.config.cfg_inversion_contents)
            self.content_init, self.content_noise = self.load_latent(batch_save, choice="content")
            latent_recon, _ = self.inversion_reverse_process_batch(
                xT=self.content_init, etas=1, prompts=[self.config.inversion.prompt],
                cfg_scales=[self.config.cfg_inversion_style], prog_bar=True, zs=self.content_noise)
            torch.cuda.empty_cache()
            save_frames(self.decode_latents_batch(latent_recon), recon_save_path, start_index=i)
        self.dtype = torch.float16

    # ------------------------------------------------------------------
    #  Main call (stylization)
    # ------------------------------------------------------------------
    def __call__(self, ablate_variable='', ablate_value=''):
        if self.config.generate_type == "image":
            img_dir = os.path.join(self.config.images_save_dir,
                                   ("use_adaptive_contrast" if self.config.use_adaptive_contrast
                                    else f"contrast_strength{self.config.contrast_strength}")
                                   + f"_swap_guidance_scale{self.config.swap_guidance_scale}"
                                   + f"_gamma{self.config.gamma}")
            target = os.path.join(img_dir, self.config.input_path.split('/')[-2] + '_'
                                  + self.config.app_image_path.split('/')[-1].split('.')[0] + '.png')
            if os.path.isfile(target):
                return

        frames = load_video(self.struct_data_path, self.frame_height, self.frame_width, device=self.device)
        gt_frames = min(len(frames), self.config.inversion.n_frames)

        self.perform_cross_frame = self.config.perform_cross_frame
        self.perform_cross_frame_with_prev = self.config.perform_cross_frame_with_prev

        # Output path
        post = (f'{self.sd_version}_chunk_size{self.chunk_size}'
                + ('_cross_frame' if self.perform_cross_frame else '')
                + ('_prev_frame' if self.perform_cross_frame_with_prev else '')
                + ('_masked_adain' if self.config.use_masked_adain else '')
                + ('_adain' if self.config.use_adain else '')
                + ('_latent_update' if self.latent_update else '')
                + (f'_matching_guidance_{self.config.update_with_matching_guidance}'
                   f'_s{self.config.update_with_matching_start_time}'
                   f'_e{self.config.update_with_matching_end_time}') if self.config.update_with_matching else '')

        if ablate_variable:
            self.cur_save_path = os.path.join(self.save_path, str(ablate_variable))
            post = post + f'_{ablate_value}'
        else:
            self.cur_save_path = self.save_path
        self.cur_save_path = self.ensure_unique_save_path(os.path.join(self.cur_save_path, post))
        os.makedirs(self.cur_save_path, exist_ok=True)
        print("cur saving dir is", self.cur_save_path)
        OmegaConf.save(self.config, os.path.join(self.cur_save_path, 'config.yaml'))

        # Subdirectories
        out = {}
        for k, v in [('stylized', 'stylized_frames'), ('recon', 'content_recon'),
                      ('style', 'style_frames'), ('inter', 'intermediate'),
                      ('match', 'matching_vis'), ('masks', 'masks')]:
            out[k] = os.path.join(self.cur_save_path, v)
            os.makedirs(out[k], exist_ok=True)

        self.matching_save_dir = out['match']
        self.attention_output_file = os.path.join(self.cur_save_path, 'attention_std.txt')
        self.mean_std_file = os.path.join(self.cur_save_path, 'latent_mean_std.txt')
        self.adaptive_contrast_file = os.path.join(self.cur_save_path, 'adaptive_contrast.txt')
        for f in [self.attention_output_file, self.mean_std_file, self.adaptive_contrast_file]:
            open(f, 'w').close()

        mask_subs = ['mask_cross_attention', 'mask_self_attention', 'mask_cluster', 'mask_binarization']
        self.mask_debug_directory = [os.path.join(out['masks'], d) for d in mask_subs]
        for d in self.mask_debug_directory:
            os.makedirs(d, exist_ok=True)

        # Load style
        self.app_image, self.struct_image = image_utils.load_video_images(
            self.style_data_path, self.struct_data_path, self.struct_data_path)
        if not self.check_latent_exists(self.style_save_path) or self.config.inversion.force:
            self.style_init, self.style_noises = invert_videos_and_image(
                self.pipe, self.app_image, self.struct_image, self.prompt,
                self.style_save_path, self.struct_save_path, self.config, "style")

        self.enable_edit = self.config.enable_edit
        self.struct_tensor_image = torch.cat([tensor_process(img) for img in self.struct_image], dim=0)
        self.style_init, self.style_noises = self.load_latent(self.style_save_path, "style")

        # Batch generation
        cur_batch_size = 10
        frames_counter = 0
        chunks_counter = 0
        start_batch = 0 if not self.config.start_frame else self.config.start_frame // cur_batch_size
        end_batch = gt_frames // cur_batch_size if not self.n_frames else self.n_frames // cur_batch_size
        start_rem = self.config.start_frame % cur_batch_size
        self.n_frames = min(len(frames), self.n_frames) if self.n_frames else len(frames)
        end_rem = self.n_frames % cur_batch_size
        cur_index = 0

        for i in range(start_batch, self.n_frames, cur_batch_size):
            if i == 0 and gt_frames > 1:
                bs = os.path.join(self.struct_save_path, 'batch_frames0_10')
            else:
                bs = os.path.join(self.struct_save_path, f'batch_frames{i}_{min(i + cur_batch_size, gt_frames)}')

            self.content_init, self.content_noise = self.load_latent(bs, "content")
            if cur_index == start_batch:
                self.content_init = self.content_init[start_rem:]
                self.content_noise = self.content_noise[start_rem:]
            if cur_index == end_batch:
                self.content_init = self.content_init[:end_rem]
                self.content_noise = self.content_noise[:end_rem]

            bt = len(self.content_init)
            cids = np.arange(0, bt, self.chunk_size - 1 if self.perform_cross_frame else self.chunk_size)
            for j in range(len(cids)):
                cs = cids[j]
                ce = bt if j == len(cids) - 1 else cids[j + 1]
                prefix = [0] if self.perform_cross_frame else []
                fids = prefix + list(range(cs, ce))
                total = self.inference_chunk(frame_ids=fids, chunk_index=j)
                ck = len(fids)
                pl = len(prefix)

                joined = np.concatenate(total[::-1], axis=1)
                Image.fromarray(joined).save(os.path.join(out['inter'], f"chunk_{chunks_counter}.png"))
                save_frames(total[:ck][pl:], out['stylized'], start_index=frames_counter)
                save_frames(total[2 * ck:][pl:], out['recon'], start_index=frames_counter)
                save_frames(total[ck:2 * ck][pl:], out['style'], start_index=frames_counter)
                frames_counter += len(total[ck:2 * ck][pl:])
                chunks_counter += 1
            cur_index += 1

        torch.cuda.empty_cache()

        if self.config.generate_type == "image":
            import shutil
            shutil.copy2(os.path.join(out['inter'], f"chunk_{chunks_counter - 1}.png"), target)
        else:
            frame_to_video(os.path.join(out['stylized'], 'generated.mp4'), out['stylized'])

        from cross_image_utils.figures_visualization.attention_map_std import (
            plot_global_average_attention_std, plot_mean_std_over_time, plot_adaptive_contrast_over_time)
        plot_global_average_attention_std(self.attention_output_file,
                                          os.path.join(self.cur_save_path, 'attention_std.png'))
        plot_mean_std_over_time(self.mean_std_file,
                                os.path.join(self.cur_save_path, 'latent_mean_std.png'))
        if self.n_frames > 1 and self.config.use_adaptive_contrast:
            plot_adaptive_contrast_over_time(self.adaptive_contrast_file,
                                             os.path.join(self.cur_save_path, 'adaptive_contrast.png'))


if __name__ == "__main__":
    start_time = time.time()
    config = load_config()
    seed_everything(config.seed)
    generator = AppearanceTransferModel(config)
    generator()
    end_time = time.time()
    print("total cost time", end_time - start_time)