from typing import List, Optional, Callable

import torch
import torch.nn.functional as F

from config import RunConfig
from constants import OUT_INDEX, STRUCT_INDEX, STYLE_INDEX
from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from cross_image_utils import attention_utils
from cross_image_utils.adain import masked_adain, adain
from cross_image_utils.model_utils import get_stable_diffusion_model
from cross_image_utils.segmentation import Segmentor
from cross_image_utils.ddpm_inversion import AttentionStore
from cross_image_utils.attention_visualization import show_cross_attention,show_self_attention_comp,visualize_and_save_features_pca


class AppearanceTransferModel:

    def __init__(self, config: RunConfig, pipe: Optional[CrossImageAttentionStableDiffusionPipeline] = None):
        ### injected layers ### 只保存自注意力
        self.down_layers = []
        self.middle_layers = []
        self.up_layers = []
        
        self.config = config
        self.pipe = get_stable_diffusion_model(choice="video") if pipe is None else pipe
        self.controller = AttentionStore() # add controller for visualization
        self.register_attention_control()
        self.segmentor = Segmentor(prompt=config.prompt, object_nouns=[config.object_noun])
        self.latents_app, self.latents_struct = None, None
        self.zs_app, self.zs_struct = None, None
        self.image_app_mask_32, self.image_app_mask_64 = None, None
        self.image_struct_mask_32, self.image_struct_mask_64 = None, None
        self.enable_edit = False
        self.step = 0 # get_adain_callback的时候修改时间步

        

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
            # Compute the masks using prompt mixing self-segmentation and use the masks for AdaIN operation
            if self.config.use_masked_adain and self.step == self.config.adain_range.start:
                masks = self.segmentor.get_object_masks()
                self.set_masks(masks)
            # Apply AdaIN operation using the computed masks
            if self.config.adain_range.start <= self.step < self.config.adain_range.end:
                if self.config.use_masked_adain:
                    latents[0] = masked_adain(latents[0], latents[1], self.image_struct_mask_64, self.image_app_mask_64)
                else:
                    latents[0] = adain(latents[0], latents[1])

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
                         perform_swap: bool = False):

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
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit: # model_self.enable_edit：True
                    if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                        should_mix = True
                        if model_self.step % 5 == 0 and model_self.step < 40:
                            # Inject the structure's keys and values
                            key[OUT_INDEX] = key[STRUCT_INDEX] # key长度为3
                            value[OUT_INDEX] = value[STRUCT_INDEX]
                        else:
                            # Inject the appearance's keys and values
                            key[OUT_INDEX] = key[STYLE_INDEX]
                            value[OUT_INDEX] = value[STYLE_INDEX]
                        # add query_preserve
                        if self.query_preserve:
                            vis_flag = True
                            query[OUT_INDEX] = query[STRUCT_INDEX]*model_self.config.gamma + query[OUT_INDEX]*(1-model_self.config.gamma)
                            query[OUT_INDEX] = query[OUT_INDEX]*model_self.config.temperature
                            


                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
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
