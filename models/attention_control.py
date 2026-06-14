"""
Attention control: cross-image-attention processor + registration
Mixin for AppearanceTransferModel
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from cross_image_utils import attention_utils


class AttentionControlMixin:
    """Mixin: 提供注意力控制相关方法
    要求宿主类提供:
        self.down_layers, self.middle_layers, self.up_layers (初始化为 [])
        self.pipe, self.up_layers_start_index, self.config
    """

    class DummyController:
        def __call__(self, *args):
            return args[0]
        def __init__(self):
            self.num_att_layers = 0

    def register_attention_control(self):
        model_self = self
        from constants import OUT_INDEX, STRUCT_INDEX, STYLE_INDEX

        class AttentionProcessor:
            def __init__(self, name, place_in_unet, query_preserve=False):
                self.name = name
                self.place_in_unet = place_in_unet
                self.query_preserve = query_preserve

            def __call__(self, attn, hidden_states, time_step=0, forward_type='inversion',
                         encoder_hidden_states=None, attention_mask=None, temb=None,
                         perform_swap=False, perform_cross_frame=True):
                use_prev_frame = False
                chunk_size = 1 if hidden_states.shape[0] < 3 else hidden_states.shape[0] // 3
                chunk_flag = hidden_states.shape[0] >= 3
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim
                if input_ndim == 4:
                    bs, c, h, w = hidden_states.shape
                    hidden_states = hidden_states.view(bs, c, h * w).transpose(1, 2)

                bs, seq_len, _ = (hidden_states.shape if encoder_hidden_states is None
                                  else encoder_hidden_states.shape)
                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bs)
                    attention_mask = attention_mask.view(bs, attn.heads, -1, attention_mask.shape[-1])
                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                if model_self.enable_edit and model_self.perform_cross_frame_with_prev and chunk_flag \
                        and model_self.controller.check_validation(hidden_states, self.place_in_unet):
                    hidden_states = model_self.controller(hidden_states, self.place_in_unet, time_step)

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

                # --- Cross-image attention: swap keys/values ---
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                    if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                        if self.place_in_unet not in model_self.key_injection_layers:
                            model_self.key_injection_layers.add(self.place_in_unet)
                        should_mix = True
                        key = rearrange(key, "(b f) d c -> b f d c", f=chunk_size)
                        value = rearrange(value, "(b f) d c -> b f d c", f=chunk_size)
                        query = rearrange(query, "(b f) d c -> b f d c", f=chunk_size)
                        if model_self.config.keep_struct and model_self.step % 5 == 0 and model_self.step < model_self.config.keep_struct_end:
                            key[OUT_INDEX] = key[STRUCT_INDEX].clone()
                            value[OUT_INDEX] = value[STRUCT_INDEX].clone()
                        else:
                            key[OUT_INDEX] = key[STYLE_INDEX].clone()
                            value[OUT_INDEX] = value[STYLE_INDEX].clone()
                        if self.query_preserve and model_self.step < model_self.config.gamma_end:
                            query[OUT_INDEX] = (query[STRUCT_INDEX] * model_self.config.gamma
                                                + query[OUT_INDEX] * (1 - model_self.config.gamma))
                        key = rearrange(key, "b f d c -> (b f) d c")
                        value = rearrange(value, "b f d c -> (b f) d c")
                        query = rearrange(query, "b f d c -> (b f) d c")

                # --- Cross-frame attention ---
                if perform_swap and (not is_cross) and "up" in self.place_in_unet and model_self.enable_edit \
                        and not should_mix and query.shape[2] < 1281:
                    if model_self.perform_cross_frame:
                        idx = [0] * chunk_size
                        key = rearrange(key, "(b f) d c -> b f d c", f=chunk_size)
                        key[OUT_INDEX] = key[OUT_INDEX][idx]
                        key = rearrange(key, "b f d c -> (b f) d c")
                        value = rearrange(value, "(b f) d c -> b f d c", f=chunk_size)
                        value[OUT_INDEX] = value[OUT_INDEX][idx]
                        value = rearrange(value, "b f d c -> (b f) d c")
                    elif model_self.perform_cross_frame_with_prev:
                        if model_self.chunk_index != 0:
                            if model_self.controller.check_validation(hidden_states, self.place_in_unet):
                                total, d, c = query.shape
                                use_prev_frame = True
                                sty_q = query[:chunk_size]
                                sty_k = key[:chunk_size]
                                sty_v = value[:chunk_size]
                                new_k = torch.zeros((chunk_size, 2 * d, c), device=query.device)
                                new_v = torch.zeros((chunk_size, 2 * d, c), device=query.device)
                                for i in range(chunk_size):
                                    sk, pk = model_self.controller.get_current_query(self.place_in_unet, time_step)
                                    new_k[i] = torch.cat([attn.to_k(sk), attn.to_k(pk)], dim=1) if i == 0 \
                                        else torch.cat([attn.to_k(sk), sty_k[i - 1].unsqueeze(0)], dim=1)
                                    new_v[i] = torch.cat([attn.to_v(sk), attn.to_v(pk)], dim=1) if i == 0 \
                                        else torch.cat([attn.to_v(sk), sty_v[i - 1].unsqueeze(0)], dim=1)
                                sty_k, sty_v = new_k, new_v
                                sc_q = query[chunk_size:]
                                sc_k = key[chunk_size:]
                                sc_v = value[chunk_size:]
                        else:
                            idx = [0] * chunk_size
                            key = rearrange(key, "(b f) d c -> b f d c", f=chunk_size)
                            key[OUT_INDEX] = key[OUT_INDEX][idx]
                            key = rearrange(key, "b f d c -> (b f) d c")
                            value = rearrange(value, "(b f) d c -> b f d c", f=chunk_size)
                            value[OUT_INDEX] = value[OUT_INDEX][idx]
                            value = rearrange(value, "b f d c -> (b f) d c")

                # --- Scaled dot-product attention ---
                with torch.no_grad():
                    if model_self.enable_edit and use_prev_frame:
                        sty_q = sty_q.view(chunk_size, -1, attn.heads, head_dim).transpose(1, 2)
                        sty_k = sty_k.view(chunk_size, -1, attn.heads, head_dim).transpose(1, 2)
                        sty_v = sty_v.view(chunk_size, -1, attn.heads, head_dim).transpose(1, 2)
                        hs1, _, _ = attention_utils.compute_scaled_dot_product_attention(
                            sty_q, sty_k, sty_v,
                            edit_map=perform_swap and model_self.enable_edit and should_mix,
                            is_cross=is_cross, contrast_strength=model_self.config.contrast_strength,
                            chunk_size=chunk_size)
                        sc_q = sc_q.view(2 * chunk_size, -1, attn.heads, head_dim).transpose(1, 2)
                        sc_k = sc_k.view(2 * chunk_size, -1, attn.heads, head_dim).transpose(1, 2)
                        sc_v = sc_v.view(2 * chunk_size, -1, attn.heads, head_dim).transpose(1, 2)
                        hs2, _, _ = attention_utils.compute_scaled_dot_product_attention(
                            sc_q, sc_k, sc_v,
                            edit_map=perform_swap and model_self.enable_edit and should_mix,
                            is_cross=is_cross, contrast_strength=model_self.config.contrast_strength,
                            chunk_size=chunk_size)
                        hidden_states = torch.cat([hs1, hs2], dim=0)
                        attn_weight = None
                    else:
                        query = query.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
                        key = key.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
                        value = value.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
                        hidden_states, attn_weight, pre_attn_map = attention_utils.compute_scaled_dot_product_attention(
                            query, key, value,
                            edit_map=perform_swap and model_self.enable_edit and should_mix,
                            is_cross=is_cross, contrast_strength=model_self.config.contrast_strength,
                            chunk_size=chunk_size, use_sparse_attention=model_self.config.use_sparse_attention)

                        if forward_type == 'forward' and model_self.enable_edit and not is_cross and perform_swap:
                            cur_std = [pre_attn_map[i * chunk_size].std(dim=-1).mean().item() for i in range(3)]
                            cur_std_after = [attn_weight[i * chunk_size].std(dim=-1).mean().item() for i in range(3)]
                            map_names = {0: 'stylized', 1: 'style', 2: 'struct'}
                            for i in range(3):
                                with open(model_self.attention_output_file, 'a') as f:
                                    f.write(f"{self.place_in_unet},{map_names[i]},{time_step},{cur_std[i]}\n")
                            if model_self.config.use_adaptive_contrast:
                                cs = max(cur_std_after[2] / cur_std_after[0], cur_std_after[1] / cur_std_after[0])
                                after_softmax = cs
                                model_self.config.contrast_strength = after_softmax
                                with open(model_self.adaptive_contrast_file, 'a') as f:
                                    f.write(f"{self.place_in_unet}, {time_step}, {model_self.config.contrast_strength},{after_softmax}\n")

                    if attn_weight is not None and model_self.config.use_masked_adain \
                            and model_self.step == model_self.config.adain_range[0] - 1:
                        model_self.segmentor.update_attention(attn_weight, is_cross)

                hidden_states = hidden_states.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[OUT_INDEX].dtype)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(bs, c, h, w)
                if attn.residual_connection:
                    hidden_states = hidden_states + residual
                hidden_states = hidden_states / attn.rescale_output_factor
                return hidden_states

        def register_recr(net_, name, count, place_in_unet):
            if net_.__class__.__name__ == 'Attention':
                post = "self" if name.endswith("attn1") else "cross"
                key = f"{place_in_unet}_{count + 1}_{post}"
                if place_in_unet == "down":
                    model_self.down_layers.append(key)
                elif place_in_unet == "mid":
                    model_self.middle_layers.append(key)
                elif place_in_unet == "up":
                    model_self.up_layers.append(key)
                qp = len(model_self.up_layers) >= model_self.up_layers_start_index
                net_.set_processor(AttentionProcessor(name, key, qp))
                return count + 1
            elif hasattr(net_, 'children'):
                for child_name, net__ in net_.named_children():
                    new_name = f"{name}.{child_name}" if name else child_name
                    count = register_recr(net__, new_name, count, place_in_unet)
            return count

        for net_name, net in self.pipe.unet.named_children():
            if "down" in net_name:
                register_recr(net, net_name, 0, "down")
            elif "up" in net_name:
                register_recr(net, net_name, 0, "up")
            elif "mid" in net_name:
                register_recr(net, net_name, 0, "mid")