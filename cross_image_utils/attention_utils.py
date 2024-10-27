import math
import torch

from constants import OUT_INDEX
import numpy as np
import os
import torchvision.transforms as T
from PIL import Image

def show_tensor_image(save_dir,tensor_imgs,pre_fix,post_fix,save_size=(512,512)):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    to_pil = T.ToPILImage()
    if len(tensor_imgs.shape) == 3:
        for i in range(tensor_imgs.shape[0]):
            mask_image = to_pil(tensor_imgs[i].float()).resize(save_size)
            save_path = os.path.join(save_dir,f"{pre_fix}_{post_fix}_{i}.png")
            mask_image.save(save_path)
    else:
        mask_image = to_pil(tensor_imgs.float()).resize(save_size)
        save_path = os.path.join(save_dir, f"{pre_fix}_{post_fix}_0.png")
        mask_image.save(save_path)


def show_cluster_image(save_dir,cluster,post_fix,segments = 5):
    # 定义颜色映射
    color_map = {
        0: (255, 0, 0),  # 红色
        1: (0, 255, 0),  # 绿色
        2: (0, 0, 255),  # 蓝色
        3: (255, 255, 0),  # 黄色
        4: (255, 0, 255)  # 品红色
    }
    if len(cluster.shape) == 2:
        row,col = cluster.shape
        rgb_image = np.zeros((row, col, 3), dtype=np.uint8)
        for i in range(row):
            for j in range(col):
                rgb_image[i, j,:] = color_map[cluster[i, j]]
        # 将 numpy 数组转换为 PIL 图像
        image = Image.fromarray(rgb_image)
        save_path = os.path.join(save_dir, f'cluster_{post_fix}_seg{segments}.png')
        image.save(save_path)
    else:
        b,row,col = cluster.shape
        for k in range(b):
            rgb_image = np.zeros((row, col, 3), dtype=np.uint8)
            for i in range(row):
                for j in range(col):
                    rgb_image[i, j,:] = color_map[cluster[k,i, j]]
            # 将 numpy 数组转换为 PIL 图像
            image = Image.fromarray(rgb_image)
            save_path = os.path.join(save_dir, f'cluster_{post_fix}_seg{segments}_{k}.png')
            image.save(save_path)
    print(f"Image saved to {save_dir}")


def should_mix_keys_and_values(model, hidden_states: torch.Tensor) -> bool:
    """ Verify whether we should perform the mixing in the current timestep.  """
    is_in_32_timestep_range = (
            model.config.cross_attn_32_range.start <= model.step < model.config.cross_attn_32_range.end
    ) # (10,70)
    is_in_64_timestep_range = (
            model.config.cross_attn_64_range.start <= model.step < model.config.cross_attn_64_range.end
    ) # (10,90)
    is_hidden_states_32_square = (hidden_states.shape[1] == 32 ** 2)
    is_hidden_states_64_square = (hidden_states.shape[1] == 64 ** 2)
    should_mix = (is_in_32_timestep_range and is_hidden_states_32_square) or \
                 (is_in_64_timestep_range and is_hidden_states_64_square)
    return should_mix


def compute_scaled_dot_product_attention(Q, K, V, edit_map=False, is_cross=False, contrast_strength=1.0):
    """ Compute the scale dot product attention, potentially with our contrasting operation. """
    torch.cuda.empty_cache()

    #attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    try:
        # 你要运行的代码
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    except torch.cuda.OutOfMemoryError as e:
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        print(f"CUDA Out of Memory Error: {e}")
        print(f"Size of Q: {Q.size()},Size of k: {K.size()}，Size of V: {V.size()}")  # 打印 Q 的大小
    if edit_map and not is_cross:
        attn_weight[OUT_INDEX] = torch.stack([
            torch.clip(enhance_tensor(attn_weight[OUT_INDEX][head_idx], contrast_factor=contrast_strength),
                       min=0.0, max=1.0)
            for head_idx in range(attn_weight.shape[1])
        ]) # attn_weight:(3,8,1024,1024)
    return attn_weight @ V, attn_weight # V:torch.Size([3, 8, 1024, 80])


def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)
    return adjusted_tensor
