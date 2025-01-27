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
            model.config.cross_attn_32_range[0] <= model.step < model.config.cross_attn_32_range[1]
    ) # (10,70)
    is_in_64_timestep_range = (
            model.config.cross_attn_64_range[0] <= model.step < model.config.cross_attn_64_range[1]
    ) # (10,90)
    is_hidden_states_32_square = (hidden_states.shape[1] == 32 ** 2)
    is_hidden_states_64_square = (hidden_states.shape[1] == 64 ** 2)
    should_mix = (is_in_32_timestep_range and is_hidden_states_32_square) or \
                 (is_in_64_timestep_range and is_hidden_states_64_square)
    return should_mix


import torch

def sparse_attention_map(attention_map, top_k=1):
    """
    对 attention map 进行稀疏化处理，保留每个 query 响应最大的区域。

    Parameters:
    - attention_map: Tensor of shape (B, N, S, S) -> (batch_size, num_heads, num_keys, num_keys)
    - top_k: 每个 query 保留的最大响应值的数量。默认值为 1，表示只保留最大响应区域。

    Returns:
    - sparse_attention_map: 稀疏化后的 attention map，大小为 (B, N, S, S)，只保留最大响应区域。
    """
    # 获取最大响应值及其索引
    max_values, max_indices = torch.topk(attention_map, top_k, dim=-1, largest=True, sorted=False)

    # 创建一个新的稀疏 attention map，初始化为全0
    sparse_attention_map = torch.zeros_like(attention_map)

    # 将最大响应的区域设置为对应值
    sparse_attention_map.scatter_(-1, max_indices, max_values)

    return sparse_attention_map


def compute_scaled_dot_product_attention(Q, K, V,chunk_size,edit_map=False, is_cross=False, contrast_strength=1.0,use_sparse_attention=False):
    """ Compute the scale dot product attention, potentially with our contrasting operation. """
    torch.cuda.empty_cache() # attn_weight:(12,8,1024,1024)

    #attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    try:
        pre_attn_map = (Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) # softmax之前的结果
        # 你要运行的代码
        attn_weight = torch.softmax(pre_attn_map, dim=-1)
    except torch.cuda.OutOfMemoryError as e:
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        print(f"CUDA Out of Memory Error: {e}")
        print(f"Size of Q: {Q.size()},Size of k: {K.size()}，Size of V: {V.size()}")  # 打印 Q 的大小
    if edit_map and not is_cross:
        attn_weight[OUT_INDEX:(OUT_INDEX+1)*chunk_size] = torch.stack([
            torch.clip(enhance_tensor(attn_weight[OUT_INDEX:(OUT_INDEX+1)*chunk_size][:,head_idx], contrast_factor=contrast_strength),
                       min=0.0, max=1.0)
            for head_idx in range(attn_weight.shape[1])
        ],dim=1) # attn_weight:(3,1024,1024)->(3,8,1024,1024)
    if edit_map and not is_cross and use_sparse_attention:
        attn_weight[:chunk_size,] = sparse_attention_map(attention_map=attn_weight[:chunk_size,]) # (6,5,4096,4096)
    return attn_weight @ V, attn_weight,pre_attn_map # V:torch.Size([3, 8, 1024, 80])


def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """ # (3,1024,1024)
    # adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)
    adjusted_tensor = (tensor - tensor.mean(dim=-1,keepdim=True)) * contrast_factor + tensor.mean(dim=-1,keepdim=True)
    return adjusted_tensor
