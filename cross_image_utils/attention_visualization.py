'''
1. tokenizer definition  ldm.tokernizer??
2.

'''
from operator import index

from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from charset_normalizer.cli import query_yes_no

from cross_image_utils.ddpm_inversion import AttentionStore
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
from cross_image_utils import ptp_utils
import math
def on_click(x,y,height=512,width=512,grid=14): # (142,133)
    grid_size = grid
    grid_index = y // (height / grid_size) * grid_size + x // (width / grid_size)
    return int(grid_index)

def aggregate_attention(prompts,attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()
def show_cross_attention(tokenizer,prompts,save_path,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts,attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.save_images(save_path,np.stack(images, axis=0))
def show_cross_attention_batch(tokenizer,prompts,attention_maps,save_dir, select: int = 0):
    '''
    input:tensor
    index: stylized or style or content
    attention_maps:(3,8,256,77) batch,num_heads,size,token_ids
    '''
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    # attention_maps = aggregate_attention(prompts,attention_store, res, from_where, True, select)

    b,heads,size,token_nums = attention_maps.shape
    res = int(size ** 0.5)
    # cross_attn = cross_attn.mean(dim=0).reshape(res, res, -1)  # (8,256,77) -> (256,77) -> (16,16,77)
    for j in range(b):
        images = []
        attention_map = attention_maps[j] # (8,256,77)
        attention_map = attention_map.mean(dim=0).reshape(res, res, -1)
        for i in range(len(tokens)):
            image = attention_map[:, :, i] # (16,16,77)
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        save_path = os.path.join(save_dir,f"batch_{j}.png")
        ptp_utils.save_images(save_path,np.stack(images, axis=0))

def show_self_attention_comp_batch(save_dir,attention_maps,max_com=10):
    '''
    input:numpy
    '''
    b,heads,size,size = attention_maps.shape # (256,256)
    res = int(size ** 0.5)
    for j in range(b):
        attention_map = attention_maps[j]
        attention_map = attention_map.mean(axis=0) # (1024,1024)
        u, s, vh = np.linalg.svd(attention_map - np.mean(attention_map, axis=1, keepdims=True))
        images = []
        for i in range(max_com):
            image = vh[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
            image = Image.fromarray(image).resize((256, 256))
            image = np.array(image)
            images.append(image)
        save_path = os.path.join(save_dir, f"batch_{j}_{res}.png")
        ptp_utils.save_images(save_path,np.concatenate(images, axis=1))
def show_self_attention_comp(save_path,attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.save_images(np.concatenate(images, axis=1))


from sklearn.decomposition import PCA
from PIL import Image
import os

# def visualize_and_save_features_pca(feats_map, t, save_dir, layer_idx,suffix=""):
#     """
#     feats_map: [B, N, D],query
#     t: timestep
#     save_dir: pic save dir
#     layer_idx: current layer idx
#     """
#     B = len(feats_map)
#     feats_map = feats_map.flatten(0, -2)
#     feats_map = feats_map.cpu().numpy()
#     pca = PCA(n_components=10)
#     pca.fit(feats_map)
#     feature_maps_pca = pca.transform(feats_map)  # N X 3
#     feature_maps_pca = feature_maps_pca.reshape(B, -1, 3)  # B x (H * W) x 3
#     for i, experiment in enumerate(feature_maps_pca):
#         pca_img = feature_maps_pca[i]  # (H * W) x 3
#         h = w = int(np.sqrt(pca_img.shape[0]))
#         pca_img = pca_img.reshape(h, w, 3)
#         pca_img_min = pca_img.min(axis=(0, 1))
#         pca_img_max = pca_img.max(axis=(0, 1))
#         pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
#         pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
#         if not suffix:
#             pca_img.save(os.path.join(save_dir, f"{i}_time_{t}_layer_{layer_idx}.png"))
#         else:
#             pca_img.save(os.path.join(save_dir, f"{i}_time_{t}_layer_{layer_idx}_{suffix}.png"))

def vis_query_distributions(query,save_dir,postfix=""):
    '''
    [N,B,D] cuda tensor
    '''
    query = query.cpu().numpy()
    query_flat = query.reshape(-1) # view为tensor使用方法
    plt.figure(figsize=(8,6))
    plt.hist(query_flat, bins=100,color='blue',alpha=0.7)
    plt.title('distribution of query feature values')
    plt.xlabel('feature values')
    plt.ylabel('frequency')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"query_distribution_{postfix}.png"))

def visualize_unet_features_pca(feats_map, t, save_dir, layer_idx,n_components=3, suffix=""):
    """
    feats_map: [B, N, D],feature:[320, 64, 64] cuda tensor
    t: timestep
    save_dir: pic save dir
    layer_idx: current layer idx
    grid_size: the size of the grid to combine the images (rows, columns)
    """
    b,size,size = feats_map.shape
    # 将特征重塑为 (320, 64*64)
    feature_reshaped = feats_map.detach().cpu().numpy().reshape(-1,b)

    # 进行PCA分解
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(feature_reshaped)

    # 将PCA结果重塑为 (64, 64, 3)
    pca_image = pca_result.reshape(size, size, n_components)

    # 将PCA结果归一化到 [0, 255]
    pca_image_normalized = ((pca_image - pca_image.min()) / (pca_image.max() - pca_image.min()) * 255).astype(np.uint8)

    # 创建PIL图像对象
    combined_img = Image.fromarray(pca_image_normalized).resize((256,256))

    # 保存图像
    if not suffix:
        combined_img.save(
            os.path.join(save_dir, f"combined_time_{t.item()}_layer_{layer_idx}_n_components_{n_components}.png"))
    else:
        combined_img.save(os.path.join(save_dir,
                                       f"combined_time_{t.item()}_layer_{layer_idx}_n_components_{n_components}_{suffix}.png"))


def visualize_and_save_mean_features_pca(feats_map, t, save_dir, layer_idx, n_components=3, suffix=""):
    """
    feats_map: [B, N, D],query:[8, 1024, 80] cuda tensor
    t: timestep,int
    save_dir: pic save dir
    layer_idx: current layer idx
    grid_size: the size of the grid to combine the images (rows, columns)
    """
    if len(feats_map.shape) == 3:  # [B,N,D]格式
        feats_map = torch.mean(feats_map, dim=0) # (4096,64)
        # feats_map = feats_map.flatten(0, -2)  # [8*256,160]
        feats_map = feats_map.detach().cpu().numpy()  #
        batch_size = 1
    if len(feats_map.shape) == 4:
        batch_size = feats_map.shape[0]
        feats_map = torch.mean(feats_map, dim=1)  # (2,256,160)
        feats_map = feats_map.flatten(0, -2) # (2*256,160)
        feats_map = feats_map.detach().cpu().numpy()  #

    # Apply PCA
    pca = PCA(n_components=n_components)  # To keep 3 components for RGB -> [8*256,3]
    pca.fit(feats_map)
    feature_maps_pca = pca.transform(feats_map)  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(batch_size,-1, 3)  # (H * W) x 3
    for i in range(batch_size):
        # Generate PCA images and add to list
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(np.sqrt(pca_img.shape[0]))  # Assuming square image
        pca_img = pca_img.reshape(h, w, 3)
        # Normalize the PCA image to [0, 1] range
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        # Convert to uint8 image
        combined_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        # Save the final combined image
        if not suffix:
            combined_img.save(
                os.path.join(save_dir, f"time_{t}_layer_{layer_idx}_n_components_{n_components}_index{i}.png"))
        else:
            combined_img.save(
                os.path.join(save_dir, f"time_{t}_layer_{layer_idx}_n_components_{n_components}_{suffix}_index{i}.png"))


def visualize_and_save_features_pca(feats_map, t, save_dir, layer_idx,n_components=3, suffix=""):
    """
    feats_map: [B, N, D],query:[8, 1024, 80] cuda tensor
    t: timestep,int
    save_dir: pic save dir
    layer_idx: current layer idx
    grid_size: the size of the grid to combine the images (rows, columns)
    """
    if len(feats_map.shape) == 3: # [B,N,D]格式
        B = len(feats_map) # [8,256,160]
        feats_map = feats_map.flatten(0, -2) # [8*256,160]
        feats_map = feats_map.detach().cpu().numpy() #

        # Apply PCA
        pca = PCA(n_components=n_components)  # To keep 3 components for RGB -> [8*256,3]
        pca.fit(feats_map)
        feature_maps_pca = pca.transform(feats_map)  # N X 3
        feature_maps_pca = feature_maps_pca.reshape(B, -1, 3)  # B x (H * W) x 3

        # List to hold individual PCA images
        pca_images = []

        # Generate PCA images and add to list
        for i, experiment in enumerate(feature_maps_pca):
            pca_img = feature_maps_pca[i]  # (H * W) x 3
            h = w = int(np.sqrt(pca_img.shape[0]))  # Assuming square image
            pca_img = pca_img.reshape(h, w, 3)

            # Normalize the PCA image to [0, 1] range
            pca_img_min = pca_img.min(axis=(0, 1))
            pca_img_max = pca_img.max(axis=(0, 1))
            pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)

            # Convert to uint8 image
            pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))

            # Append to list
            pca_images.append(pca_img)

        # # Combine all PCA images into one large grid
        # grid_rows, grid_cols = 1,n_components
        # assert grid_rows * grid_cols >= B, "Grid size is too small to fit all images."
        grid_cols = 1
        grid_rows = B

        # Get the size of a single image
        img_width, img_height = pca_images[0].size

        # Create a blank image for the grid
        combined_img = Image.new('RGB', (img_width * grid_cols, img_height * grid_rows))

        # Paste each PCA image into the correct position in the grid
        for idx, img in enumerate(pca_images):
            row, col = divmod(idx, grid_cols)
            combined_img.paste(img, (col * img_width, row * img_height))

        # Save the final combined image
        if not suffix:
            combined_img.save(os.path.join(save_dir, f"combined_time_{t}_layer_{layer_idx}_n_components_{n_components}.png"))
        else:
            combined_img.save(os.path.join(save_dir, f"combined_time_{t}_layer_{layer_idx}_n_components_{n_components}_{suffix}.png"))
### 输入均为numpy格式 ###
def grid_show(to_shows, cols,save_path):
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows * 8.5, cols * 2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # 关闭当前图像，避免显示



def visualize_head(att_map, save_path):
    '''attn_map:(197,197)'''
    # 将 tensor 转为 NumPy 数组并转移到 CPU
    att_map = att_map.detach().cpu().numpy()

    # 创建一个新的图形
    plt.figure(figsize=(8, 8))  # 设置图形大小，可以根据需要调整

    # 绘制热图
    im = plt.imshow(att_map, cmap='viridis')  # 选择合适的颜色映射

    # 创建 colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Attention Value')  # 设置 colorbar 标签

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # 清除当前图形，以便下次绘制不叠加
    plt.clf()


def visualize_heads(att_map, cols):
    '''attn_map:(1,6,197,197)'''
    att_map = att_map.detach().cpu().numpy()
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def gray2rgb(image):
    return np.repeat(image[..., np.newaxis], 3, 2)


def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W / 4), int(delta_H / 4)), 'CLS', fill=(0, 0, 0))  # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)
    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0: delta_W, :] = 1

    return padded_image, padded_mask, meta_mask


def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]

    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    padded_image, padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)

    if grid_index != 0:  # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index - 1) // grid_size[1]

    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1] + 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')


def visualize_grid_to_grid(att_map, grid_index, image,save_path, grid_size=14, alpha=0.6):
    '''
    att_map:(196,196) heatmap
    '''
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    H, W = att_map.shape
    with_cls_token = False

    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0],
                                       grid_size[1])  # att_map[grid_index]:对应索引的query在这个点的attention map响应,(14,14)
    mask = Image.fromarray(mask).resize((image.size))  # resize为正常图像大小

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # 关闭当前图像，避免显示


def visualize_grid_to_grid_normal(x,y,att_map, image,save_path,postfix, alpha=0.6):
    '''
    x,y: 在(512,512)图像上面的坐标
    调用:visualize_grid_to_grid_normal(att_map, grid_index, image,save_path)
    att_map:(4096,4096),cuda tensor
    image:pil格式读取的
    '''

    att_map = att_map.cpu().numpy()
    grid_size = int(att_map.shape[0] ** 0.5)
    grid_index = on_click(x, y, grid=grid_size)
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    H, W = att_map.shape
    with_cls_token = False

    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0],grid_size[1])  # att_map[grid_index]:对应索引的query在这个点的attention map响应,(14,14)
    mask = Image.fromarray(mask).resize((image.size))  # resize为正常图像大小

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    # ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].imshow(mask / np.max(mask))
    ax[1].axis('off')
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # 保存图像
    plt.savefig(os.path.join(save_path,f'{postfix}_index{grid_index}.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # 关闭当前图像，避免显示
    # 保存单独的注意力图
    plt.figure()
    plt.imshow(mask / np.max(mask), cmap='rainbow')
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f'{postfix}_mask_index{grid_index}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()


def highlight_grid(image, grid_indexes, grid_size=14):
    '''这个函数的作用是根据指定的网格索引，在图像上高亮显示该网格区域。显示一个红色的框'''
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image
if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained("StableDiffusion_Models/stable-diffusion-v1-5").to(device)
    tokenizer = ldm_stable.tokenizer
    controller = AttentionStore()
    show_cross_attention(tokenizer,["an apple"],"Codes/cross-image-attention/outputs_debug/attentions/cross_attn.png",controller, res=16, from_where=("up", "down"))
    # show_self_attention("Codes/cross-image-attention/outputs_debug/attentions/self_attn.png",controller, res=16, from_where=("up", "down"))