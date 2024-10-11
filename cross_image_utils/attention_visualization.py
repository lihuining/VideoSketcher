'''
1. tokenizer definition  ldm.tokernizer??
2.

'''

from cross_image_utils.ddpm_inversion import AttentionStore
from PIL import Image
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import cross_image_utils.ptp_utils
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
    '''
    
    
    '''
    
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
def visualize_and_save_features_pca(feats_map, t, save_dir, layer_idx,suffix=""):
    """
    feats_map: [B, N, D],query
    t: timestep
    save_dir: pic save dir
    layer_idx: current layer idx
    """
    B = len(feats_map)
    feats_map = feats_map.flatten(0, -2)
    feats_map = feats_map.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feats_map)
    feature_maps_pca = pca.transform(feats_map)  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(B, -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(feature_maps_pca):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(np.sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        if not suffix:
            pca_img.save(os.path.join(save_dir, f"{i}_time_{t}_layer_{layer_idx}.png"))
        else:
            pca_img.save(os.path.join(save_dir, f"{i}_time_{t}_layer_{layer_idx}_{suffix}.png"))

if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained("/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stable-diffusion-v1-5").to(device)
    tokenizer = ldm_stable.tokenizer
    controller = AttentionStore()
    show_cross_attention(tokenizer,["an apple"],"/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs_debug/attentions/cross_attn.png",controller, res=16, from_where=("up", "down"))
    # show_self_attention("/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs_debug/attentions/self_attn.png",controller, res=16, from_where=("up", "down"))