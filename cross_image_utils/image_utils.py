import pathlib
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from config import RunConfig
import os
from utils.utils import glob_frame_paths
from  cross_image_utils.edge_extraction import load_edge

def load_video_images(app_image_path,struct_dir,struct_save_dir,edge_method='') -> Tuple[Image.Image, Image.Image]:
    '''
    return numpy array,size:(512,512）
    struct_dir:原始图片
    struct_save_dir:edge存放图片
    edge_method: 只在使用edge的时候传入
    '''
    image_style = load_size(app_image_path)
    image_struct_list= []
    if edge_method:
        frame_paths = glob_frame_paths(struct_dir)  # struct当中全部图片
        for frame_path in frame_paths:
            save_path = os.path.join(struct_save_dir, os.path.basename(frame_path))
            if os.path.exists(save_path):  # edge已经存在
                image_struct = load_size(save_path)
            else:
                _ = load_edge(frame_path, device="cpu", method=edge_method, thresholds=[0.55], save_path=save_path) # return tensor
                image_struct = load_size(save_path)
            image_struct_list.append(image_struct)
    else:
        for image_name in sorted(os.listdir(struct_dir)):
            if image_name.endswith(".png") or image_name.endswith(".jpg"):
                struct_image_path = os.path.join(struct_dir,image_name)
                image_struct = load_size(struct_image_path)
                image_struct_list.append(image_struct)
    # if save_path is not None:
    #     Image.fromarray(image_style).save(save_path / f"in_style.png")
    #     Image.fromarray(image_struct).save(save_path / f"in_struct.png")
    return image_style, image_struct_list
def load_images(cfg: RunConfig, save_path: Optional[pathlib.Path] = None) -> Tuple[Image.Image, Image.Image]:
    image_style = load_size(cfg.app_image_path)
    image_struct = load_size(cfg.struct_image_path)
    if save_path is not None:
        Image.fromarray(image_style).save(save_path / f"in_style.png")
        Image.fromarray(image_struct).save(save_path / f"in_struct.png")
    return image_style, image_struct


def load_size(image_path: pathlib.Path,
              left: int = 0,
              right: int = 0,
              top: int = 0,
              bottom: int = 0,
              size: int = 512) -> Image.Image:
    if isinstance(image_path, (str, pathlib.Path)):
        image = np.array(Image.open(str(image_path)).convert('RGB'))  
    else:
        image = image_path

    h, w, _ = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]

    h, w, c = image.shape

    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    image = np.array(Image.fromarray(image).resize((size, size)))
    return image


def save_generated_masks(model, cfg: RunConfig):
    tensor2im(model.image_app_mask_32).save(cfg.output_path / f"mask_style_32.png")
    tensor2im(model.image_struct_mask_32).save(cfg.output_path / f"mask_struct_32.png")
    tensor2im(model.image_app_mask_64).save(cfg.output_path / f"mask_style_64.png")
    tensor2im(model.image_struct_mask_64).save(cfg.output_path / f"mask_struct_64.png")


def tensor2im(x) -> Image.Image:
    return Image.fromarray(x.cpu().numpy().astype(np.uint8) * 255)