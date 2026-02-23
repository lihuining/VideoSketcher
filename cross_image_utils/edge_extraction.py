'''
method:hed,teed,canny,lineart
'''
import os.path

import torch
import sys
from torch.utils.tensorboard.summary import image

sys.path.append('../')
from TEED.ted import TED
from cross_image_utils.edge_networks import HED,weight_init
from controlnet_aux import LineartDetector

import numpy as np

from PIL import Image
from PIL import ImageOps
import torchvision.transforms as transforms
from TEED.ted import TED
import cv2
from controlnet_aux import LineartDetector
def initialize_ted(device):
    model = TED()
    checkpoint_path = 'Codes/cross-image-attention/TEED/checkpoints/BIPED/7/7_model.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model

def initialize_hed(device):
    hed_model = HED()
    hed_model.apply(weight_init)
    hed_checkpoint = torch.load('Codes/mixsa/checkpoint/network-bsds500.pytorch', map_location=device)
    hed_model.load_state_dict(hed_checkpoint)
    hed_model = hed_model.to(device)
    hed_model.eval()
    return hed_model

# 初始化 LineartDetector
ctrn_processor_path = "lllyasviel/Annotators"
def initialize_lineart_detector(ctrn_processor_path):
    return LineartDetector.from_pretrained(ctrn_processor_path)


# def load_edge(path, device, method='teed', alpha=0.55, save_path=None):
#     image = Image.open(path).convert("RGB")
#     x, y = image.size
#     max_dim = max(x, y)
#     new_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
#     new_image.paste(image, ((max_dim - x) // 2, (max_dim - y) // 2))
#
#     # Resize to 512x512
#     h = w = 512
#     image = new_image.resize((w, h), resample=Image.Resampling.LANCZOS)
#
#     if method == 'hed':
#         e_net = initialize_hed(device)
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         image_tensor = transform(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             edges = e_net(image_tensor)
#         edges = torch.sigmoid(edges[0]).cpu()
#         edges = (edges > alpha).float()
#         edges = 1 - edges
#         edges_pil = transforms.ToPILImage()(edges).convert("L")
#
#     elif method == 'teed':
#         e_net = initialize_ted(device)
#         #image = Image.open(path).convert("RGB")
#
#         width, height = image.size
#         new_width = (width + 7) // 8 * 8
#         new_height = (height + 7) // 8 * 8
#         if new_width != width or new_height != height:
#             image = image.resize((new_width, new_height), Image.LANCZOS)
#         image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
#
#         mean_rgb = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1).to(device)
#         image_tensor = image_tensor * 255 - mean_rgb
#         image_tensor = image_tensor[:, [2, 1, 0], :, :]
#         with torch.no_grad():
#             preds = e_net(image_tensor)[-1][0, 0]
#             pred = torch.sigmoid(preds).cpu().numpy()
#             pred = (pred > alpha).astype(np.uint8)
#             pred = 1 - pred
#             edges_pil = Image.fromarray(pred * 255).convert('L')
#
#     elif method == 'canny':
#         image_np = np.array(image)
#         edges = cv2.Canny(image_np, 100, 200)
#         edges = 255 - edges  # Invert colors to match white background and black edges
#         edges_pil = Image.fromarray(edges).convert('L')
#
#     elif method == 'lineart':
#         processor = initialize_lineart_detector(ctrn_processor_path)
#         edges = processor(image)
#         if isinstance(edges, Image.Image):
#             edges_pil = edges.convert('L')
#             edges_pil = ImageOps.invert(edges_pil)
#         else:
#             edges_pil = Image.fromarray(edges).convert('L')
#             edges_pil = ImageOps.invert(edges_pil)
#
#     if save_path:
#        print(f"Saving edge image to: {save_path}")
#        edges_pil.save(save_path)
#
#     edge_tensor = transforms.ToTensor()(edges_pil)  # Convert PIL image to tensor
#     edge_tensor = edge_tensor.repeat(3, 1, 1)  # Repeat the single channel across 3 channels
# #    del e_net
# #    torch.cuda.empty_cache()
#     return edge_tensor.unsqueeze(0).to(device)
def load_edge(path, device, method='teed', save_path=None, thresholds=None):
    '''
    给定原始图像返回edge tensor
    '''
    image = Image.open(path).convert("RGB")
    x, y = image.size
    max_dim = max(x, y)
    new_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    new_image.paste(image, ((max_dim - x) // 2, (max_dim - y) // 2))

    # Resize to 512x512
    h = w = 512
    image = new_image.resize((w, h), resample=Image.Resampling.LANCZOS)

    if method == 'hed':
        e_net = initialize_hed(device)
        alpha = thresholds[0] if thresholds and len(thresholds) > 0 else 0.55  # 提取 alpha 参数
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            edges = e_net(image_tensor)
        edges = torch.sigmoid(edges[0]).cpu()
        edges = (edges > alpha).float()
        edges = 1 - edges
        edges_pil = transforms.ToPILImage()(edges).convert("L")

    elif method == 'teed':
        e_net = initialize_ted(device)
        alpha = thresholds[0] if thresholds and len(thresholds) > 0 else 0.55  # 提取 alpha 参数

        width, height = image.size
        new_width = (width + 7) // 8 * 8
        new_height = (height + 7) // 8 * 8
        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height), Image.LANCZOS)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        mean_rgb = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1).to(device)
        image_tensor = image_tensor * 255 - mean_rgb
        image_tensor = image_tensor[:, [2, 1, 0], :, :]
        with torch.no_grad():
            preds = e_net(image_tensor)[-1][0, 0]
            pred = torch.sigmoid(preds).cpu().numpy()
            pred = (pred > alpha).astype(np.uint8)
            pred = 1 - pred
            edges_pil = Image.fromarray(pred * 255).convert('L')

    elif method == 'canny':
        low_threshold = thresholds[1] if thresholds and len(thresholds) > 1 else 50  # 提取低阈值
        high_threshold = thresholds[2] if thresholds and len(thresholds) > 2 else 100  # 提取高阈值
        image_np = np.array(image)
        edges = cv2.Canny(image_np, low_threshold, high_threshold)
        edges = 255 - edges  # Invert colors to match white background and black edges
        edges_pil = Image.fromarray(edges).convert('L')

    elif method == 'lineart':
        processor = initialize_lineart_detector(ctrn_processor_path)
        edges = processor(image)
        if isinstance(edges, Image.Image):
            edges_pil = edges.convert('L')
            edges_pil = ImageOps.invert(edges_pil)
        else:
            edges_pil = Image.fromarray(edges).convert('L')
            edges_pil = ImageOps.invert(edges_pil)

    if save_path:
       # print(f"Saving edge image to: {save_path}")
       edges_pil.save(save_path)

    edge_tensor = transforms.ToTensor()(edges_pil)  # Convert PIL image to tensor
    edge_tensor = edge_tensor.repeat(3, 1, 1)  # Repeat the single channel across 3 channels
    return edge_tensor.unsqueeze(0).to(device)


if __name__ == "__main__":
    image_path = "Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/camel/imgs_crop_fore/00002.jpg"
    class_name = image_path.split('/')[-3]
    save_dir = "Codes/cross-image-attention/debug/edge_results"
    name = os.path.basename(image_path).split('.')[0]
    method = "teed"
    alpha = 0.55
    # method_list = ["teed","hed","canny"]
    # for method in method_list:
    # for alpha in np.arange(0.1, 1.0, 0.1): # range不支持浮点数
    save_path = os.path.join(save_dir, f"{class_name}_{name}_{method}_{alpha}.png")
    load_edge(image_path, device="cpu", method=method, thresholds=[alpha], save_path=save_path)

    # method = "canny"
    # start = 50
    # for j in range(100,200,20):
    #     save_path = os.path.join(save_dir, f"{name}_{method}_{start}_{j}.png")
    #     load_edge(image_path, device="cuda", method=method, thresholds=[start,j], save_path=save_path)
