import os
import cv2
import numpy as np
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers import DDIMScheduler, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import torchvision.transforms.functional as F
import torch
import torchvision
from torchvision import transforms
from cross_image_utils.CSD_Score.model import CSD_CLIP, convert_state_dict


style_image_dir = "/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance-flare/imgs_crop_fore/00009.jpg"
content_image_dir = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/1.5_2matching_guidance_1start_time51end_time301/recon_result/0000.png"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# init style clip model
clip_model = CSD_CLIP("vit_large", "default", model_path="/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/cross_image_utils/CSD_Score/models/ViT-L-14.pt")
model_path = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/cross_image_utils/CSD_Score/models/checkpoint.pth"
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = convert_state_dict(checkpoint['model_state_dict'])
clip_model.load_state_dict(state_dict, strict=False)
clip_model = clip_model.to(device)
def spherical_dist_loss(x, y):
    return -x@y.T

def tensor_process(image_path):
    '''
    input:path or tensor
    inout_tensor:[1,3,512,512] cuda
    '''
    if type(image_path) == str:
        pil_image = Image.open(image_path)#.convert("L")
        input_tensor = transforms.ToTensor()(pil_image).unsqueeze(0).to("cuda")
    elif isinstance(image_path, Image.Image):
        # 转换为张量并移动到 GPU
        input_tensor = transforms.ToTensor()(image_path).unsqueeze(0).to("cuda")
    elif isinstance(image_path, np.ndarray):
        # 将 numpy 数组转为 tensor，并调整维度
        input_tensor = torch.from_numpy(image_path).float() / 255.0  # 归一化到 [0, 1]
        if input_tensor.ndimension() == 2:  # 如果是灰度图
            input_tensor = input_tensor.unsqueeze(0)  # 添加 channel 维度
        else:  # 如果是 RGB 图像
            input_tensor = input_tensor.permute(2, 0, 1)  # 调整维度为 [C, H, W]
        input_tensor = input_tensor.unsqueeze(0).to("cuda")  # 扩展批处理维度并移动到 CUDA
        return input_tensor
    else:
        input_tensor = image_path

    # normalize = transforms.Normalize((0.5), (0.5)) # normalize表示每个通道的均值和方差
    # return normalize(transforms.Resize(224)(input_tensor)).repeat(1,3,1,1)
    normalize = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # normalize表示每个通道的均值和方差
    return normalize(transforms.Resize(224)(input_tensor))
if __name__ == "__main__":
    # computer style embedding
    style_image_ = tensor_process(style_image_dir).to(device) # torch.Size([1, 3, 224, 224])
    with torch.no_grad():
        _, content1, style1 = clip_model(style_image_)

    # computer content embedding
    content_image_ = tensor_process(content_image_dir).to(device) # torch.Size([1, 3, 224, 224])
    with torch.no_grad():
        _, content2, style2 = clip_model(content_image_)


    print(spherical_dist_loss(content1,content2),spherical_dist_loss(style1,style2))
    # print(content_output.shape,style_output.shape) # torch.Size([1, 768]) torch.Size([1, 768])
    # _, content_output, image_embeddings_clip = self.clip_model(self.normalize((image[0:1])))