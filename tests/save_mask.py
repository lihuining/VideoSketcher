import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import  Image

# 读取图像并转换为灰度图像
def read_image(path, grayscale=True,size=(512,512)):
    # image = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    image = Image.open(path).convert('L')
    image = image.resize(size)

    resized_image = image.resize(size)
    return np.array(resized_image)


# 生成二值化的前景掩码
def generate_mask(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY,
                  blockSize=11, C=2):
    mask = cv2.adaptiveThreshold(image, maxValue=maxValue, adaptiveMethod=adaptiveMethod, thresholdType=thresholdType,
                                 blockSize=blockSize, C=C)
    return mask


# 将图像转换为张量
def image_to_tensor(image, device='cpu'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(image).to(device)
    return tensor


# 计算均值和标准差
def calc_mean_std(feat, mask=None):
    if mask is not None:
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(feat.device)
        feat = feat * mask
        num_elements = mask.sum()
    else:
        num_elements = feat.numel() / feat.size(0)

    mean = feat.view(feat.size(0), -1).sum(1) / num_elements
    std = torch.sqrt((feat.view(feat.size(0), -1).pow(2).sum(1) / num_elements) - mean.pow(2))
    return mean, std


# AdaIN
def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.unsqueeze(-1).unsqueeze(-1)) / content_std.unsqueeze(-1).unsqueeze(
        -1)
    return normalized_feat * style_std.unsqueeze(-1).unsqueeze(-1) + style_mean.unsqueeze(-1).unsqueeze(-1)


# Masked AdaIN
def masked_adain(content_feat, style_feat, content_mask, style_mask):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    c, h, w = content_feat.shape
    h_mask, w_mask = content_mask.shape
    if h != h_mask or w != w_mask:
        content_mask = cv2.resize(content_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        style_mask = cv2.resize(style_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat, mask=style_mask)
    content_mean, content_std = calc_mean_std(content_feat, mask=content_mask)
    normalized_feat = (content_feat - content_mean.unsqueeze(-1).unsqueeze(-1)) / content_std.unsqueeze(-1).unsqueeze(
        -1)
    style_normalized_feat = normalized_feat * style_std.unsqueeze(-1).unsqueeze(-1) + style_mean.unsqueeze(
        -1).unsqueeze(-1)
    return content_feat * (1 - torch.from_numpy(content_mask).unsqueeze(0).unsqueeze(0).float().to(
        content_feat.device)) + style_normalized_feat * torch.from_numpy(content_mask).unsqueeze(0).unsqueeze(
        0).float().to(content_feat.device)


# 读取内容和风格图像
content_image_path = '/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/2.1_2_1/recon_result/0000.png'
style_image_path = '/media/allenyljiang/5234E69834E67DFB/Dataset/Sketch_dataset/ref2sketch_yr/ref/ref0021.jpg'

content_image = read_image(content_image_path)
style_image = read_image(style_image_path)

# 生成掩码
content_mask = generate_mask(content_image)
style_mask = generate_mask(style_image)
cv2.imwrite("content_mask.png",content_mask)
cv2.imwrite("style_mask.png",style_mask)
# 将图像和掩码转换为张量
content_tensor = image_to_tensor(content_image)
style_tensor = image_to_tensor(style_image)

# 调用 AdaIN 和 Masked AdaIN
adain_result = adain(content_tensor, style_tensor)
masked_adain_result = masked_adain(content_tensor, style_tensor, content_mask, style_mask)


# 将结果转换回图像
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach().numpy()
    tensor = (tensor * 0.5) + 0.5  # 反归一化
    image = (tensor * 255).astype(np.uint8)
    return image


adain_image = tensor_to_image(adain_result)
masked_adain_image = tensor_to_image(masked_adain_result)
print("adain image",adain_image.shape)
print("masked adain image",masked_adain_image.shape)
# 保存结果
cv2.imwrite('adain_result.png',
            adain_image)
cv2.imwrite('masked_adain_result.png',
            masked_adain_image[0])
