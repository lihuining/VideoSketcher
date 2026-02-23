import os.path

from PIL import Image

import numpy as np
import torch
from sympy.codegen.cnodes import struct
from torchvision import transforms
from skimage.transform import resize

from cross_image_utils.u2net import U2NET
from pathlib import Path
class BinaryPreprocessor:
    def __init__(self, bin_threshold=100, resolution=512):
        self.bin_threshold = bin_threshold
        self.resolution = resolution

    def execute(self, image):
        # Resize image
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        # Convert to grayscale
        image = image.convert("L")
        # Apply binary threshold
        image_np = np.array(image)
        image_np = np.where(image_np > self.bin_threshold, 255, 0).astype(np.uint8)
        # Convert back to PIL Image
        image = Image.fromarray(image_np)
        return image
def get_mask_u2net(pil_im, output_dir, u2net_path,name, device="cpu"):
    '''
    pil_im:pil image
    torch.from_numpy(mask).to("cuda")
    '''
    output_dir = Path(output_dir)
    # input preprocess
    w, h = pil_im.size[0], pil_im.size[1]
    im_size = min(w, h)
    if pil_im.mode != 'RGB':
        pil_im = pil_im.convert("RGB")
    data_transforms = transforms.Compose([
        transforms.Resize(min(320, im_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    input_im_trans = data_transforms(pil_im).unsqueeze(0).to(device)

    # load U^2 Net model
    net = U2NET(in_ch=3, out_ch=1)
    net.load_state_dict(torch.load(u2net_path))
    net.to(device)
    net.eval()

    # get mask
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.detach())
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    mask = torch.cat([predict, predict, predict], dim=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    mask = resize(mask, (h, w), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    # predict_np = predict.clone().cpu().data.numpy()
    im = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8)).convert('RGB')
    save_path_ = output_dir / f"{name}_mask.png"
    im.save(save_path_)

    im_np = np.array(pil_im)
    im_np = im_np / im_np.max()
    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    # free u2net
    del net
    torch.cuda.empty_cache()

    return im_final, predict
if __name__ == "__main__":

    # output_dir = "Codes/cross-image-attention/debug/u2net_mask"
    # image_path = "Codes/StyleID/data/style_4sketch_style/4sketch_style1.png"
    # name = os.path.basename(image_path).split('.')[0]
    # pil_im = Image.open(image_path)
    # u2net_path = "Codes/DiffSketcher/checkpoint/u2net/u2net.pth"
    # im_final, predict = get_mask_u2net(pil_im, output_dir, u2net_path,name, device="cpu")
    # bin_threshold = 100
    for bin_threshold in range(150,210,10):
        struct_path = "Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance-flare/imgs_crop_fore/00000.jpg"
        pil_im = Image.open(struct_path)
        preprocessor = BinaryPreprocessor(bin_threshold=bin_threshold)
        bin_image = preprocessor.execute(pil_im)  # bin_image is a binary image (2D grayscale)
        bin_dir = "Codes/cross-image-attention/debug/binarization"
        os.makedirs(bin_dir,exist_ok=True)
        bin_image.save(Path(bin_dir)/f'bin_{bin_threshold}_crop.png')