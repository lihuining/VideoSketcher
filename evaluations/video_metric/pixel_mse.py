import os.path
import sys
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(__file__)))
from flow_utils import *
import os
import glob
from deps.gmflow.gmflow.gmflow import GMFlow
from deps.ControlNet.annotator.util import HWC3
import cv2
from PIL import Image
# Assume I1 and I2 are the two consecutive frames, and the edited results are O1 and O2.
flow_model = GMFlow(
    feature_channels=128,
    num_scales=1,
    upsample_factor=8,
    num_head=1,
    attention_type='swin',
    ffn_dim_expansion=4,
    num_transformer_layers=6,
).to('cuda')


def process_frame(image_path, h=512, w=512):
    # 读取图片
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 获取图片的高度和宽度
    fh, fw = frame.shape[:2]

    # 调整 h 和 w 为 64 的倍数
    h = int(np.floor(h / 64.0)) * 64
    w = int(np.floor(w / 64.0)) * 64

    # 计算新的宽度
    nw = int(fw / fh * h)
    if nw >= w:
        size = (nw, h)
    else:
        size = (w, int(fh / fw * w))

    #print(f"[INFO] frame size {(fh, fw)} resize to {size} and centercrop to {(w, h)}")

    # 缩放
    resized_frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

    # 中心裁剪
    ch, cw = resized_frame.shape[:2]
    top = (ch - h) // 2
    left = (cw - w) // 2
    cropped_frame = resized_frame[top:top + h, left:left + w]

    # # 转换为 CHW 格式
    # cropped_frame = np.transpose(cropped_frame, (2, 0, 1))

    return cropped_frame
def preprocess(image_path):
    frame = process_frame(image_path)
    # frame = cv2.resize(frame,(512,512))

    img = HWC3(frame)
    image2 = torch.from_numpy(img).permute(2, 0, 1).float().to("cuda")
    # 归一化到 [0, 1]
    image2 = image2 / 255.0
    # 归一化到 [-1, 1]
    image2 = (image2 - 0.5) * 2
    return image2
def save_tensor(input_tensor, name):
    # 1. 转换张量为 NumPy 格式
    bwd_occ_np = input_tensor.squeeze().detach().cpu().numpy()

    # 2. 检查并调整张量形状
    if bwd_occ_np.ndim == 3 and bwd_occ_np.shape[0] == 3:
        # 如果张量形状为 [3, 512, 512]，转置为 [512, 512, 3]
        bwd_occ_np = bwd_occ_np.transpose(1, 2, 0)
    elif bwd_occ_np.ndim == 2:
        # 如果张量形状为 [512, 512]，直接使用
        pass
    else:
        raise ValueError(f"不支持的张量形状: {input_tensor.shape}")

    # 3. 归一化张量值到 0-255 范围
    bwd_occ_np = (bwd_occ_np - bwd_occ_np.min()) / (bwd_occ_np.max() - bwd_occ_np.min()) * 255
    bwd_occ_np = bwd_occ_np.astype(np.uint8)

    # 4. 使用 PIL 保存图像
    image = Image.fromarray(bwd_occ_np)
    image.save(f'{name}.png')

    print(f"图像已保存为 {name}.png")


def get_sorted_image_names(directory):
    # 定义图片文件的扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

    # 使用 glob 模块查找所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))

    # 提取文件名并排序
    image_names = [os.path.basename(file) for file in image_files]
    image_names.sort()

    return image_names
def calculate_pixle_mse(video_dir,stylized_dir,n_frames,flow_model=''):
    if not flow_model:
        checkpoint = torch.load('/media/allenyljiang/564AFA804AFA5BE51/Codes/Video_Editing/Rerender_A_Video/models/gmflow_sintel-0c07dcb3.pth',
                                map_location=lambda storage, loc: storage)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        flow_model.load_state_dict(weights, strict=False)
        flow_model.eval()
    else:
        flow_model = flow_model

    cnt = 0
    total_err = 0
    edited_image_names=get_sorted_image_names(stylized_dir)

    ori_image_names = get_sorted_image_names(video_dir)
    for i in range(n_frames-1):
        # 生成文件名
        # file_name1 = f"{i:05d}.jpg"
        # file_name2 = f"{i + 1:05d}.jpg"
        file_name1 = ori_image_names[i]
        file_name2 = ori_image_names[i + 1]
        # 生成完整路径
        image_path1 = os.path.join(video_dir, file_name1)
        image_path2 = os.path.join(video_dir, file_name2)
        I1 = preprocess(image_path1)
        I2 = preprocess(image_path2)
        cnt += 1

        # edit_file_name1 = f"{i:04d}.png"
        # edit_file_name2 = f"{i+1:04d}.png"
        edit_file_name1 = edited_image_names[i]
        edit_file_name2 = edited_image_names[i+1]
        path1 = os.path.join(stylized_dir,edit_file_name1)
        path2 = os.path.join(stylized_dir,edit_file_name2)
        # I1 =preprocess(path1)
        # 使用原始的flow来warp编辑后的结果
        # I2 = preprocess(path2)
        O1 = preprocess(path1).unsqueeze(0).to("cuda")
        O2 = preprocess(path2).unsqueeze(0).to("cuda")
        warped_O1, mask, optical_flow =  get_warped_and_mask(flow_model,  I1,  I2, O1)
        cur_err = F.mse_loss(warped_O1*(1-mask), O2*(1-mask).to("cuda")) # 然后在整个视频上平均
        total_err += cur_err
        # print(cur_err)
    # print("total error",total_err,"average",total_err / cnt)
    return (total_err / cnt).item()

if __name__ == "__main__":
    video_dir="/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/cows/imgs_crop_fore"
    stylized_dir="/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/cows/2/2.1_chunk_size2_cross_frame/generated_result"
    print(calculate_pixle_mse(video_dir,stylized_dir,10,flow_model=flow_model))
# path1 = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/camel/ref0001/2.1_chunk_size1/generated_result/0000.png"
# path2 = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/camel/ref0001/2.1_chunk_size1/generated_result/0001.png"
# I1 =preprocess(path1)
# # 使用原始的flow来warp编辑后的结果
# I2 = preprocess(path2)
# O1 = I1.unsqueeze(0).to("cuda")
# print(I1.shape,I2.shape,O1.shape)
# warped_O1, mask, optical_flow =  get_warped_and_mask(flow_model,  I1,  I2, O1)
# print(warped_O1.shape,mask.shape,optical_flow.shape)
# # save_tensor(mask,'mask')
# # save_tensor(warped_O1,'warped_O1')
# # save_tensor(warped_O1*(1-mask),'warped_O1_mask')
# # save_tensor(I2*(1-mask),'I2_mask')
# err = F.mse_loss(warped_O1*(1-mask), I2*(1-mask).to("cuda")) # 然后在整个视频上平均
# print(err)


