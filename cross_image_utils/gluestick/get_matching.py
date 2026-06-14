import argparse
import os
from os.path import join
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from cross_image_utils.gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT,batch_to_single,batch_to_np_keep_dim
from cross_image_utils.gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from cross_image_utils.gluestick.models.two_view_pipeline import TwoViewPipeline
from cross_image_utils.sketch_loss_utils import compute_sketch_matching_loss
import torch
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union

_PIPELINE_MODEL = None
_PIPELINE_CACHE_KEY = None


def save_grayscale_tensor(tensor, save_path):
    # 检查张量的维度为 (1, 1, H, W)，如果维度不同则调整
    if tensor.dim() == 4 and tensor.size(1) == 1:
        # 将张量从 CUDA 转移到 CPU，转换为 NumPy 格式，并移除批次和通道维度
        tensor_np = tensor.squeeze().clone().detach().cpu().numpy()
    else:
        raise ValueError("输入张量应为形状 [1, 1, H, W]")

    # 将值缩放到 [0, 255] 范围内
    tensor_np = (tensor_np * 255).astype(np.uint8)

    # 使用 PIL 将 NumPy 数组保存为图像
    Image.fromarray(tensor_np, mode='L').save(save_path)



def _get_pipeline_model(device, max_pts=1000, max_lines=300):
    global _PIPELINE_MODEL, _PIPELINE_CACHE_KEY
    cache_key = (device, max_pts, max_lines)
    if _PIPELINE_MODEL is not None and _PIPELINE_CACHE_KEY == cache_key:
        return _PIPELINE_MODEL

    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': max_pts,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': max_lines,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }
    _PIPELINE_MODEL = TwoViewPipeline(conf).to(device).eval()
    _PIPELINE_CACHE_KEY = cache_key
    return _PIPELINE_MODEL


def get_sparse_matching_results(img1,img2,cal_img1: Optional[torch.FloatTensor] = None,cal_img2: Optional[torch.FloatTensor] = None,max_pts=1000,max_lines=300,timestep=0,save_dir="",chunk_index=0):

    '''
    img1,img2:tensor [1,3,1024,1024] on cuda
    实际使用的tensor:(1,1,1024,1024) on cuda
    '''
    # # 如果是彩色图像，将其转换为灰度
    # if len(img1.shape) == 3:
    #     img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    # if len(img2.shape) == 3:
    #     img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    tensor_list = [img1,img2,cal_img1,cal_img2]
    for idx,img in enumerate(tensor_list):
        if len(img.shape) == 3:
            tensor_list[idx] = img.unsqueeze(0)
    [img1, img2, cal_img1, cal_img2] = tensor_list
    device = 'cpu'
    img1_for_matching = img1.detach().to(device)
    img2_for_matching = img2.detach().to(device)
    pipeline_model = _get_pipeline_model(device, max_pts=max_pts, max_lines=max_lines)
    # print("matching pipeline requires_grad:", any(param.requires_grad for param in pipeline_model.parameters()))
    # gray0 = cv2.imread(args.img1, 0) # 0表示以灰度模式加载图像
    # gray1 = cv2.imread(args.img2, 0)
    # gray0 = img1
    # gray1 = img2
    # torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
    ### 彩色转化为灰度tensor

    # 假设 img 是 RGB 彩色张量，形状为 (N, 3, H, W)
    # 使用浮点权重来保持梯度
    weights = torch.tensor([0.299, 0.587, 0.114], device=device,requires_grad=False) # 变为 (3, 1, 1)

    # 计算加权和，保留梯度
    torch_gray0 = torch.tensordot(img1_for_matching, weights, dims=([1], [0])).unsqueeze(1)  # (N, 1, H, W)
    torch_gray1 = torch.tensordot(img2_for_matching, weights, dims=([1], [0])).unsqueeze(1)  # (N, 1, H, W)

    # torch_gray0, torch_gray1 = numpy_image_to_torch(img1), numpy_image_to_torch(img2)
    # torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
    x = {'image0': torch_gray0.clone().detach(), 'image1': torch_gray1.clone().detach()} # 不能干扰原始张量的梯度流,这里的结果之后需要转化为numpy
    #x = {'image0': torch_gray0, 'image1': torch_gray1} # 不能干扰原始张量的梯度流,这里的结果之后需要转化为numpy
    with torch.no_grad():
        pred = pipeline_model(x) # tensor

    # pred = batch_to_np(pred)
    pred = batch_to_single(pred)
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"] # [1,270,2] -> [270,2]
    m0 = pred["matches0"]

    line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
    line_matches = pred["line_matches0"]

    valid_matches = m0 != -1
    match_indices = m0[valid_matches]
    matched_kps0 = kp0[valid_matches]
    matched_kps1 = kp1[match_indices]

    valid_matches = line_matches != -1
    match_indices = line_matches[valid_matches]
    matched_lines0 = line_seg0[valid_matches]
    matched_lines1 = line_seg1[match_indices]

    # cur_loss = compute_sketch_matching_loss(img1, img2, [matched_lines0, matched_lines1], [matched_kps0, matched_kps1])
    # cur_loss = cur_loss.to(dtype=img1.dtype, device=img1.device)
    # grads = -torch.autograd.grad(cur_loss, img1, create_graph=True)[0]
    # print("loss and grad",cur_loss,torch.max(grads))
    #
    # perturbed_image1 = image1 + torch.randn_like(image1) * 0.01
    # loss_perturbed = compute_sketch_matching_loss(perturbed_image1, img2, [matched_lines0, matched_lines1], [matched_kps0, matched_kps1])
    #
    # print("损失变化量:", loss_perturbed - cur_loss)
    # grads = -torch.autograd.grad(loss_perturbed, img1, create_graph=True)[0]
    # print("loss and grad",loss_perturbed,torch.max(grads))
    if len(cal_img1) == 0 and len(cal_img2) == 0:
        cal_img1,cal_img2 = img1,img2
    if save_dir:
        os.makedirs(save_dir,exist_ok=True)
        post = 't'+str(timestep) + 'chunk_index_'+str(chunk_index)
        variable_list = [line_seg0, line_seg1,kp0, kp1,matched_lines0, matched_lines1,matched_kps0, matched_kps1]
        line_seg0, line_seg1,kp0, kp1,matched_lines0, matched_lines1,matched_kps0, matched_kps1 = batch_to_np_keep_dim(variable_list)
        # print("debug,img1 shape",cal_img1[0].shape)
        img0_color,img1_color = cal_img1[0].detach().cpu().numpy().transpose(1, 2, 0) ,cal_img2[0].detach().cpu().numpy().transpose(1, 2, 0)  # (3,512,512)
        plot_images([img0_color,img1_color], ['Image 1 - detected lines', 'Image 2 - detected lines'], dpi=200, pad=2.0)
        plot_lines([line_seg0, line_seg1], ps=4, lw=2)
        plt.gcf().canvas.manager.set_window_title('Detected Lines')
        plt.savefig(f'{save_dir}/detected_lines_{post}.png')

        plot_images([img0_color,img1_color], ['Image 1 - detected points', 'Image 2 - detected points'], dpi=200, pad=2.0)
        plot_keypoints([kp0, kp1], colors='c')
        plt.gcf().canvas.manager.set_window_title('Detected Points')
        plt.savefig(f'{save_dir}/detected_points_{post}.png')

        plot_images([img0_color,img1_color], ['Image 1 - line matches', 'Image 2 - line matches'], dpi=200, pad=2.0)
        plot_color_line_matches([matched_lines0, matched_lines1], lw=2)
        plt.gcf().canvas.manager.set_window_title('Line Matches')
        plt.savefig(f'{save_dir}/line_matches_{post}.png')

        plot_images([img0_color,img1_color], ['Image 1 - point matches', 'Image 2 - point matches'], dpi=200, pad=2.0)
        plot_matches(matched_kps0, matched_kps1, 'green', lw=1, ps=0)
        plt.gcf().canvas.manager.set_window_title('Point Matches')
        plt.savefig(f'{save_dir}/point_matches_{post}.png')
    # print("gradient info",img1.grad,img2.grad)


    cur_loss,line_loss,point_loss = compute_sketch_matching_loss(cal_img1, cal_img2, [matched_lines0, matched_lines1], [matched_kps0, matched_kps1])
    # print(cur_loss.item(),line_loss.item(),point_loss.item()) # tensor(21729.7539) 相邻帧的sketch
    return [matched_lines0,matched_lines1],[matched_kps0,matched_kps1]
    # if not args.skip_imshow:
    #     plt.show()


if __name__ == '__main__':
    # image1 = torch.rand((1, 3, 1024, 1024), requires_grad=True).to("cuda")
    # image2 = torch.rand((1, 3, 1024, 1024), requires_grad=True).to("cuda")
    # 加载图像并转换为张量
    # image1_path = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/generated_result_3_cross_frame_masked_adain/0000.png"
    # image2_path = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/generated_result_3_cross_frame_masked_adain/0001.png"
    #image1_path = "/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance-flare/imgs_crop_fore/00000.jpg"
    # image2_path = "/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance-flare/imgs_crop_fore/00001.jpg"
    # image1_path = "/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance-flare/imgs_crop_fore/00000.jpg"
    # image2_path = "/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance-flare/imgs_crop_fore/00000.jpg"
    # image1_path = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/generated_result_3_cross_frame_masked_adain/0000.png"
    #image2_path = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/generated_result_3_cross_frame_masked_adain/0000.png"
    # #image2_path = "/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance/imgs_crop_fore/00001.jpg"
    # image1_path = "/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance/imgs_crop_fore/00001.jpg"
    #
    #
    # for j in range(15,60,2):
    #     image1_path = f"/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance/imgs_crop_fore/000{j-5}.jpg"
    #     image2_path = f"/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance/imgs_crop_fore/000{j}.jpg"
    #     print(image1_path,image2_path)
    #     pil_image = Image.open(image1_path)  # 读取图像
    #     image1 = transforms.ToTensor()(pil_image).unsqueeze(0).to("cuda")  # 转换为 (C, H, W) 归一化张量
    #     image1.requires_grad = True
    #
    #     pil_image = Image.open(image2_path)  # 读取图像
    #     image2 = transforms.ToTensor()(pil_image).unsqueeze(0).to("cuda")  # 转换为 (C, H, W) 归一化张量
    #     image2.requires_grad = True
    #
    #     # print("gradient info", image1.grad, image2.grad)
    #     res1,res2 = get_sparse_matching_results(image1,image2)
    image1_path = f"/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/camel/imgs_crop_fore/00000.jpg"
    #image2_path = f"/media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/camel/imgs_crop_fore/00001.jpg"
    image2_path = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/camel/ref0001/2.1_chunk_size2_12/generated_result/0000.png"
    # image2_path = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/camel/ref0001/2.1_chunk_size2_12/generated_result/0001.png"
    print(image1_path, image2_path)
    pil_image = Image.open(image1_path)  # 读取图像
    image1 = transforms.ToTensor()(pil_image).unsqueeze(0).to("cuda")  # 转换为 (C, H, W) 归一化张量
    image1.requires_grad = True

    pil_image = Image.open(image2_path)  # 读取图像
    image2 = transforms.ToTensor()(pil_image).unsqueeze(0).to("cuda")  # 转换为 (C, H, W) 归一化张量
    image2.requires_grad = True

    # print("gradient info", image1.grad, image2.grad)
    res1, res2 = get_sparse_matching_results(image1, image2,save_dir="/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug/matching_vis/camel")
    '''
    两张相邻sketch loss:2.6099
    两张相邻彩色图 loss: tensor:1.0885
    彩色图与对应的sketch: tensor(3.2856, device='cuda:0', grad_fn=<AddBackward0>)
    相同的图 loss = 0
    两张毫无关系的彩色图: 2.9693
    '''
'''
python -m gluestick.run -img1 resources/img1.jpg -img2 resources/img2.jpg
python -m gluestick.run -img1 /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/generated_result_3_cross_frame_masked_adain/0000.png -img2 /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/generated_result_3_cross_frame_masked_adain/0001.png
python -m gluestick.run -img1 /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug/edge_results/00001_teed_0.55.png -img2 /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug/edge_results/00000_teed.png
'''
