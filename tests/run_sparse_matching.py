import argparse
import os
from os.path import join

import cv2
import torch
from matplotlib import pyplot as plt
from numpy.ma.core import reshape

from cross_image_utils.gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from cross_image_utils.gluestick.drawing import plot_images, plot_lines, plot_color_line_matches, plot_keypoints, plot_matches
from cross_image_utils.gluestick.models.two_view_pipeline import TwoViewPipeline

from cross_image_utils.sketch_loss_utils import compute_sketch_matching_loss
def main():
    # Parse input parameters
    parser = argparse.ArgumentParser(
        prog='GlueStick Demo',
        description='Demo app to show the point and line matches obtained by GlueStick')
    parser.add_argument('-img1', default=join('resources' + os.path.sep + 'img1.jpg'))
    parser.add_argument('-img2', default=join('resources' + os.path.sep + 'img2.jpg'))
    parser.add_argument('--max_pts', type=int, default=1000)
    parser.add_argument('--max_lines', type=int, default=300)
    parser.add_argument('--skip-imshow', default=False, action='store_true')
    args = parser.parse_args()

    # Evaluation config
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': args.max_pts,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': args.max_lines,
        },
        'matcher': {
            'name': 'gluestick',
            # 'weights': str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar'),
            'weights': "/media/allenyljiang/564AFA804AFA5BE51/Codes/sparse_matching/GlueStick/resources/weights/checkpoint_GlueStick_MD.tar",
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline_model = TwoViewPipeline(conf).to(device).eval()

    gray0 = cv2.imread(args.img1, 0) # 0表示以灰度模式加载图像
    gray1 = cv2.imread(args.img2, 0)

    # # 调整大小
    # gray0 = cv2.resize(gray0, (512, 512))  # 将图像调整为 512x512
    # gray1 = cv2.resize(gray1, (512, 512))

    torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1) # (1,512,512)
    torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None] # (1,1,512,512)
    x = {'image0': torch_gray0, 'image1': torch_gray1} # （1，1，512，512）
    pred = pipeline_model(x)

    pred = batch_to_np(pred)
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]

    line_seg0, line_seg1 = pred["lines0"], pred["lines1"] # (16,2,2) (23,2,2)
    line_matches = pred["line_matches0"]

    valid_matches = m0 != -1
    match_indices = m0[valid_matches]
    matched_kps0 = kp0[valid_matches]
    matched_kps1 = kp1[match_indices]

    valid_matches = line_matches != -1
    match_indices = line_matches[valid_matches]
    matched_lines0 = line_seg0[valid_matches]
    matched_lines1 = line_seg1[match_indices]

    concat = os.path.basename(args.img1).split('.')[0] + os.path.basename(args.img2).split('.')[0]
    # Plot the matches
    img0, img1 = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR) # (1080,1080,3)
    plot_images([img0, img1], ['Image 1 - detected lines', 'Image 2 - detected lines'], dpi=200, pad=2.0)
    plot_lines([line_seg0, line_seg1], ps=4, lw=2) # (181,2,2)
    plt.gcf().canvas.manager.set_window_title('Detected Lines')
    plt.savefig(f'{concat}_detected_lines.png')

    plot_images([img0, img1], ['Image 1 - detected points', 'Image 2 - detected points'], dpi=200, pad=2.0)
    plot_keypoints([kp0, kp1], colors='c')
    plt.gcf().canvas.manager.set_window_title('Detected Points')
    plt.savefig(f'{concat}_detected_points.png')

    plot_images([img0, img1], ['Image 1 - line matches', 'Image 2 - line matches'], dpi=200, pad=2.0)
    plot_color_line_matches([matched_lines0, matched_lines1], lw=2) # (3,2,2) (3,2,2)
    plt.gcf().canvas.manager.set_window_title('Line Matches')
    plt.savefig(f'{concat}_line_matches.png')

    plot_images([img0, img1], ['Image 1 - point matches', 'Image 2 - point matches'], dpi=200, pad=2.0)
    plot_matches(matched_kps0, matched_kps1, 'green', lw=1, ps=0) # (163,2) (163,2)
    plt.gcf().canvas.manager.set_window_title('Point Matches')
    plt.savefig(f'{concat}_point_matches.png')
    # if not args.skip_imshow:
    #     plt.show()
    # sketch1 = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/1_prev_frame/generated_result/0000.png"
    # sketch2 = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/1_prev_frame/generated_result/0001.png"
    # # gray0 = cv2.imread(sketch1, 0) # 0表示以灰度模式加载图像
    # # gray1 = cv2.imread(sketch2, 0)
    # cur_loss = compute_sketch_matching_loss(gray0, gray1, [matched_lines0, matched_lines1], [matched_kps0, matched_kps1])
    # print(cur_loss) # tensor(21729.7539) 相邻帧的sketch
    # 同一张图 -- 0
    # 彩色图相邻两帧 -- tensor(877.4590)
    # sketch 相邻两帧 -- tensor(7616.9111)
    # 使用彩色图匹配结果用于计算sketch的损失 -- tensor(4521.0581)



if __name__ == '__main__':
    main()
'''
python -m gluestick.run -img1 resources/img1.jpg -img2 resources/img2.jpg
python -m gluestick.run -img1 /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/generated_result_3_cross_frame_masked_adain/0000.png -img2 /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/generated_result_3_cross_frame_masked_adain/0001.png
python -m gluestick.run -img1 /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug/edge_results/00001_teed_0.55.png -img2 /media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/debug/edge_results/00000_teed.png
# 彩色图相邻两帧
python -m gluestick.run -img1 /media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance-flare/imgs_crop_fore/00000.jpg -img2 /media/allenyljiang/5234E69834E67DFB/Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/breakdance-flare/imgs_crop_fore/00001.jpg
-img1
/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/1_prev_frame/generated_result/0000.png
-img2
/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/outputs/breakdance-flare/ref0020/1_prev_frame/generated_result/0001.png
'''