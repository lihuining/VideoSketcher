import os
import shutil
#
# def copy_images_from_paths(file_path, target_dir):
#     # 读取txt文件中的路径
#     with open(file_path, 'r') as file:
#         paths = [line.strip() for line in file if line.strip()]
#
#     # 确保目标目录存在
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     for path in paths:
#         # 获取结构名称和参考图片名称
#         parts = path.split('/')
#         struct_name = parts[-4]
#         ref_name = parts[-3]
#
#         # 创建目标子文件夹
#         dest_subfolder = os.path.join(target_dir, struct_name,ref_name)
#         if not os.path.exists(dest_subfolder):
#             os.makedirs(dest_subfolder)
#
#         # 复制图片文件
#         for img_file in os.listdir(path):
#             if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                 src_img_path = os.path.join(path, img_file)
#                 dest_img_path = os.path.join(dest_subfolder, img_file)
#                 shutil.copy(src_img_path, dest_img_path)
#                 print(f"Copied {src_img_path} to {dest_img_path}")
#
# # 使用示例
# file_path = 'Codes/cross-image-attention/experiments/style_results/1109.txt'
# target_dir = '/media/allenyljiang/2CD8318DD83155F4/CVPR2025/ours'
# copy_images_from_paths(file_path, target_dir)


## copy mp4 ##
mp4_data_dir="/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Rebuttal/new_results"
#mp4_data_dir="/media/allenyljiang/2CD8318DD83155F4/CVPR2025"
os.makedirs(mp4_data_dir,exist_ok=True)

#mp_video_list = ['Codes/cross-image-attention/experiments/generated_video_list/video_sketcher_1109_additional_mp4.txt']
#mp_video_list = ['/home/allenyljiang/Desktop/CVPR25/ipadapter/output/ipadapter_1109_all_mp4.txt','/home/allenyljiang/Desktop/CVPR25/semi_ref2sketch/semi_ref2sketch_1109_all_mp4.txt','/home/allenyljiang/Desktop/CVPR25/ref2sketch/ref2sketch_1109_all_mp4.txt','/home/allenyljiang/Desktop/CVPR25/Style_ID/output/styleid_1109_all_mp4.txt','Codes/cross-image-attention/experiments/generated_video_list/videosketcher_1119_generated_video_list_mp4.txt','Codes/cross-image-attention/experiments/generated_video_list/video_sketcher_1109_mp4.txt','/home/allenyljiang/Desktop/CVPR25/Cross_image_attention/cross_image_attention_1109_all_mp4.txt']
#mp_video_list = ['Codes/cross-image-attention/experiments/comparison_with_bessel/sketch_video_synthesis_1121_mp4.txt','Codes/cross-image-attention/experiments/comparison_with_bessel/video_sketcher_bessel_1121_mp4.txt']
#mp_video_list = ['Codes/cross-image-attention/experiments/sketch_video_synthesis_mp4.txt','Codes/cross-image-attention/experiments/bessel_style_video_mp4.txt']
mp_video_list = ['Codes/cross-image-attention/experiments/rebuttal/generated_list/kid_football.txt',
                 'Codes/cross-image-attention/experiments/rebuttal/generated_list/jump_dance_gamma0.6.txt',
                 'Codes/cross-image-attention/experiments/rebuttal/generated_list/libby.txt'
                 ]

# 遍历每个视频列表文件
for video_list_file in mp_video_list:
    # 获取文件名
    file_name = os.path.basename(video_list_file)

    # 以 '_' 划分文件名的第一个部分作为新的文件夹名称
    folder_name = file_name.split('_')[0]
    folder_path = os.path.join(mp4_data_dir, folder_name)

    # 创建文件夹
    os.makedirs(folder_path, exist_ok=True)

    # 读取视频列表文件
    with open(video_list_file, 'r') as f:
        video_paths = f.readlines()

    # 遍历每个视频路径
    for video_path in video_paths:
        video_path = video_path.strip()  # 去除末尾的换行符
        if not video_path:
            continue

        # 获取视频文件名
        video_name = os.path.basename(video_path)

        # 获取视频路径的倒数第4个和倒数第5个部分
        parts = video_path.split('/')
        new_video_name = f"{parts[-5]}_{parts[-4]}.mp4"

        # 目标路径
        target_path = os.path.join(folder_path, new_video_name)

        # 复制文件
        shutil.copy(video_path, target_path)
        print(f"Copied {video_path} to {target_path}")

print("All videos have been copied successfully.")



