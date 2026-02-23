import os
import json
import glob
from video_metric.frame_acc_tem_con import folder_consistency,folder_consistency_clip # CLIP一致性
from eval_artfid import compute_art_fid,compute_cfsd
import torch
import clip
from video_metric.deps.gmflow.gmflow.gmflow import GMFlow
import inception
'''
clip_metric = folder_consistency(folder)
'''
from video_metric.pixel_mse import calculate_pixle_mse
#video_list_path = "Codes/cross-image-attention/experiments/style_results/1109.txt"
# model loading
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

flow_checkpoint = torch.load(
    'Codes/Video_Editing/Rerender_A_Video/models/gmflow_sintel-0c07dcb3.pth',
    map_location=lambda storage, loc: storage)
weights = flow_checkpoint['model'] if 'model' in flow_checkpoint else flow_checkpoint
flow_model = GMFlow(
    feature_channels=128,
    num_scales=1,
    upsample_factor=8,
    num_head=1,
    attention_type='swin',
    ffn_dim_expansion=4,
    num_transformer_layers=6,
).to('cuda')
flow_model.load_state_dict(weights, strict=False)
flow_model.eval()

device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
art_inception_model_path = "Codes/cross-image-attention/pretrained_models/art_fid/art_inception.pth"
ckpt = torch.load(art_inception_model_path, map_location=device)

art_inception_model = inception.Inception3().to(device)
art_inception_model.load_state_dict(ckpt, strict=False)
art_inception_model.eval()
def count_images_in_folder(folder_path):
    # 定义图片文件的扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']

    # 定义视频文件的扩展名
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']

    # 初始化计数器
    image_count = 0

    # 遍历文件夹中的所有文件
    for ext in image_extensions:
        image_files = glob.glob(os.path.join(folder_path, '**', ext), recursive=True)
        image_count += len(image_files)

    # 遍历文件夹中的所有视频文件
    for ext in video_extensions:
        video_files = glob.glob(os.path.join(folder_path, '**', ext), recursive=True)
        image_count -= len(video_files)

    return image_count
# 定义文件路径
# dict = {"ours":"Codes/cross-image-attention/experiments/style_results/1109.txt",
#         "styleid":"/home/allenyljiang/Desktop/CVPR25/Style_ID/output/styleid_1109.txt",
#         "cross_image":"/home/allenyljiang/Desktop/CVPR25/Cross_image_attention/cross_image_attention_1109.txt",
#         "ref2sketch":"/home/allenyljiang/Desktop/CVPR25/ref2sketch/ref2sketch_1109.txt",
#         "semi_ref2sketch":"/home/allenyljiang/Desktop/CVPR25/semi_ref2sketch/semi_ref2sketch_1109.txt"}

# dict = {"all":"Codes/cross-image-attention/experiments/rebuttal/generated_list/0123_all_components.txt",
#         "minus latent":"Codes/cross-image-attention/experiments/rebuttal/generated_list/0123_all_minus_latent_update.txt",
#         "minus TLA":"Codes/cross-image-attention/experiments/rebuttal/generated_list/0123_all_minus_TLA.txt"}
# all ,minus latent,minus TLA,minus SDA,gamma0.4,minus_SDA_and_swap_guidance,gamma0.2
# dict = {"gamma0.4":"Codes/cross-image-attention/experiments/rebuttal/generated_list/0125_gamma0.4.txt",
#         "minus_SDA_and_swap_guidance":"Codes/cross-image-attention/experiments/rebuttal/generated_list/0125_minus_SDA_add_swap_guidance.txt"}
#dict = {"gamma0.2":"Codes/cross-image-attention/experiments/rebuttal/generated_list/0125_gamma0.2.txt",}
dict = {"graph":"Codes/cross-image-attention/experiments/rebuttal/generated_list/0125_add_graph_matching.txt"}
frames_end = 50
results_txt = "Codes/cross-image-attention/experiments/rebuttal/generated_list/metric_results/all_vs_graph_matching.txt"
frames = 8 #10
# with open(results_txt, "w") as f:
#     f.write("Clip Video,Pixel MSE,FID,LPIPS,ArtFID,CFSD,LPIPS (Gray),NFrames\n")
# for frames in range(10,frames_end+1,10):
for key in dict:
    cur_model= key
    video_list_path = dict[key]
    #video_list_path = "/home/allenyljiang/Desktop/CVPR25/Cross_image_attention/cross_image_attention_1109.txt"
    json_output_dir = "Codes/cross-image-attention/experiments/rebuttal/generated_list/metric_results"
    json_file_path = os.path.join(json_output_dir, f"{cur_model}_metrics.json")

    # 确保输出目录存在
    os.makedirs(json_output_dir, exist_ok=True)

    # 读取视频列表文件，跳过空行
    with open(video_list_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    # 解析每一行的内容
    results = {}
    '''
    核心需要确定ori_video，style_path，line(风格化的图片）
    '''
    for line in lines:
        if line.endswith('.mp4'):
            line = os.path.dirname(line)
        n_frames = min(count_images_in_folder(line),frames)
        metric_dir = os.path.dirname(line)
        metric_file_path = os.path.join(metric_dir, "metric.txt")
        struct_name = line.split('/')[-4].split('_')[0] # for bessel
        #struct_name = line.split('/')[-4]
        ori_video = f"Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/{struct_name}/imgs_crop_fore"
        # style_name = line.split('/')[-3]
        # style_path = f"Dataset/Sketch_dataset/ref2sketch_yr/ref_camel/{style_name}.jpg"

        style_path = "Dataset/Sketch_dataset/ref2sketch_yr/ref_camel/best_generated.jpg" # for bessel

        cfsd = compute_cfsd(n_frames, line, ori_video,device="cuda")
        pixel_mse = calculate_pixle_mse(ori_video,line,n_frames,flow_model)
        ### 这里计算每个视频的指标 ###
        clip_video = folder_consistency_clip(line,clip_model,clip_preprocess)
        #clip_video = folder_consistency(line)
        artfid, fid, lpips, lpips_gray = compute_art_fid(line, style_path, ori_video,n_frames,art_inception_model)

        with open(metric_file_path, 'a') as metric_file:
            # 写入 metric.txt 文件
            metric_file.write(f"clip_image: {clip_video}, pixel_mse: {pixel_mse}, fid: {fid}, lpips: {lpips},artfid: {artfid},cfsd:{cfsd},lpips_gray: {lpips_gray}\n")
        results[line] = {
            "clip_video": clip_video,
            "pixel_mse": pixel_mse,
            "fid": fid,
            "fpips": lpips,
            "artfid": artfid,
            "cfsd":cfsd,
            "lpips_gray": lpips_gray,
            "nframes": n_frames
        }


    # 写入 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print("处理完成，结果已保存到指定路径。")
    # 读取 JSON 文件
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # 初始化总和和计数器
    total_clip_video_weighted = 0
    total_pixel_mse_weighted = 0
    total_fid_weighted = 0
    total_fpips_weighted = 0
    total_artfid_weighted = 0
    total_lpips_gray_weighted = 0
    total_cfsd_weighted = 0
    total_nframes = 0

    # 遍历每个视频的数据
    for video_data in data.values():
        nframes = video_data["nframes"]
        total_clip_video_weighted += video_data["clip_video"] * nframes
        total_pixel_mse_weighted += video_data["pixel_mse"] * nframes
        total_fid_weighted += video_data["fid"] * nframes
        total_fpips_weighted += video_data["fpips"] * nframes
        total_artfid_weighted += video_data["artfid"] * nframes
        total_lpips_gray_weighted += video_data["lpips_gray"] * nframes
        total_cfsd_weighted += video_data["cfsd"] * nframes
        total_nframes += nframes

    # 计算加权平均值
    if total_nframes > 0:
        average_clip_video = total_clip_video_weighted / total_nframes
        average_pixel_mse = total_pixel_mse_weighted / total_nframes
        average_fid = total_fid_weighted / total_nframes
        average_fpips = total_fpips_weighted / total_nframes
        average_artfid = total_artfid_weighted / total_nframes
        average_cfsd = total_cfsd_weighted / total_nframes
        average_lpips_gray = total_lpips_gray_weighted / total_nframes
    else:
        average_clip_video = 0
        average_pixel_mse = 0
        average_fid = 0
        average_fpips = 0
        average_cfsd = 0
        average_artfid = 0
        average_lpips_gray = 0

    # 输出结果
    print(f"Weighted Average clip_video: {average_clip_video}")
    print(f"Weighted Average pixel_mse: {average_pixel_mse}")
    print(f"Weighted Average fid: {average_fid}")
    print(f"Weighted Average lpips: {average_fpips}")
    print(f"Weighted Average artfid: {average_artfid}")
    print(f"Weighted Average cfsd: {average_cfsd}")
    print(f"Weighted Average lpips_gray: {average_lpips_gray}")
    with open(results_txt, "a") as f:
        f.write(f"{average_clip_video},{average_pixel_mse},{average_fid},{average_fpips},{average_artfid},{average_cfsd},{average_lpips_gray},{total_nframes}\n")