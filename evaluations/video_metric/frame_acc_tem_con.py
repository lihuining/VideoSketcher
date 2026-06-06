import torch
import clip
from PIL import Image
import glob
import numpy as np
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def crop_read_image_path(image_path):
    origin_image = Image.open(image_path)
    w, h = origin_image.size
    if h > w:
        origin_image = origin_image.crop((0, h-w, w, h))
    return origin_image


def edit_success(image_path, source_prompt,target_prompt):
    image = preprocess(crop_read_image_path(image_path)).unsqueeze(0).to(device)

    text = clip.tokenize([source_prompt, target_prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  
    return probs[0,1] >= probs[0,0], image_features


def encode_images(image_path):
    image = preprocess(crop_read_image_path(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features
def encode_images_clip(image_path,model,preprocess):
    image = preprocess(crop_read_image_path(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features
def folder_success(folder, source_prompt, target_prompt):
    print(folder)
    # file_list = sorted(glob(folder+'/*png'))
    # 匹配 .jpg 和 .png 文件
    jpg_files = glob.glob(os.path.join(folder, '*.jpg'))
    png_files = glob.glob(os.path.join(folder, '*.png'))

    # 合并文件列表并排序
    file_list = sorted(jpg_files + png_files)
    normalized_feature_list = []
    print(file_list)
    count = 0.0
    for f_path in file_list:        
        success, image_feature = edit_success(f_path, source_prompt,target_prompt)
        if success: count +=1.0
        normalized_feature_list.append(image_feature/torch.sqrt(torch.sum(image_feature**2, axis=1, keepdims=True)))
    frame_const_list = []
    frame_const_list_sum = 0.0
    for i in range(len(normalized_feature_list)-1):
        sim_i = torch.sum(normalized_feature_list[i]*normalized_feature_list[i+1], axis=1)
        frame_const_list.append( sim_i )
        frame_const_list_sum += sim_i
    frame_const_list_avg = frame_const_list_sum/(len(normalized_feature_list)-1)
    print(f'average temporal frame consistency: {frame_const_list_avg}')

    return count/len(file_list), frame_const_list_sum/(len(normalized_feature_list)-1)
def folder_consistency(folder,n_frames=50):
    # print(folder)
    # file_list = sorted(glob(folder+'/*jpg'))
    jpg_files = glob.glob(os.path.join(folder, '*.jpg'))
    png_files = glob.glob(os.path.join(folder, '*.png'))

    # 合并文件列表并排序
    file_list = sorted(jpg_files + png_files)
    normalized_feature_list = []
    # print(file_list)
    count = 0.0
    for f_path in file_list:
        image_feature = encode_images(f_path)
        normalized_feature_list.append(image_feature/torch.sqrt(torch.sum(image_feature**2, axis=1, keepdims=True)))
    frame_const_list = []
    frame_const_list_sum = 0.0
    for i in range(len(normalized_feature_list)-1):
        sim_i = torch.sum(normalized_feature_list[i]*normalized_feature_list[i+1], axis=1)
        frame_const_list.append( sim_i )
        frame_const_list_sum += sim_i
    frame_const_list_avg = frame_const_list_sum/(len(normalized_feature_list)-1)

    print(f'average temporal frame consistency: {frame_const_list_avg}')
    return frame_const_list_avg.item()

    return count/len(file_list), frame_const_list_sum/(len(normalized_feature_list)-1)
def folder_consistency_clip(folder,model,preprocess):
    # print(folder)
    # file_list = sorted(glob(folder+'/*jpg'))
    jpg_files = glob.glob(os.path.join(folder, '*.jpg'))
    png_files = glob.glob(os.path.join(folder, '*.png'))

    # 合并文件列表并排序
    file_list = sorted(jpg_files + png_files)
    normalized_feature_list = []
    # print(file_list)
    count = 0.0
    for f_path in file_list:
        image_feature = encode_images_clip(f_path,model,preprocess)
        normalized_feature_list.append(image_feature/torch.sqrt(torch.sum(image_feature**2, axis=1, keepdims=True)))
    frame_const_list = []
    frame_const_list_sum = 0.0
    for i in range(len(normalized_feature_list)-1):
        sim_i = torch.sum(normalized_feature_list[i]*normalized_feature_list[i+1], axis=1)
        frame_const_list.append( sim_i )
        frame_const_list_sum += sim_i
    frame_const_list_avg = frame_const_list_sum/(len(normalized_feature_list)-1)

    print(f'average temporal frame consistency: {frame_const_list_avg}')
    return frame_const_list_avg.item()

    return count/len(file_list), frame_const_list_sum/(len(normalized_feature_list)-1)
if __name__ == "__main__":
    #folder_path ="/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/blackswan/ref0000/2.1_chunk_size2matching_guidance_1start_time1end_time361_4/generated_result" # 0.9736
    #folder_path ="/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/blackswan/ref0000/2.1_chunk_size2_latent_update_6" # 0.9736
    folder_path="/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/blackswan/ref0000/2.1_chunk_size2_latent_update_2/generated_result"
    n_frames = 8 # graph matching:0.98583984375  latent update:0.92626953125 all:0.98681640625
    #folder_path ="/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/cows/2/2.1_chunk_size2_cross_frame/generated_result" # 0.9736
    print(folder_consistency(folder = folder_path,n_frames=n_frames))

