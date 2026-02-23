import os.path

from eval_artfid import compute_art_fid,compute_cfsd


# /media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/camel/ref0001/2.1_chunk_size2matching_guidance_1start_time1end_time621_3/generated_result
generated_dir = "/media/allenyljiang/2CD8318DD83155F4/CVPR2025/Struct_latents/camel/ref0001/2.1_chunk_size2_14"
line = os.path.join(generated_dir,'generated_result')
style_path ="Dataset/Sketch_dataset/ref2sketch_yr/ref_camel/ref0001.jpg"
n_frames = 4
ori_video = "Dataset/Video_Dataset/DAVIS-2017-trainval-Full-Resolution/DAVIS/dataset/camel/imgs_crop_fore"
artfid, fid, lpips, lpips_gray = compute_art_fid(line, style_path, ori_video,n_frames)
print(artfid,fid,lpips,lpips_gray)