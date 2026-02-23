import os
import shutil

# 定义需要处理的目录列表
directories = [
    'Dataset/Video_Dataset/loveu-tgve-2023/videvo_480p/480p_frames/bird-on-feeder',
    'Dataset/Video_Dataset/loveu-tgve-2023/videvo_480p/480p_frames/cat-in-the-sun',
    'Dataset/Video_Dataset/loveu-tgve-2023/videvo_480p/480p_frames/fireworks-display',
    'Dataset/Video_Dataset/loveu-tgve-2023/videvo_480p/480p_frames/setting-sun',
    'Dataset/Video_Dataset/loveu-tgve-2023/videvo_480p/480p_frames/ship-sailing',
    'Dataset/Video_Dataset/loveu-tgve-2023/videvo_480p/480p_frames/wind-turbines-at-dusk'
]

# 遍历每个目录
for directory in directories:
    # 创建新目录
    new_directory = os.path.join(directory, 'ori')
    os.makedirs(new_directory, exist_ok=True)

    # 获取当前目录下的所有文件
    files = os.listdir(directory)

    # 移动文件到新目录
    for file in files:
        source_path = os.path.join(directory, file)
        destination_path = os.path.join(new_directory, file)

        if os.path.isfile(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved: {source_path} -> {destination_path}")
