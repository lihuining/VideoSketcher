import os

import cv2
import torch
import imageio
import numpy as np
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

def extract_frames_fix_cnt(video_path, output_folder, num_frames=6):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算每隔多少帧抽取一帧
    if total_frames < num_frames:
        print(f"Warning: Video {video_path} has fewer frames than requested ({total_frames} < {num_frames}). Extracting all frames.")
        interval = 1
    else:
        interval = total_frames // num_frames

    frame_count = 0
    extracted_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or extracted_count >= num_frames:
            break

        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:02d}.png")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
def video_to_frame(video_path: str,
                   frame_dir: str = None,
                   filename_pattern: str = 'frame%03d.jpg',
                   log: bool = True,
                   frame_edit_func=None):
    if not frame_dir:
        frame_dir = os.path.dirname(video_path)
    os.makedirs(frame_dir, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    if log:
        print('img shape: ', image.shape[0:2])

    count = 0
    while success:
        if frame_edit_func is not None:
            image = frame_edit_func(image)

        cv2.imwrite(os.path.join(frame_dir, filename_pattern % count), image)
        success, image = vidcap.read()
        if log:
            print('Read a new frame: ', success, count)
        count += 1

    vidcap.release()

def Pic2Video():
    imgPath = "youimgPath"  # 读取图片路径
    videoPath = "youvideoPath"  # 保存视频路径

    images = os.listdir(imgPath)
    fps = 25  # 每秒25帧数

    # VideoWriter_fourcc为视频编解码器 ('I', '4', '2', '0') —>(.avi) 、('P', 'I', 'M', 'I')—>(.avi)、('X', 'V', 'I', 'D')—>(.avi)、('T', 'H', 'E', 'O')—>.ogv、('F', 'L', 'V', '1')—>.flv、('m', 'p', '4', 'v')—>.mp4
    fourcc = VideoWriter_fourcc(*"MJPG")

    image = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])  # 这里的路径只能是英文路径
        # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype=np.uint8), 1)  # 此句话的路径可以为中文路径
        print(im_name)
        videoWriter.write(frame)
    print("图片转视频结束！")
    videoWriter.release()
    cv2.destroyAllWindows()
def frame_to_video(video_path: str, frame_dir: str,target_size = (512,512), fps=10, log=False,frame_count=100,start_frame=0):

    first_img = True
    writer = imageio.get_writer(video_path, format='FFMPEG', fps=fps)

    file_list = sorted(os.listdir(frame_dir))[start_frame:frame_count]
    for file_name in file_list:
        # if not (file_name.endswith('jpg') or file_name.endswith('png')):
        if not any(file_name.lower().endswith(ext) for ext in image_extensions):
            continue
        if file_name == "combined.png":
            continue

        fn = os.path.join(frame_dir, file_name)
        curImg = imageio.imread(fn)

        if first_img:
            H, W = curImg.shape[0:2]
            if log:
                print('img shape', (H, W))
            first_img = False
        curImg = cv2.resize(curImg, target_size, interpolation=cv2.INTER_AREA)
        writer.append_data(curImg)

    writer.close()


def get_fps(video_path: str):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def get_frame_count(video_path: str):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    aspect_ratio = W / H
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    if H < W:
        W = resolution
        H = int(resolution / aspect_ratio)
    else:
        H = resolution
        W = int(aspect_ratio * resolution)
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image, (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def prepare_frames(input_path: str, output_dir: str, resolution: int, crop, use_limit_device_resolution=False):
    l, r, t, b = crop

    if use_limit_device_resolution:
        resolution = vram_limit_device_resolution(resolution)

    def crop_func(frame):
        H, W, C = frame.shape
        left = np.clip(l, 0, W)
        right = np.clip(W - r, left, W)
        top = np.clip(t, 0, H)
        bottom = np.clip(H - b, top, H)
        frame = frame[top:bottom, left:right]
        return resize_image(frame, resolution)

    video_to_frame(input_path, output_dir, '%04d.png', False, crop_func)


def vram_limit_device_resolution(resolution, device="cuda"):
    # get max limit target size
    gpu_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    # table of gpu memory limit
    gpu_table = {24: 1280, 18: 1024, 14: 768, 10: 640, 8: 576, 7: 512, 6: 448, 5: 320, 4: 192, 0: 0}
    # get user resize for gpu
    device_resolution = max(val for key, val in gpu_table.items() if key <= gpu_vram)
    print(f"Limit VRAM is {gpu_vram} Gb and size {device_resolution}.")
    if gpu_vram < 4:
        print(f"Small VRAM to use GPU. Configuration resolution will be used.")
    if resolution < device_resolution:
        print(f"Video will not resize")
        return resolution
    return device_resolution
