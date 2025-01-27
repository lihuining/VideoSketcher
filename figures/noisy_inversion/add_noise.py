import os
from PIL import Image
import torch
from torchvision import transforms
from diffusers import DDIMScheduler, StableDiffusionPipeline

# 设置 CUDA_LAUNCH_BLOCKING 环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 初始化模型和调度器
model_id = "/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, use_auth_token=True).to("cuda")
# 设置调度器的时间步数
num_inference_steps = 1000
scheduler.set_timesteps(num_inference_steps)
# 输出调度器的时间步范围
print("Scheduler timesteps range:", scheduler.timesteps)

# 定义图片路径
image_dir = "/home/allenyljiang/Desktop/论文用图/stylized图片"
save_dir = "/media/allenyljiang/564AFA804AFA5BE51/Codes/cross-image-attention/figures/add_noise_surf_new"
os.makedirs(save_dir, exist_ok=True)

# 定义10个不同的时间步
timesteps = [100]

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # 转换为 (C, H, W) 格式并缩放至 [0, 1]
])

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)

    # 加载和预处理图片
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to("cuda")  # 添加 batch 维度

    # 对图片进行加噪
    noise = torch.randn_like(image)

    for timestep in timesteps:
        # # 确保 timestep 在允许的范围内
        # if timestep >= scheduler.num_timesteps:
        #     print(f"Timestep {timestep} 超出范围，跳过该时间步。")
        #     continue

        noisy_image = scheduler.add_noise(image, noise, timesteps=torch.tensor([timestep]).to("cuda"))

        # 裁剪到 [0, 1] 范围
        noisy_image = torch.clamp(noisy_image, 0, 1)

        # 将加噪后的图片转换回PIL图像
        noisy_image = noisy_image.detach().cpu().numpy()[0]
        noisy_image = (noisy_image * 255).astype("uint8")
        noisy_image = Image.fromarray(noisy_image.transpose(1, 2, 0))

        # 保存加噪后的图片
        save_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_timestep_{timestep}.png")
        noisy_image.save(save_path)

