import torch
from diffusers import DDIMScheduler
import time
from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from models.stable_diffusion_video import CrossImageAttentionStableDiffusionVideoPipeline
from models.stable_diffusion_video_wo_cfg import CrossImageAttentionStableDiffusionVideoPipeline as CrossImageAttentionStableDiffusionVideoPipelineWocfg
from models.unet_2d_condition import FreeUUNet2DConditionModel


def get_stable_diffusion_model(sd_version="1.5",choice="video") -> CrossImageAttentionStableDiffusionPipeline:
    print("Loading Stable Diffusion model...")
    start_time = time.time()
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_id = "/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stable-diffusion-v1-5"
    if sd_version == "1.5":
        model_id = "/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stable-diffusion-v1-5"
        # [b1,b2,s1,s2]
        free_u = [1.5,1.6,0.9,0.2]
        # free_u = [1.2,1.4,0.9,0.2] # 原始参数
    if sd_version == "2.1":
        model_id = "/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stable-diffusion-2-1-base"
        free_u = [1.4,1.6,0.9,0.2]
    if sd_version == "sdxl":
        model_id = "/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stabilityai/stable-diffusion-xl-base-1.0"
        free_u = [1.3,1.4,0.9,0.2]
    # if sd_version == "2.0":
    #     model_id = ""
    if choice == "video":
        pipe = CrossImageAttentionStableDiffusionVideoPipeline.from_pretrained(model_id,
                                                                          safety_checker=None).to(device)

    elif choice == "video_wo_cfg":
        pipe = CrossImageAttentionStableDiffusionVideoPipelineWocfg.from_pretrained(model_id,
                                                                          safety_checker=None).to(device)   
    else:
        pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained(model_id,
                                                                          safety_checker=None).to(device)
    pipe.unet = FreeUUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    pipe.unet.freeu_args = free_u
    # FreeUUNet2DConditionModel.freeu_args = free_u
    # pipe.unet.forward = FreeUUNet2DConditionModel.forward
    # pipe.unet.to(device)
    # pipe.unet.forward = FreeUUNet2DConditionModel.from_pretrained(
    #     model_id,
    #     subfolder="unet",
    #     freeu_args=[1.1, 1.2, 1.0, 0.2]
    # ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    end_time = time.time()
    print(f"Model loaded, took {(end_time - start_time):.2f} s")
    # print("Done.")
    if choice == "video":
        return pipe,model_id
    else:
        return pipe
