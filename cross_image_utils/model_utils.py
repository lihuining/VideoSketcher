import torch
from diffusers import DDIMScheduler
import time
from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from models.stable_diffusion_video import CrossImageAttentionStableDiffusionVideoPipeline
from models.stable_diffusion_video_wo_cfg import CrossImageAttentionStableDiffusionVideoPipeline as CrossImageAttentionStableDiffusionVideoPipelineWocfg
from models.unet_2d_condition import FreeUUNet2DConditionModel


def get_stable_diffusion_model(choice="video") -> CrossImageAttentionStableDiffusionPipeline:
    print("Loading Stable Diffusion model...")
    start_time = time.time()
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_id = "/media/allenyljiang/5234E69834E67DFB/StableDiffusion_Models/stable-diffusion-v1-5"
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
    pipe.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    end_time = time.time()
    print(f"Model loaded, took {(end_time - start_time):.2f} s")
    # print("Done.")
    if choice == "video":
        return pipe,model_id
    else:
        return pipe
