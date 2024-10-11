from invert import Inverter
from generate import Generator
from utils import load_config, init_model, seed_everything, get_frame_ids
import time
if __name__ == "__main__":
    start_time = time.time()
    config = load_config()
    pipe, scheduler, model_key = init_model(
        config.device, config.sd_version, config.model_key, config.generation.control, config.float_precision)
    config.model_key = model_key
    seed_everything(config.seed)
    
    print("Start inversion!")
    is_time = time.time()
    inversion = Inverter(pipe, scheduler, config)
    inversion(config.input_path, config.inversion.save_path)
    ie_time = time.time()
    print("DDIM inversion time",ie_time - is_time)

    print("Start generation!")
    gs_time = time.time()
    generator = Generator(pipe, scheduler, config)
    frame_ids = get_frame_ids(
        config.generation.frame_range, config.generation.frame_ids)
    generator(config.input_path, config.generation.latents_path,
              config.generation.output_path, frame_ids=frame_ids)
    end_time = time.time()
    print("generated time is",end_time - gs_time)
    print("cost time is",end_time - start_time)
'''
python run_vidtome.py --config configs/tea-pour-debug.yaml
'''