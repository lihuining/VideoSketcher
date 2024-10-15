import argparse
from audioop import cross

from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path
import yaml
from config import RunConfig, Range
def load_config(print_config = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/tea-pour-debug.yaml',
                        help="Config file path")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Recursively merge base configs
    cur_config_path = args.config
    cur_config = config
    while "base_config" in cur_config and cur_config.base_config != cur_config_path:
        base_config = OmegaConf.load(cur_config.base_config)
        config = OmegaConf.merge(base_config, config)
        cur_config_path = cur_config.base_config
        cur_config = base_config

    prompt = config.generation.prompt
    if isinstance(prompt, str):
        prompt = {"edit": prompt}
    config.generation.prompt = prompt
    OmegaConf.resolve(config)
    if print_config:
        print("[INFO] loaded config:")
        print(OmegaConf.to_yaml(config))
    cross_image_config = RunConfig(
        app_image_path=Path(config['app_image_path']),
        struct_image_path=Path(config['struct_image_path']),
        domain_name=config.get('domain_name'),
        use_masked_adain=config.get('use_masked_adain', True),
        contrast_strength=config.get('contrast_strength', 1.67),
        swap_guidance_scale=config.get('swap_guidance_scale', 1.0),
        gamma=config.get('gamma', 0.75),
        prompt=config.inversion.prompt,
        load_latents=config.load_latents,
    ) # 将读取的 YAML 数据赋值给 RunConfig
    return config,cross_image_config

def save_config(config: DictConfig, path, gene = False, inv = False):
    os.makedirs(path, exist_ok = True)
    config = OmegaConf.create(config)
    if gene:
        config.pop("inversion")
    if inv:
        config.pop("generation")
    OmegaConf.save(config, os.path.join(path, "config.yaml"))