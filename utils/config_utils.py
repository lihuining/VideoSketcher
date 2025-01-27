import argparse
from audioop import cross

from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path
import yaml
from config import RunConfig, Range
# 从列表创建 Range 对象的辅助函数
def parse_range(range_list):
    return Range(start=range_list[0], end=range_list[1])
def load_config(print_config = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/tea-pour-debug.yaml',
                        help="Config file path")

    parser.add_argument('--ablation-value', type=str,
                        default='',
                        help="ablation value")
    parser.add_argument('--save-file-path', type=str,
                        default='',
                        help="save-file-path")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.ablation_value = args.ablation_value
    config.save_file_path = args.save_file_path
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

    # 将列表格式的 cross_attn_*_range 转换为 Range 对象
    # config.cross_attn_64_range = parse_range(config['cross_attn_64_range'])
    # config.cross_attn_32_range = parse_range(config['cross_attn_32_range'])
    # config.adain_range = parse_range(config['adain_range'])
    if config.use_masked_adain:
        if config.domain_name is None and config.style_domain_name is None:
            config.domain_name = "object"
            config.style_domain_name = "object"
        elif config.domain_name is None:
            config.domain_name = "object"
        else:
            config.style_domain_name = "object"
    if config.prompt is None:
        config.prompt = f"a photo of an {config.domain_name}"  # a photo of an object
        config.style_prompt = f"a photo of an {config.style_domain_name}"
    if config.object_noun is None:
        config.object_noun = config.domain_name
    if config.style_domain_name is None:
        config.style_object_noun = config.style_domain_name
    # cross_image_config = RunConfig(
    #     app_image_path=Path(config['app_image_path']),
    #     struct_image_path=Path(config['struct_image_path']),
    #     domain_name=config.get('domain_name',None),
    #     use_masked_adain=config.get('use_masked_adain', True),
    #     use_adain = config.get('use_adain',True),
    #     contrast_strength=config.get('contrast_strength', 1.67),
    #     swap_guidance_scale=config.get('swap_guidance_scale', 1.0),
    #     gamma=config.get('gamma', 0.75),
    #     # prompt=config.inversion.prompt,
    #     object_noun = config.get('object_noun',None),
    #     load_latents=config.load_latents,
    #     cfg_inversion_style = config.get('cfg_inversion_style', 0.0),
    #     cfg_inversion_contents = config.get('cfg_inversion_contents',0.0),
    #     style_domain_name = config.get("style_domain_name",None)
    # ) # 将读取的 YAML 数据赋值给 RunConfig
    return config

def save_config(config: DictConfig, path, gene = False, inv = False):
    os.makedirs(path, exist_ok = True)
    config = OmegaConf.create(config)
    if gene:
        config.pop("inversion")
    if inv:
        config.pop("generation")
    OmegaConf.save(config, os.path.join(path, "config.yaml"))