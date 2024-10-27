from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

from numpy.random import choice


class Range(NamedTuple):
    start: int
    end: int


@dataclass
class RunConfig:
    # Appearance image path
    app_image_path: Path
    # Struct image path
    struct_image_path: Path
    # Domain name (e.g., buildings, animals)
    domain_name: Optional[str] = None # 描述content的类别
    # Output path
    output_path: Path = Path('./output')
    # Random seed
    seed: int = 42
    # Input prompt for inversion (will use domain name as default)
    prompt: Optional[str] = None
    # Number of timesteps
    num_timesteps: int = 100
    # Whether to use a binary mask for performing AdaIN
    use_masked_adain: bool = False
    # Whether to performing AdaIN
    use_adain: bool = False
    # Timesteps to apply cross-attention on 64x64 layers
    cross_attn_64_range: Range = Range(start=10, end=90)
    # Timesteps to apply cross-attention on 32x32 layers
    cross_attn_32_range: Range = Range(start=10, end=70)
    # Timesteps to apply AdaIn
    adain_range: Range = Range(start=20, end=100)
    # adain_range: Range = Range(start=1, end=100)
    # Swap guidance scale
    swap_guidance_scale: float = 3.5
    # Attention contrasting strength
    contrast_strength: float = 1.67
    # Object nouns to use for self-segmentation (will use the domain name as default)
    object_noun: Optional[str] = None
    # Whether to load previously saved inverted latent codes
    load_latents: bool = True
    # Number of steps to skip in the denoising process (used value from original edit-friendly DDPM paper)
    skip_steps: int = 32
    # gamma context preservation
    gamma: float = 0.75
    # temperature rescale attention map
    temperature: float = 1.0 # 1.5
    choice: Optional[str] = "image"
    cfg_inversion_style: float = 3.5
    cfg_inversion_contents: float = 3.5
    style_domain_name: Optional[str] = None  # 描述style的domain name
    num_segments: int = 3 # 聚类的段个数，影响分割结果

    def __post_init__(self):
        save_name = f'app={self.app_image_path.stem}---struct={self.struct_image_path.stem}'
        self.output_path = Path(self.output_path) / self.domain_name / save_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Handle the domain name, prompt, and object nouns used for masking, etc.
        if self.use_masked_adain and self.domain_name is None:
            raise ValueError("Must provide --domain_name and --prompt when using masked AdaIN")
        if self.use_masked_adain:
            if self.domain_name is None and self.style_domain_name is None:
                self.domain_name = "object"
                self.style_domain_name = "object"
            elif self.domain_name is None:
                self.domain_name = "object"
            else:
                self.style_domain_name = "object"
        if self.prompt is None:
            self.prompt = f"a photo of an {self.domain_name}" # a photo of an object
            self.style_prompt = f"a photo of an {self.style_domain_name}"
        if self.object_noun is None:
            self.object_noun = self.domain_name
            self.style_object_noun = self.style_domain_name
        self.choice = self.choice
        self.cfg_inversion_style = self.cfg_inversion_style
        self.cfg_inversion_contents = self.cfg_inversion_contents
        self.num_segments = self.num_segments
        # Define the paths to store the inverted latents to
        self.latents_path = Path(self.output_path) / "latents"
        self.latents_path.mkdir(parents=True, exist_ok=True)
        self.app_latent_save_path = self.latents_path / f"{self.app_image_path.stem}.pt"
        self.struct_latent_save_path = self.latents_path / f"{self.struct_image_path.stem}.pt"
        # define the paths to visualize the attention map
        self.cross_attention_dir = Path(self.output_path) / "cross_attention"
        self.cross_attention_dir.mkdir(parents=True, exist_ok=True)

        self.self_attention_dir = Path(self.output_path) / "self_attention"
        self.self_attention_dir.mkdir(parents=True, exist_ok=True)

        # cluster
        self.cluster_path = Path(self.output_path) / "clusters"
        self.cluster_path.mkdir(parents=True, exist_ok=True)
        # masks
        self.mask_path = Path(self.output_path) / "masks"
        self.mask_path.mkdir(parents=True, exist_ok=True)
    def update_dir(self):
        # Define the paths to store the inverted latents to
        self.latents_path = Path(self.output_path) / "latents"
        self.latents_path.mkdir(parents=True, exist_ok=True)
        self.app_latent_save_path = self.latents_path / f"{self.app_image_path.stem}.pt"
        self.struct_latent_save_path = self.latents_path / f"{self.struct_image_path.stem}.pt"
        # define the paths to visualize the attention map
        self.cross_attention_dir = Path(self.output_path) / "cross_attention"
        self.cross_attention_dir.mkdir(parents=True, exist_ok=True)
        
        self.self_attention_dir = Path(self.output_path) / "self_attention"
        self.self_attention_dir.mkdir(parents=True, exist_ok=True)

        # cluster
        self.cluster_path = Path(self.output_path) / "clusters"
        self.cluster_path.mkdir(parents=True, exist_ok=True)
        # masks
        self.mask_path = Path(self.output_path) / "masks"
        self.mask_path.mkdir(parents=True, exist_ok=True)


        
        
