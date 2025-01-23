from calendar import c
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import open_clip

CACHE_DIR = "./.cache/clip"


@dataclass
class OpenClipConfig:
    model_name: str
    checkpoint_name: Optional[str]


ViT_B_32 = OpenClipConfig(
    model_name="ViT-B-32",
    checkpoint_name="laion2b_s34b_b79k",
)

apple_DFN5B_CLIP_ViT_H_14_384 = OpenClipConfig(
    model_name="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", checkpoint_name=None
)


def load_inference_clip_model(config: OpenClipConfig, device: str, cache_dir: str = CACHE_DIR):
    model, _, preprocess_image = open_clip.create_model_and_transforms(
        config.model_name, pretrained=config.checkpoint_name, device=device, cache_dir=cache_dir
    )
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer(config.model_name, cache_dir=cache_dir)

    return model, preprocess_image, tokenizer
