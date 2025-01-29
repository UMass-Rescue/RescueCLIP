import logging.config
import os
from calendar import c
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import open_clip
import torch
import torchvision
from PIL import Image
from torch import Tensor

from rescueclip.logging_config import LOGGING_CONFIG

logger = logging.getLogger(__name__)

CACHE_DIR = "./.cache/clip"


def torch_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"

    logger.info("Using device: %s", device)
    return device


@dataclass
class OpenClipModelConfig:
    model_name: str
    checkpoint_name: Optional[str]
    tokenizer_model_name: str
    weaviate_friendly_model_name: str


ViT_B_32 = OpenClipModelConfig(
    model_name="ViT-B-32",
    checkpoint_name="laion2b_s34b_b79k",
    tokenizer_model_name="ViT-B-32",
    weaviate_friendly_model_name="ViT-B-32",
)

# https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378
apple_DFN5B_CLIP_ViT_H_14_384 = OpenClipModelConfig(
    model_name="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
    checkpoint_name=None,
    tokenizer_model_name="ViT-H-14",
    weaviate_friendly_model_name="Apple_DFN5B_CLIP_ViT_H_14_384",
)

laion_CLIP_ViT_bigG_14_laion2B_39B_b160k = OpenClipModelConfig(
    model_name="hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    checkpoint_name=None,
    tokenizer_model_name="hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    weaviate_friendly_model_name="CLIP_ViT_bigG_14_laion2B_39B_b160k",
)

@dataclass
class CollectionConfig:
    name: str
    model_config: OpenClipModelConfig

CUHK_Apple_Collection = CollectionConfig(
    name=apple_DFN5B_CLIP_ViT_H_14_384.weaviate_friendly_model_name + '_CUHK',
    model_config=apple_DFN5B_CLIP_ViT_H_14_384
)


def load_inference_clip_model(config: OpenClipModelConfig, device: str, cache_dir: str = CACHE_DIR) -> tuple[
    open_clip.CLIP,
    torchvision.transforms.Compose,
    open_clip.tokenizer.HFTokenizer | open_clip.tokenizer.SimpleTokenizer,
]:
    model, _, preprocess_image = open_clip.create_model_and_transforms(
        config.model_name, pretrained=config.checkpoint_name, device=device, cache_dir=cache_dir
    )
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer(config.tokenizer_model_name, cache_dir=cache_dir)

    logger.info("Loaded model: %s", config.model_name)
    return model, preprocess_image, tokenizer  # type: ignore


def encode_image(
    base_dir: str | Path,
    file: str | Path,
    device: str,
    model: open_clip.CLIP,
    preprocess: torchvision.transforms.Compose,
) -> Tensor:
    pil_image = Image.open(os.path.join(base_dir, file))
    image = preprocess(pil_image).unsqueeze(0).to(device)  # type: ignore
    pil_image.close()
    return model.encode_image(image)
