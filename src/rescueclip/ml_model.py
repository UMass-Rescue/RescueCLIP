import logging.config
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional, assert_never

import open_clip
import torch
import torchvision
import transformers
from dotenv import load_dotenv
from PIL import Image
from torch import Tensor

from rescueclip import cuhk
from rescueclip.cuhk import FileToHashesMap

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./.cache")


def torch_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"

    logger.info("Using device: %s", device)
    return device


class LIPModelProvider(StrEnum):
    OPEN_CLIP = "open_clip"
    SIGLIP = "siglip"
    PDNA = "pdna"
    RANDOM = "random"


@dataclass
class LIPModelConfig:
    model_name: str
    checkpoint_name: Optional[str]
    tokenizer_model_name: Optional[str]
    weaviate_friendly_model_name: str
    provider: LIPModelProvider

    def __str__(self):
        return f"({self.model_name})"


PDNA_Model = LIPModelConfig(
    model_name="PhotoDNA",
    checkpoint_name=None,
    tokenizer_model_name=None,
    weaviate_friendly_model_name="PDNA_Model",
    provider=LIPModelProvider.PDNA,
)

Random_Model = LIPModelConfig(
    model_name="Random",
    checkpoint_name=None,
    tokenizer_model_name=None,
    weaviate_friendly_model_name="Random_Model",
    provider=LIPModelProvider.RANDOM,
)

ViT_B_32 = LIPModelConfig(
    model_name="ViT-B-32",
    checkpoint_name="laion2b_s34b_b79k",
    tokenizer_model_name="ViT-B-32",
    weaviate_friendly_model_name="ViT_B_32",
    provider=LIPModelProvider.OPEN_CLIP,
)

# https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378
apple_DFN5B_CLIP_ViT_H_14_384 = LIPModelConfig(
    model_name="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384",
    checkpoint_name=None,
    tokenizer_model_name="ViT-H-14",
    weaviate_friendly_model_name="Apple_DFN5B_CLIP_ViT_H_14_384",
    provider=LIPModelProvider.OPEN_CLIP,
)

laion_CLIP_ViT_bigG_14_laion2B_39B_b160k = LIPModelConfig(
    model_name="hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    checkpoint_name=None,
    tokenizer_model_name="hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    weaviate_friendly_model_name="CLIP_ViT_bigG_14_laion2B_39B_b160k",
    provider=LIPModelProvider.OPEN_CLIP,
)

# https://github.com/facebookresearch/metaclip?tab=readme-ov-file
metaclip_ViT_bigG_14_quickgelu_224 = LIPModelConfig(
    model_name="ViT-bigG-14-quickgelu",
    checkpoint_name="metaclip_fullcc",
    tokenizer_model_name="ViT-bigG-14-quickgelu",
    weaviate_friendly_model_name="MetaCLIP_ViT_bigG_14_quickgelu_224",
    provider=LIPModelProvider.OPEN_CLIP,
)

# https://huggingface.co/docs/transformers/en/model_doc/siglip#using-the-model-yourself
google_siglip_base_patch16_224 = LIPModelConfig(
    model_name="google/siglip-base-patch16-224",
    checkpoint_name=None,
    tokenizer_model_name=None,
    weaviate_friendly_model_name="SIGLIP_Base_Patch16_224",
    provider=LIPModelProvider.SIGLIP,
)

# https://huggingface.co/google/siglip-so400m-patch14-384
google_siglip_so400m_patch14_384 = LIPModelConfig(
    model_name="google/siglip-so400m-patch14-384",
    checkpoint_name=None,
    tokenizer_model_name=None,
    weaviate_friendly_model_name="SIGLIP_SO400M_Patch14_384",
    provider=LIPModelProvider.SIGLIP,
)


@dataclass
class CollectionConfig:
    name: str
    model_config: LIPModelConfig

    def __str__(self):
        return f"\n\tCollection name: {self.name}" + f"\n\tModel Config: {str(self.model_config)}\n"


# CLIP

CUHK_ViT_B_32_Collection = CollectionConfig(
    name=ViT_B_32.weaviate_friendly_model_name + "_CUHK", model_config=ViT_B_32
)

CUHK_laion_CLIP_ViT_bigG_14_laion2B_39B_b160k_Collection = CollectionConfig(
    name=laion_CLIP_ViT_bigG_14_laion2B_39B_b160k.weaviate_friendly_model_name + "_CUHK",
    model_config=laion_CLIP_ViT_bigG_14_laion2B_39B_b160k,
)

CUHK_Apple_Collection = CollectionConfig(
    name=apple_DFN5B_CLIP_ViT_H_14_384.weaviate_friendly_model_name + "_CUHK",
    model_config=apple_DFN5B_CLIP_ViT_H_14_384,
)

CUHK_MetaCLIP_ViT_bigG_14_quickgelu_224_Collection = CollectionConfig(
    name=metaclip_ViT_bigG_14_quickgelu_224.weaviate_friendly_model_name + "_CUHK",
    model_config=metaclip_ViT_bigG_14_quickgelu_224,
)

## SigLIP

CUHK_Google_Siglip_Base_Patch16_224_Collection = CollectionConfig(
    name=google_siglip_base_patch16_224.weaviate_friendly_model_name + "_CUHK",
    model_config=google_siglip_base_patch16_224,
)

CUHK_Google_Siglip_SO400M_Patch14_384_Collection = CollectionConfig(
    name=google_siglip_so400m_patch14_384.weaviate_friendly_model_name + "_CUHK",
    model_config=google_siglip_so400m_patch14_384,
)

## PDNA

CUHK_PDNA_Collection = CollectionConfig(
    name=PDNA_Model.weaviate_friendly_model_name + "_CUHK", model_config=PDNA_Model
)

## Random

CUHK_Random_Model_Collection = CollectionConfig(
    name=Random_Model.weaviate_friendly_model_name + "_CUHK", model_config=Random_Model
)


@dataclass
class CLIPModel:
    model: open_clip.CLIP
    preprocess: torchvision.transforms.Compose
    tokenizer: open_clip.tokenizer.HFTokenizer | open_clip.tokenizer.SimpleTokenizer


@dataclass
class SiglipModel:
    model: transformers.SiglipModel
    processor: transformers.SiglipProcessor


@dataclass
class PhotoDNAModel:
    filename_to_hashes: FileToHashesMap

@dataclass
class RandomModel:
    vector_dim: int

type LIPModel = CLIPModel | SiglipModel | PhotoDNAModel | RandomModel


def load_embedding_model(config: LIPModelConfig, device: str, cache_dir: Path = CACHE_DIR) -> LIPModel:
    match config.provider:
        case LIPModelProvider.OPEN_CLIP:
            clip_model, _, preprocess_image = open_clip.create_model_and_transforms(
                config.model_name,
                pretrained=config.checkpoint_name,
                device=device,
                cache_dir=str(cache_dir / "clip"),
            )
            clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
            assert (
                config.tokenizer_model_name is not None
            ), f"Tokenizer model name must be specified for {config}"
            tokenizer = open_clip.get_tokenizer(
                config.tokenizer_model_name, cache_dir=str(cache_dir / "clip")
            )

            logger.info("Loaded CLIP model: %s", config.model_name)
            return CLIPModel(clip_model, preprocess_image, tokenizer)  # type: ignore
        case LIPModelProvider.SIGLIP:
            siglip_model: transformers.SiglipModel = transformers.AutoModel.from_pretrained(
                config.model_name, cache_dir=str(cache_dir / "siglip"), device_map=device
            )
            processor: transformers.SiglipProcessor = transformers.AutoProcessor.from_pretrained(
                config.model_name, cache_dir=str(cache_dir / "siglip"), device_map=device
            )
            logger.info("Loaded SigLIPmodel: %s", config.model_name)
            return SiglipModel(siglip_model, processor)
        case LIPModelProvider.PDNA:
            load_dotenv()
            hashes, _ = cuhk.get_pdna_hashes(Path(os.environ["PDNA_HASHES_FILE"]))
            return PhotoDNAModel(filename_to_hashes=hashes)
        case LIPModelProvider.RANDOM:
            return RandomModel(vector_dim=512)
        case _:
            assert_never(config.provider)


def encode_image(
    base_dir: str | Path,
    file: str | Path,
    device: str,
    m: LIPModel,
) -> Tensor:
    match m:
        case CLIPModel():
            pil_image = Image.open(os.path.join(base_dir, file))
            image = m.preprocess(pil_image).unsqueeze(0).to(device)  # type: ignore
            pil_image.close()
            return m.model.encode_image(image)
        case SiglipModel():
            pil_image = Image.open(os.path.join(base_dir, file))
            with torch.autocast(device):
                pil_image = pil_image.convert("RGB")
                inputs = m.processor(images=pil_image, padding="max_length", return_tensors="pt")
                pil_image.close()
                inputs.to(device)
                return m.model.get_image_features(inputs.data["pixel_values"])
        case PhotoDNAModel():
            return torch.from_numpy(m.filename_to_hashes[str(file)]).to(device)
        case RandomModel():
            return torch.randn(1, m.vector_dim).to(device)
        case _:
            raise ValueError(f"Unknown model type: {type(model)}")
