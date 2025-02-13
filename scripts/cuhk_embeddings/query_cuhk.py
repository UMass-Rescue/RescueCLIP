import logging.config
import os
from pathlib import Path
from typing import Sequence, cast

from dotenv import load_dotenv
import torch
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.util import get_vector

from rescueclip.logging_config import LOGGING_CONFIG
from rescueclip.ml_model import (
    CUHK_Apple_Collection,
    encode_image,
    load_inference_clip_model,
    torch_device,
)
from rescueclip.weaviate import WeaviateClientEnsureReady

logger = logging.getLogger(__name__)
load_dotenv()

def main(client: weaviate.WeaviateClient):
    BASE_DIR = Path(os.environ["CUHK_PEDES_DATASET"]) / "out"
    SAMPLE_IMAGE = "0002004.jpg"
    COLLECTION = CUHK_Apple_Collection

    # Get the torch device
    device = torch_device()

    # Load the model into memory
    m = load_inference_clip_model(COLLECTION.model_config, device)

    collection = client.collections.get(COLLECTION.name)

    vector = get_vector(cast(Sequence, encode_image(BASE_DIR, SAMPLE_IMAGE, device, m)))
    result = collection.query.near_vector(
        vector,
        limit=10,
        return_metadata=MetadataQuery(distance=True, certainty=True),
        return_properties=True,
    )
    logger.info(result.objects[0])


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    with WeaviateClientEnsureReady() as client:
        main(client)
