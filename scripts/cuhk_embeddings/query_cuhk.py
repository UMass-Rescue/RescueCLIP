import logging.config
from typing import Sequence, cast

import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.util import get_vector

from rescueclip.logging_config import LOGGING_CONFIG
from rescueclip.open_clip import (
    apple_DFN5B_CLIP_ViT_H_14_384,
    encode_image,
    load_inference_clip_model,
    torch_device,
)
from rescueclip.weaviate import WeaviateClientEnsureReady

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main(client: weaviate.WeaviateClient):
    BASE_DIR = "./data/CUHK-PEDES/out"
    SAMPLE_IMAGE = "0002004.jpg"
    model_config = apple_DFN5B_CLIP_ViT_H_14_384
    COLLECTION_NAME = model_config.weaviate_collection_name

    # Get the torch device
    device = torch_device()

    # Load the model into memory
    model, preprocess, _ = load_inference_clip_model(model_config, device)

    collection = client.collections.get(COLLECTION_NAME)

    vector = get_vector(cast(Sequence, encode_image(BASE_DIR, SAMPLE_IMAGE, device, model, preprocess)))
    result = collection.query.near_vector(
        vector, limit=10, return_metadata=MetadataQuery(distance=True, certainty=True), return_properties=True
    )
    logger.info(result.objects[0])


if __name__ == "__main__":
    with WeaviateClientEnsureReady() as client:
        main(client)
