import logging.config
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import weaviate
from PIL import Image
from tqdm import tqdm
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.util import generate_uuid5, get_vector

from rescueclip.cuhk import get_sets
from rescueclip.logging_config import LOGGING_CONFIG
from rescueclip.open_clip import (
    CUHK_Apple_Collection,
    encode_image,
    load_inference_clip_model,
    torch_device,
)
from rescueclip.weaviate import WeaviateClientEnsureReady

logger = logging.getLogger(__name__)


def create_or_get_collection(client: weaviate.WeaviateClient, collection_name: str):
    try:
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=[
                Configure.NamedVectors.none(
                    name="embedding",
                    vector_index_config=Configure.VectorIndex.hnsw(),
                )
            ],
            properties=[
                Property(name="set_number", data_type=DataType.INT),
                Property(name="file_name", data_type=DataType.TEXT),
            ],
        )
        logger.info("New collection created: %s", collection.name)
    except weaviate.exceptions.UnexpectedStatusCodeError as e:
        if e.status_code == 422 and "already exists" in e.message:
            logger.info("Collection already exists. Skipping creation.")
            collection = client.collections.get(collection_name)
        else:
            raise
    return collection


@dataclass
class Metadata:
    set_number: int
    file_name: str

    def __repr__(self):
        return str(self.set_number) + ":" + self.file_name


def add_to_batch(
    batch: weaviate.collections.BatchCollection,
    uuid: weaviate.types.UUID,
    metadata: Metadata,
    vector: torch.Tensor,
):
    uuid = generate_uuid5(metadata)

    batch.add_object(
        properties=asdict(metadata),
        vector={
            "embedding": get_vector(vector),  # type: ignore
        },
        uuid=uuid,
    )


def embed_cuhk_dataset(
    client: weaviate.WeaviateClient, input_folder: Path, stops_file: Path, collection_name: str
):
    # Retrieving sets
    sets = get_sets(input_folder, stops_file)
    n_images = sum(len(sett) for sett in sets.values())
    logger.info("Retrieved %s sets and %s images", len(sets), n_images)

    # Loading the CLIP model
    # Get the torch device
    device = torch_device()

    # Load the model into memory
    model, preprocess, _ = load_inference_clip_model(CUHK_Apple_Collection.model_config, device)

    # Ingesting
    collection = create_or_get_collection(client, collection_name)

    with collection.batch.dynamic() as batch:
        for i, basenames in tqdm(sets.items(), total=len(sets)):
            for basename in basenames:
                metadata = Metadata(i, basename)
                uuid = generate_uuid5(metadata)
                if not collection.data.exists(uuid):
                    add_to_batch(
                        batch,
                        uuid,
                        metadata,
                        encode_image(input_folder, basename, device, model, preprocess),
                    )
            if batch.number_errors > 10:
                logger.error("Batch import stopped due to excessive errors.")
                break

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        logger.error(f"Number of failed imports: {len(failed_objects)}")
        logger.error(f"First failed object: {failed_objects[0]}")


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    INPUT_FOLDER = Path("./data/CUHK-PEDES/out")
    STOPS_FILE = Path("./scripts/cuhk_embeddings/cuhk_stops.txt")
    COLLECTION_NAME = CUHK_Apple_Collection.name
    with WeaviateClientEnsureReady() as client:
        embed_cuhk_dataset(client, INPUT_FOLDER, STOPS_FILE, COLLECTION_NAME)
