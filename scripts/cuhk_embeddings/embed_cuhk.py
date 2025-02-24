import logging.config
import os
from dataclasses import asdict
from pathlib import Path

import torch
import weaviate
from dotenv import load_dotenv
from tqdm import tqdm
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
)
from weaviate.util import generate_uuid5, get_vector

from rescueclip import cuhk
from rescueclip.cuhk import Metadata, get_sets_new, keep_sets_containing_n_images
from rescueclip.logging_config import LOGGING_CONFIG
from rescueclip.ml_model import (
    CollectionConfig,
    CUHK_PDNA_Collection,
    LIPModelProvider,
    encode_image,
    load_embedding_model,
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
                Property(name="file_name", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
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
    client: weaviate.WeaviateClient, input_folder: Path, stops_file: Path, collection_config: CollectionConfig
):
    # Retrieving sets
    sets = get_sets_new(input_folder, stops_file)
    n_images = sum(len(sett) for sett in sets.values())
    logger.info("Retrieved %s sets and %s images", len(sets), n_images)

    # Filter: keep sets with exactly 4 images
    sets = keep_sets_containing_n_images(sets, 4)
    if collection_config.model_config.provider == LIPModelProvider.PDNA:
        hashes_file = Path(os.environ["PDNA_HASHES_FILE"])
        set_lookup_map = {filename: set_id for set_id, file_list in sets.items() for filename in file_list}
        filename_to_hash_map, missing_filenames = cuhk.get_pdna_hashes(
            hashes_file, include_only=set(set_lookup_map.keys())
        )
        sets = cuhk.eliminate_sets_containing_files(filename_to_hash_map, missing_filenames, sets)
        set_lookup_map = {filename: set_id for set_id, file_list in sets.items() for filename in file_list}

    # Loading the CLIP model
    # Get the torch device
    device = torch_device()

    # Load the model into memory
    m = load_embedding_model(collection_config.model_config, device)

    # Ingesting
    collection = create_or_get_collection(client, collection_config.name)

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
                        encode_image(input_folder, basename, device, m),
                    )
            if batch.number_errors > 10:
                logger.error("Batch import stopped due to excessive errors.")
                break

    # Unload the model
    logger.info("Unloading model by empyting torch cache")
    del m
    torch.cuda.empty_cache()

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        logger.error(f"Number of failed imports: {len(failed_objects)}")
        logger.error(f"First failed object: {failed_objects[0]}")


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    load_dotenv()
    INPUT_FOLDER = Path(os.environ["CUHK_PEDES_DATASET"]) / "out"
    STOPS_FILE = Path("/scratch3/gbiss/images/CUHK-PEDES-OFFICIAL/caption_all.json")
    collection_config = CUHK_PDNA_Collection
    with WeaviateClientEnsureReady() as client:
        embed_cuhk_dataset(client, INPUT_FOLDER, STOPS_FILE, collection_config)
