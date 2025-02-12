import logging.config
import os

from dotenv import load_dotenv

from rescueclip.logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
import shutil
from pathlib import Path

import numpy as np
import weaviate
from tqdm import tqdm
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5

from rescueclip import cuhk
from rescueclip.open_clip import (
    CollectionConfig,
    CUHK_Apple_Collection,
    CUHK_laion_CLIP_ViT_bigG_14_laion2B_39B_b160k_Collection,
    CUHK_ViT_B_32_Collection,
)
from rescueclip.weaviate import WeaviateClientEnsureReady

from .embed_cuhk import Metadata, embed_cuhk_dataset


def delete_backup(backup_id: str):
    try:
        shutil.rmtree(Path("weaviate-data/backups") / backup_id)
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            raise


def experiment_with_top_k(
    top_k: int,
    collection_config: CollectionConfig,
    collection: weaviate.collections.Collection,
    test_images: list[Metadata],
    test_vectors: list[list[float]],
):
    logger.info(f"Running experiment for {top_k=}")

    # Query the DB with the test set
    sum_found = 0
    for image_metadata, test_vector in tqdm(zip(test_images, test_vectors), total=len(test_images)):
        results = collection.query.near_vector(
            near_vector=test_vector,
            limit=top_k,
            # return_metadata=MetadataQuery(distance=True, certainty=True),
        )
        if any(
            image_metadata.set_number == result.properties.get("set_number") for result in results.objects
        ):
            sum_found += 1

    # Save/Print results
    accuracy = sum_found / len(test_images)
    logger.info(f"Accuracy: {accuracy}")

    results_csv = Path("scripts/cuhk_embeddings/exclude_one_image_per_set_exp_results.csv")
    pre_existing = results_csv.exists()
    with open(results_csv, "a") as f:
        if not pre_existing:
            f.write("collection,model,top_k,accuracy\n")
        f.write(f"{collection_config.name},{collection_config.model_config.model_name},{top_k},{accuracy}\n")


def embed_dataset(
    client: weaviate.WeaviateClient, input_folder: Path, stops_file: Path, collection_config: CollectionConfig
):
    # Re-embed the entire database just in case -- this is fast if all images are present
    logger.info("Using collection %s", str(collection_config))
    logger.info(f"Re-embedding entire dataset {input_folder}")
    logger.info(f"This is fast if all images are present")
    embed_cuhk_dataset(client, input_folder, stops_file, collection_config)


def experiment(client: weaviate.WeaviateClient):
    INPUT_FOLDER = Path(os.environ["CUHK_PEDES_DATASET"]) / "out"
    STOPS_FILE = Path("./scripts/cuhk_embeddings/cuhk_stops.txt")
    collection_config = CUHK_Apple_Collection
    top_ks = [1, 2, 5, 10, 15, 20]

    # Embed the dataset and get the weaviate collection
    embed_dataset(client, INPUT_FOLDER, STOPS_FILE, collection_config)
    collection = client.collections.get(collection_config.name)

    # Remove one random image from each series
    sets = cuhk.get_sets(INPUT_FOLDER, STOPS_FILE)
    images_to_remove: list[Metadata] = []
    for set_number, file_names in sets.items():
        if len(file_names) > 1:
            images_to_remove.append(Metadata(set_number, np.random.choice(file_names)))

    images_to_remove_uuid = [generate_uuid5(image) for image in images_to_remove]
    images_to_remove_vectors = [
        collection.query.fetch_object_by_id(uuid, include_vector=True).vector["embedding"]
        for uuid in images_to_remove_uuid
    ]
    logger.info(f"Removing {len(images_to_remove_uuid)} vectors from the collection {collection_config.name}")
    result = collection.data.delete_many(where=Filter.by_id().contains_any(images_to_remove_uuid))

    assert result.successful == len(
        images_to_remove_uuid
    ), f"Failed to remove {len(images_to_remove_uuid)} images | {result.successful}"

    # The removed images become our test set
    test_images = images_to_remove
    test_vectors = images_to_remove_vectors

    for top_k in top_ks:
        experiment_with_top_k(top_k, collection_config, collection, test_images, test_vectors)

    embed_dataset(client, INPUT_FOLDER, STOPS_FILE, collection_config)


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    load_dotenv()

    top_ks = [1, 2, 5, 10, 15, 20]
    with WeaviateClientEnsureReady() as client:
        experiment(client)

"""
(top_k, Accuracy) for Apple Model on CUHK dataset
1, 31.13377324535093
2, 55.42891421715657
5, 67
10, 72
15, 77
20, 81
"""
