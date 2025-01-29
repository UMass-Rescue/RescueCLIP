import logging.config

from rescueclip.logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
import shutil
from pathlib import Path

import numpy as np
import weaviate
from tqdm import tqdm
from weaviate.backup import BackupStorage
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5

from rescueclip import cuhk
from rescueclip.open_clip import CUHK_Apple_Collection
from rescueclip.weaviate import WeaviateClientEnsureReady

from .embed_cuhk import Metadata, embed_cuhk_dataset


def delete_backup(backup_id: str):
    try:
        shutil.rmtree(Path("weaviate-data/backups") / backup_id)
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            raise


def experiment(client: weaviate.WeaviateClient):
    INPUT_FOLDER = Path("./data/CUHK-PEDES/out")
    STOPS_FILE = Path("./scripts/cuhk_embeddings/cuhk_stops.txt")
    COLLECTION = CUHK_Apple_Collection
    TOP_K = 20

    # Re-embed the entire database just in case -- this is fast if all images are present
    logger.info(f"Re-embedding entire dataset {INPUT_FOLDER}")
    embed_cuhk_dataset(client, INPUT_FOLDER, STOPS_FILE, COLLECTION.name)

    # Make a copy of the collection and use it for the test
    collection = client.collections.get(COLLECTION.name)
    backup_id = "backup-for-experiment"
    logger.info(f"Making a backup of the current database state with backup ID {backup_id}")

    delete_backup(backup_id)

    status = collection.backup.create(
        backup_id=backup_id,
        backend=BackupStorage.FILESYSTEM,
        wait_for_completion=True,
    )
    assert status.error is None, "Failed to make a backup of collection %s" % COLLECTION.name
    logger.info(f"Completed backup: {status.status}")

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
    logger.info(f"Removing {len(images_to_remove_uuid)} vectors from the collection {COLLECTION.name}")
    result = collection.data.delete_many(where=Filter.by_id().contains_any(images_to_remove_uuid))

    assert result.successful == len(
        images_to_remove_uuid
    ), f"Failed to remove {len(images_to_remove_uuid)} images | {result.successful}"

    # The removed images become our test set
    test_images = images_to_remove
    test_vectors = images_to_remove_vectors

    sum_found = 0
    for image_metadata, test_vector in tqdm(zip(test_images, test_vectors), total=len(test_images)):
        results = collection.query.near_vector(
            near_vector=test_vector,
            limit=TOP_K,
            # return_metadata=MetadataQuery(distance=True, certainty=True),
        )
        if any(
            image_metadata.set_number == result.properties.get("set_number") for result in results.objects
        ):
            sum_found += 1

    logger.info(f"Accuracy: {sum_found/len(test_images)}")

    logger.info(f"Restoring the backup with backup ID {backup_id}")
    client.collections.delete(COLLECTION.name)
    status = collection.backup.restore(
        backup_id=backup_id, backend=BackupStorage.FILESYSTEM, wait_for_completion=True
    )
    assert status.error is None, "Failed to restore a backup of collection %s" % COLLECTION.name
    logger.info("Restored backup")

    delete_backup(backup_id)


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    with WeaviateClientEnsureReady() as client:
        experiment(client)

"""
(TOP_K, Accuracy) for Apple Model on CUHK dataset
5, 67
10, 72
15, 77,
20, 81
"""
