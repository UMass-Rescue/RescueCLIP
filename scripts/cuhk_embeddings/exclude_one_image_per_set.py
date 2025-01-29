from pathlib import Path

import numpy as np
import weaviate
from tqdm import tqdm
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5

from rescueclip import cuhk
from rescueclip.open_clip import apple_DFN5B_CLIP_ViT_H_14_384
from rescueclip.weaviate import WeaviateClientEnsureReady

from .embed_cuhk import Metadata, embed_cuhk_dataset


def experiment(client: weaviate.WeaviateClient):
    INPUT_FOLDER = Path("./data/CUHK-PEDES/out")
    STOPS_FILE = Path("./scripts/cuhk_embeddings/cuhk_stops.txt")
    COLLECTION_NAME = apple_DFN5B_CLIP_ViT_H_14_384.weaviate_collection_name
    TOP_K = 5

    # Re-embed the entire database just in case -- this is fast
    embed_cuhk_dataset(client, INPUT_FOLDER, STOPS_FILE, COLLECTION_NAME)

    # Remove one random image from each series
    sets = cuhk.get_sets(INPUT_FOLDER, STOPS_FILE)
    images_to_remove: list[Metadata] = []
    for set_number, file_names in sets.items():
        if len(file_names) > 1:
            images_to_remove.append(Metadata(set_number, np.random.choice(file_names)))

    collection = client.collections.get(COLLECTION_NAME)
    images_to_remove_uuid = [generate_uuid5(image) for image in images_to_remove]
    images_to_remove_vectors = [
        collection.query.fetch_object_by_id(uuid, include_vector=True).vector["embedding"]
        for uuid in images_to_remove_uuid
    ]
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

    print(f"Accuracy: {sum_found/len(test_images)}")


if __name__ == "__main__":
    with WeaviateClientEnsureReady() as client:
        experiment(client)
