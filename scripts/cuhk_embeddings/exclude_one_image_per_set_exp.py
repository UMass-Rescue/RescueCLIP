import argparse
import logging.config
import os
from collections import defaultdict

from dotenv import load_dotenv

from rescueclip.logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
from pathlib import Path

import weaviate
from tqdm import tqdm
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5

from rescueclip import cuhk
from rescueclip.ml_model import (
    CollectionConfig,
    CUHK_Apple_Collection,
    CUHK_Google_Siglip_Base_Patch16_224_Collection,
    CUHK_Google_Siglip_SO400M_Patch14_384_Collection,
    CUHK_laion_CLIP_ViT_bigG_14_laion2B_39B_b160k_Collection,
    CUHK_MetaCLIP_ViT_bigG_14_quickgelu_224_Collection,
    CUHK_ViT_B_32_Collection,
)
from rescueclip.weaviate import WeaviateClientEnsureReady

from .embed_cuhk import Metadata, embed_cuhk_dataset


def experiment_with_top_ks(
    top_ks: list[int],
    collection_config: CollectionConfig,
    collection: weaviate.collections.Collection,
    test_images: list[Metadata],
    test_vectors: list[list[float]],
):
    logger.info(f"Running experiment for {top_ks=}")

    # Query the DB with the test set
    sum_found_map = defaultdict(int)  # map from top_k to sum_found

    max_top_k = max(top_ks)
    for image_metadata, test_vector in tqdm(zip(test_images, test_vectors), total=len(test_images)):
        results = collection.query.near_vector(
            near_vector=test_vector,
            limit=max_top_k,
            filters=Filter.by_property("set_number").not_equal(image_metadata.set_number)
            | Filter.by_property("file_name").not_equal(image_metadata.file_name),
            return_metadata=MetadataQuery(distance=True, certainty=True),
        )
        # Assert that the results are ordered by distance
        assert results.objects == sorted(results.objects, key=lambda x: x.metadata.distance), "Results are not ordered by distance"  # type: ignore

        for top_k in top_ks:
            top_k_capped = min(top_k, len(results.objects))
            if any(
                image_metadata.set_number == result.properties.get("set_number")
                for result in results.objects[:top_k_capped]
            ):
                sum_found_map[top_k] += 1

    # Save/Print results
    accuracies = {top_k: sum_found_map[top_k] / len(test_images) for top_k in top_ks}
    logger.info(
        f"Accuracies: \n\t%s",
        str("\n\t".join(f"{top_k}: {accuracy}" for top_k, accuracy in accuracies.items())),
    )

    results_csv = Path("scripts/cuhk_embeddings/exclude_one_image_per_set_exp_results.csv")
    pre_existing = results_csv.exists()
    with open(results_csv, "a") as f:
        if not pre_existing:
            f.write("similarity_metric,collection,model,top_k,accuracy\n")
        for top_k, accuracy in accuracies.items():
            f.write(
                f"cosine,{collection_config.name},{collection_config.model_config.model_name},{top_k},{accuracy}\n"
            )


def embed_dataset(
    client: weaviate.WeaviateClient, input_folder: Path, stops_file: Path, collection_config: CollectionConfig
):
    # Re-embed the entire database just in case -- this is fast if all images are present
    logger.info("Using collection %s", str(collection_config))
    logger.info(f"Re-embedding entire dataset {input_folder}")
    logger.info(f"This is fast if all images are present")
    embed_cuhk_dataset(client, input_folder, stops_file, collection_config)


def experiment(client: weaviate.WeaviateClient, collection_config: CollectionConfig):
    INPUT_FOLDER = Path(os.environ["CUHK_PEDES_DATASET"]) / "out"
    STOPS_FILE = Path("/scratch3/gbiss/images/CUHK-PEDES-OFFICIAL/caption_all.json")
    top_ks = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]

    # Embed the dataset and get the weaviate collection
    embed_dataset(client, INPUT_FOLDER, STOPS_FILE, collection_config)
    collection = client.collections.get(collection_config.name)

    # Remove one random image from each series
    sets = cuhk.get_sets_new(INPUT_FOLDER, STOPS_FILE)
    sets = cuhk.keep_sets_containing_n_images(sets, 4)

    # Some images become our test set. We must be sure to exclude them when querying the DB.
    test_images = cuhk.get_one_random_image_per_set(sets)
    images_to_remove_uuid = [generate_uuid5(image) for image in test_images]
    test_vectors = [
        collection.query.fetch_object_by_id(uuid, include_vector=True).vector["embedding"]
        for uuid in images_to_remove_uuid
    ]

    experiment_with_top_ks(top_ks, collection_config, collection, test_images, test_vectors)


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_config", type=str)
    args = parser.parse_args()

    if args.collection_config == "cuhk_apple":
        collection_config = CUHK_Apple_Collection
    elif args.collection_config == "cuhk_laion":
        collection_config = CUHK_laion_CLIP_ViT_bigG_14_laion2B_39B_b160k_Collection
    elif args.collection_config == "cuhk_vit_b_32":
        collection_config = CUHK_ViT_B_32_Collection
    elif args.collection_config == "cuhk_google_siglip_base_patch16_224":
        collection_config = CUHK_Google_Siglip_Base_Patch16_224_Collection
    elif args.collection_config == "cuhk_google_siglip_so400m_patch14_384":
        collection_config = CUHK_Google_Siglip_SO400M_Patch14_384_Collection
    elif args.collection_config == "cuhk_metaclip_vit_bigg14_quickgelu_224":
        collection_config = CUHK_MetaCLIP_ViT_bigG_14_quickgelu_224_Collection
    else:
        raise ValueError(f"Invalid collection_config {args.collection_config}")

    with WeaviateClientEnsureReady() as client:
        experiment(client, collection_config)
