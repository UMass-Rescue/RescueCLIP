import logging.config
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from rescueclip import cuhk
from rescueclip.logging_config import LOGGING_CONFIG

logger = logging.getLogger(__name__)


def euclidean_distance(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a - b)


def query_hash(
    filename_to_hash_map: dict[str, np.ndarray],
    test_image: cuhk.Metadata,
    top_k: int,
    set_lookup_map: cuhk.ImageToSetNumMap,
) -> list[int]:
    """
    Given a test image and a dictionary mapping file names to hashes, return a list of set numbers
    that the test image is potentially a member of limited by k.
    """
    test_hash = filename_to_hash_map[test_image.file_name]

    filename_to_hash_map_excluding_test_image = {
        filename: hash for filename, hash in filename_to_hash_map.items() if filename != test_image.file_name
    }
    filename_distance_pairs = sorted(
        filename_to_hash_map_excluding_test_image.items(),
        key=lambda x: float(euclidean_distance(x[1], test_hash)),
    )

    filename_distance_pairs = filename_distance_pairs[:top_k]

    return [set_lookup_map[filename_distance_pair[0]] for filename_distance_pair in filename_distance_pairs]


def main(INPUT_FOLDER, STOPS_FILE):
    hashes_file = Path("/scratch3/atharva/photodna/hashes.csv")
    top_ks = [1, 2, 5, 10, 15, 20]

    # Get the sets
    sets = cuhk.get_sets(INPUT_FOLDER, STOPS_FILE)
    sets = cuhk.keep_sets_containing_n_images(sets, 4)
    set_lookup_map = {filename: set_id for set_id, file_list in sets.items() for filename in file_list}

    # Get the PDNA hashes
    filename_to_hash_map = cuhk.get_pdna_hashes(hashes_file, include_only=set(set_lookup_map.keys()))
    assert len(filename_to_hash_map) == len(set_lookup_map), "Some images are missing hashes"
    logger.info(f"Loaded {len(filename_to_hash_map)} hashes")

    # Get the test images
    test_images = cuhk.get_one_random_image_per_set(sets)

    # Query the DB with the test set
    sum_found_map = defaultdict(int)  # map from top_k to sum_found

    max_top_k = max(top_ks)
    for test_image in tqdm(test_images):
        set_numbers = query_hash(filename_to_hash_map, test_image, max_top_k, set_lookup_map)

        for top_k in top_ks:
            top_k_capped = min(top_k, len(set_numbers))
            if any(test_image.set_number == set_number for set_number in set_numbers[:top_k_capped]):
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
            f.write("collection,model,top_k,accuracy\n")
        for top_k, accuracy in accuracies.items():
            f.write(f"CUHK_PDNA,PDNA,{top_k},{accuracy}\n")


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    load_dotenv()
    INPUT_FOLDER = Path(os.environ["CUHK_PEDES_DATASET"]) / "out"
    STOPS_FILE = Path("./scripts/cuhk_embeddings/cuhk_stops.txt")
    main(INPUT_FOLDER, STOPS_FILE)
