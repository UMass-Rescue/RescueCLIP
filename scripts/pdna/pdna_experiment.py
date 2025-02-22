import argparse
import logging.config
import os
from collections import defaultdict
from html import parser
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from rescueclip import cuhk
from rescueclip.logging_config import LOGGING_CONFIG

logger = logging.getLogger(__name__)


def euclidean_distance(a: np.ndarray | np.floating, b: np.ndarray | np.floating) -> np.floating:
    return np.linalg.norm(a - b)


def transform_hash(arr: np.ndarray) -> np.float64:
    """
    Convert a 1D numpy array of length 144 with values in [0, 255]
    into a normalized floating point value between 0 and 1 by:

      1. Interpreting the array as a 144-byte sequence.
      2. Converting that sequence into a large integer (big-endian).
      3. Normalizing by dividing by the maximum possible 144-byte value.

    Parameters:
        arr (np.ndarray): 1D array of length 144 with integer values (0-255).

    Returns:
        np.float64: A normalized value between 0 and 1.
    """
    if arr.shape != (144,):
        raise ValueError("Input array must be a 1D array of length 144.")

    # Ensure the array is in the uint8 format.
    arr_uint8 = arr.astype(np.uint8)

    # Convert the array to bytes.
    byte_data = arr_uint8.tobytes()

    # Convert the byte data to a large integer using big-endian byte order.
    int_val = int.from_bytes(byte_data, byteorder="big")

    # Maximum possible value for a 144-byte integer.
    max_val = (1 << (8 * 144)) - 1

    # Normalize the integer to a float in the interval [0, 1].
    norm_val = int_val / max_val

    return np.float64(norm_val)


def transform_pdna_hashes(filename_to_hash_map: cuhk.FileToHashesMap) -> cuhk.FileToHashesMap:
    new_filename_to_hash_map: cuhk.FileToHashesMap = {}
    for filename, hash in filename_to_hash_map.items():
        assert isinstance(hash, np.ndarray)
        new_filename_to_hash_map[filename] = transform_hash(hash)

    return new_filename_to_hash_map


def query_hash(
    filename_to_hash_map: cuhk.FileToHashesMap,
    test_image: cuhk.Metadata,
    top_k: int,
    set_lookup_map: cuhk.ImageToSetNumMap,
) -> list[tuple[cuhk.Metadata, float]]:
    """
    Given a test image and a dictionary mapping file names to hashes, return a list of set numbers
    that the test image is potentially a member of limited by k.
    """
    test_hash = filename_to_hash_map[test_image.file_name]

    filename_to_hash_map_excluding_test_image = {
        filename: hash for filename, hash in filename_to_hash_map.items() if filename != test_image.file_name
    }
    filename_distance_pairs = sorted(
        [
            (filename, float(euclidean_distance(hash, test_hash)))
            for filename, hash in filename_to_hash_map_excluding_test_image.items()
        ],
        key=lambda x: x[1],
    )

    filename_distance_pairs = filename_distance_pairs[:top_k]

    return [
        (
            cuhk.Metadata(
                file_name=filename_distance_pair[0],
                set_number=set_lookup_map[filename_distance_pair[0]],
            ),
            filename_distance_pair[1],
        )
        for filename_distance_pair in filename_distance_pairs
    ]


def main(INPUT_FOLDER, STOPS_FILE, apply_pdna_transformation=False):
    hashes_file = Path("/scratch3/atharva/photodna/hashes.csv")
    top_ks = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]

    # Get the sets
    sets = cuhk.get_sets_new(INPUT_FOLDER, STOPS_FILE)
    sets = cuhk.keep_sets_containing_n_images(sets, 4)
    set_lookup_map = {filename: set_id for set_id, file_list in sets.items() for filename in file_list}

    # Get the PDNA hashes
    filename_to_hash_map, missing_filenames = cuhk.get_pdna_hashes(
        hashes_file, include_only=set(set_lookup_map.keys())
    )
    sets = cuhk.eliminate_sets_containing_files(filename_to_hash_map, missing_filenames, sets)
    set_lookup_map = {filename: set_id for set_id, file_list in sets.items() for filename in file_list}

    logger.info(f"Missing filenames length: {len(missing_filenames)}")
    assert len(filename_to_hash_map) == len(
        set_lookup_map
    ), f"Some images are missing hashes, {len(filename_to_hash_map)} != {len(set_lookup_map)}"
    logger.info(f"Loaded {len(filename_to_hash_map)} hashes")

    # (Optional) Apply the PDNA transformation to the hashes
    if apply_pdna_transformation:
        filename_to_hash_map = transform_pdna_hashes(filename_to_hash_map)
        logger.info(f"NOTE: PDNA transformation applied to hashes")

    # Get the test images
    test_images = cuhk.get_one_random_image_per_set(sets)

    # Query the DB with the test set
    sum_found_map = defaultdict(int)  # map from top_k to sum_found

    max_top_k = max(top_ks)
    for test_image in tqdm(test_images):
        results = query_hash(filename_to_hash_map, test_image, max_top_k, set_lookup_map)

        for top_k in top_ks:
            top_k_capped = min(top_k, len(results))
            if any(
                test_image.set_number == result_metadata.set_number
                for result_metadata, distance in results[:top_k_capped]
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
            sim_metric = "custom_metric" if apply_pdna_transformation else "euclidean"
            f.write(f"{sim_metric},CUHK_PDNA,PDNA,{top_k},{accuracy}\n")


def tests():
    test_pdna_hash_ff_00_ff = np.zeros(144)
    test_pdna_hash_ff_00_ff[0] = 255
    test_pdna_hash_ff_00_ff[143] = 255
    MAX_PDNA_VAL = 0xFF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF

    def test1():
        float_val = transform_hash(test_pdna_hash_ff_00_ff)

        assert float_val == 0xFF_00_00_00_00_00_00_00_00_00_00_FF / MAX_PDNA_VAL, "Transformation failed."

    def test2():
        test_pdna_hash_all_ones = np.ones(144)
        filename_to_hash_map: cuhk.FileToHashesMap = {
            "sample_file_name1.txt": test_pdna_hash_all_ones,
            "sample_file_name2.txt": test_pdna_hash_all_ones,
            "sample_file_name3.txt": test_pdna_hash_ff_00_ff,
        }
        set_lookup_map: cuhk.ImageToSetNumMap = {
            "sample_file_name1.txt": 0,
            "sample_file_name2.txt": 1,
            "sample_file_name3.txt": 2,
        }

        test_image = cuhk.Metadata(file_name="sample_file_name1.txt", set_number=0)

        # Applying the transformation to the hashes
        filename_to_hash_map = transform_pdna_hashes(filename_to_hash_map)

        results = query_hash(filename_to_hash_map, test_image, 2, set_lookup_map)

        assert results[0][0].file_name == "sample_file_name2.txt", "Query failed."
        assert results[0][1] == 0.0, "Distance computation failed."
        assert results[1][0].file_name == "sample_file_name3.txt", "Query 2 failed."

        assert isinstance(
            test_pdna_hash_ff_00_ff, np.ndarray
        ), f"hash3 is not a numpy array. {type(test_pdna_hash_ff_00_ff)}"
        dist = euclidean_distance(
            transform_hash(test_pdna_hash_all_ones), transform_hash(test_pdna_hash_ff_00_ff)
        )
        assert results[1][1] == dist, f"Distance computation 2 failed. Expected {dist}, got {results[1][1]}"

    test1()
    test2()

    logger.info(f"Tests passed.")


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    load_dotenv()
    INPUT_FOLDER = Path(os.environ["CUHK_PEDES_DATASET"]) / "out"
    STOPS_FILE = Path("/scratch3/gbiss/images/CUHK-PEDES-OFFICIAL/caption_all.json")
    tests()

    parser = argparse.ArgumentParser()
    parser.add_argument("--apply_pdna_transformation", action="store_true")
    args = parser.parse_args()

    apply_pdna_transformation = args.apply_pdna_transformation
    main(INPUT_FOLDER, STOPS_FILE, apply_pdna_transformation)
