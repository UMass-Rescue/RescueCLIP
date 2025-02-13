import json
import logging
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

type SetNumToImagesMap = dict[int, list[str]]
type ImageToSetNumMap = dict[str, int]
type FileToHashesMap = dict[str, np.ndarray | np.floating]


@dataclass
class Metadata:
    set_number: int
    file_name: str

    def __repr__(self):
        return str(self.set_number) + ":" + self.file_name


def get_sets(in_folder: str | Path, stops_file: str | Path, debug=False) -> SetNumToImagesMap:
    """
    Returns a dictionary mapping set number to a list of file base names.
    """
    sets = defaultdict(list)

    with open(stops_file) as fd:
        stops_files = [line.rstrip() for line in fd.readlines()]

    next_stop = stops_files.pop(0)
    file_names = sorted(os.listdir(in_folder))

    set_num = 0
    for file_name in file_names:
        if debug:
            print("\tfile:", file_name)
        sets[set_num].append(file_name)
        if file_name == next_stop:
            if len(stops_files) == 0:
                break
            set_num += 1
            next_stop = stops_files.pop(0)
            if debug:
                print("stop:", next_stop)

    # stop failure detection
    sizes = sorted(
        {(set_id, tuple(fls), len(fls)) for set_id, fls in sets.items()},
        key=lambda x: x[2],
        reverse=True,
    )
    if sizes[0][2] > 4:
        # warnings.warn("Stop failure detected.", RuntimeWarning)
        set_id = sizes[0][0]
        n_fls = sizes[0][2]
        file_basenames = sizes[0][1]
        files = "\t\t\n".join(str(in_folder / Path(bn)) for bn in file_basenames)
        raise RuntimeError(f"Stop failure detected. Stops: {set_id} has {n_fls} files.\n\tFiles: {files}")

    return sets


def keep_sets_containing_n_images(sets: SetNumToImagesMap, n=4) -> SetNumToImagesMap:
    sets = {set_id: set_images for set_id, set_images in sets.items() if len(set_images) == n}
    n_images = sum(len(sett) for sett in sets.values())
    logger.info("After filtering, using %s sets and %s images", len(sets), n_images)
    return sets


def get_sets_new(in_folder, meta_file):
    with open(meta_file) as fd:
        meta_list = json.load(fd)

    sets = defaultdict(list)
    for meta_map in meta_list:
        set_num = int(meta_map["id"]) - 1
        file_name = os.path.basename(meta_map["file_path"])
        sets[set_num].append(file_name)

    return sets


def get_one_random_image_per_set(sets: SetNumToImagesMap):
    images_to_remove: list[Metadata] = []
    for set_number, file_names in sets.items():
        if len(file_names) > 1:
            images_to_remove.append(Metadata(set_number, np.random.choice(file_names)))
    return images_to_remove


def get_pdna_hashes(hashes_file: Path, include_only: Optional[set[str]] = None) -> FileToHashesMap:
    hashes = pd.read_csv(hashes_file, header=None)
    filename_to_hash_map: FileToHashesMap = {}
    for idx, row in hashes.iterrows():
        filename = re.sub(r"\\", "/", row[0])
        filename = Path(filename).name
        hsh = np.array([int(num) for num in row[1].split(",")])
        if include_only is not None and filename not in include_only:
            continue
        filename_to_hash_map[filename] = hsh
    return filename_to_hash_map
