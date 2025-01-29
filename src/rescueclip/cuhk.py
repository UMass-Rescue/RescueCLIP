import json
import os
import warnings
from collections import defaultdict
from pathlib import Path


def get_sets(in_folder: str | Path, stops_file: str | Path, debug=False) -> dict[int, list[str]]:
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


def get_sets_new(in_folder, meta_file):
    with open(meta_file) as fd:
        meta_list = json.load(fd)

    sets = defaultdict(list)
    for meta_map in meta_list:
        set_num = int(meta_map["id"]) - 1
        file_name = os.path.basename(meta_map["file_path"])
        sets[set_num].append(file_name)

    return sets
