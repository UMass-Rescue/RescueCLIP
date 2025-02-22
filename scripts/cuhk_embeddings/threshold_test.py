import argparse
from email.mime import base
import logging.config
import os
from collections import defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv
import optuna
from optuna.storages.journal import JournalFileBackend

from scipy.spatial.distance import cdist

from rescueclip.logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Literal

import numpy as np
import weaviate
from tqdm import tqdm

from rescueclip import cuhk
from rescueclip.ml_model import (
    CollectionConfig,
    CUHK_Apple_Collection,
    CUHK_ViT_B_32_Collection,
)
from rescueclip.weaviate import WeaviateClientEnsureReady

QUERY_MAXIMUM_RESULTS = 200_000


def create_kv_database(results):
    kv_database = defaultdict(dict)
    for obj in results.objects:
        kv_database[obj.properties["set_number"]][obj.properties["file_name"]] = obj.vector["embedding"]
    return kv_database


@dataclass
class ConfusionMatrix:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, *, tp: int = 0, tn: int = 0, fp: int = 0, fn: int = 0):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def as_array(self):
        # Returns a 2x2 array: [[tn, fp], [fn, tp]]
        return [[self.tn, self.fp], [self.fn, self.tp]]

    def precision(self):
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    def f1(self):
        prec = self.precision()
        rec = self.recall()
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)

    def __str__(self):
        return f"TP: {self.tp}, TN: {self.tn}, FP: {self.fp}, FN: {self.fn}"


def get_set_number_distance_pairs_of_neighbors_within_t_ordered_by_t(
    vector: np.ndarray,
    t: float,
    X: np.ndarray,
    y_labels: np.ndarray,
    distance_metric: Literal["cosine"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the set numbers of the closest neighboring images within t, ordered by t.

    t in [0, 1]
    """
    assert X.shape[0] == len(y_labels), "Expected X.shape[0] == len(y_labels)"
    assert len(vector) == X.shape[1], "Expected len(vector) == X.shape[1]"

    if distance_metric == "cosine":
        # Get the distances between the test image and all the images in X
        distances = cdist(X, [vector], metric="cosine")
        assert distances.shape == (
            X.shape[0],
            1,
        ), f"Expected distances.shape == (X.shape[0], 1), got {distances.shape}"
        distances = distances.reshape(-1)
        distances /= 2
    else:
        raise NotImplementedError(f"distance_metric {distance_metric} not implemented")

    assert distances.shape == (
        X.shape[0],
    ), f"Expected distances.shape == (X.shape[0], ), expected {(X.shape[0], )}, got {distances.shape}"

    # Get the indices of the closest images within t
    indices = np.where(distances <= t)[0]
    indices = indices[np.argsort(distances[indices])]

    sorted_distances = distances[indices]
    sorted_y_lables = y_labels[indices]

    return sorted_y_lables, sorted_distances


def threshold_test(
    x: np.ndarray,
    x_set_num: int,
    X_train: np.ndarray,
    y_train_set_labels: np.ndarray,
    t: float,
    confusion_matrix: ConfusionMatrix,
    is_in_sample=True,
) -> None:
    # Get the neighbors within t
    neighboring_sets, distances = get_set_number_distance_pairs_of_neighbors_within_t_ordered_by_t(
        x, t, X_train, y_train_set_labels, "cosine"
    )
    assert len(neighboring_sets.shape) == 1, "Expected neighboring_sets to be a 1D array"

    if is_in_sample:
        assert len(neighboring_sets) >= 1, f"Expected at least one zero distance with the test image itself, because it is in the DB (expected set {x_set_num},\n first set in neighs {neighboring_sets[0]},\n distances {distances[:5]} \n vector: {x[:5]})a"
        if x_set_num == neighboring_sets[0]:
            neighboring_sets = neighboring_sets[1:]
        elif x_set_num == neighboring_sets[1]:
            neighboring_sets = np.delete(neighboring_sets, 1)
            logger.warning(f"Found two zero distances sets with the test image itself (expected set {x_set_num},\n first set in neighs {neighboring_sets[0]},\n distances {distances[:5]} \n vector: {x[:5]})")
        elif x_set_num == neighboring_sets[2]:
            neighboring_sets = np.delete(neighboring_sets, 2)
            logger.warning(f"Found two zero distances sets with the test image itself (expected set {x_set_num},\n first set in neighs {neighboring_sets[0]},\n distances {distances[:5]} \n vector: {x[:5]})")
        else:
            assert False, f"Expected the best distance set to be the test image itself (expected {x_set_num},\n got zero distances sets {neighboring_sets[0]},\n distances {distances[:5]} \n vector: {x[:5]})"

    if is_in_sample:
        if x_set_num in neighboring_sets:
            confusion_matrix.tp += 1
        else:
            confusion_matrix.fp += 1
    else:
        if len(neighboring_sets) == 0:
            confusion_matrix.tn += 1
        else:
            confusion_matrix.fn += 1


def experiment(client: weaviate.WeaviateClient, collection_config: CollectionConfig, threshold: float):
    INPUT_FOLDER = Path(os.environ["CUHK_PEDES_DATASET"]) / "out"
    STOPS_FILE = Path("/scratch3/gbiss/images/CUHK-PEDES-OFFICIAL/caption_all.json")
    collection = client.collections.get(collection_config.name)

    # Assertions
    number_of_objects: int = collection.aggregate.over_all(total_count=True).total_count  # type: ignore
    logger.info(f"Number of objects %s", number_of_objects)
    assert (
        number_of_objects <= QUERY_MAXIMUM_RESULTS
    ), "Ensure docker-compose.yml has QUERY_MAXIMUM_RESULTS to greater than 200_000 or the experiment's results may be inaccurate"

    # Train test split
    sets = cuhk.get_sets_new(INPUT_FOLDER, STOPS_FILE)
    sets = cuhk.keep_sets_containing_n_images(sets, 4)

    set_number_set_list_pairs = list(sets.items())
    np.random.shuffle(set_number_set_list_pairs)

    _in_sample_series = set_number_set_list_pairs[: len(set_number_set_list_pairs) // 2]
    in_sample_series = {set_num: file_names for set_num, file_names in _in_sample_series}
    _heldout_series = set_number_set_list_pairs[len(set_number_set_list_pairs) // 2 :]
    heldout_series = {set_num: file_names for set_num, file_names in _heldout_series}

    logger.info(f"Total series: {len(set_number_set_list_pairs)}")
    logger.info(f"In-sample series: {len(in_sample_series)}")
    logger.info(f"Held-out series: {len(heldout_series)}")

    # Creating a kv database
    logger.info("Retrieving the entire database into memory")
    results = collection.query.fetch_objects(
        limit=QUERY_MAXIMUM_RESULTS,
        include_vector=True,
        return_properties=True,
    )
    assert len(results.objects) == number_of_objects, "Expected the entire database to be retrieved"
    kv_database = create_kv_database(results)

    # Creating numpy arrays
    X = np.array([obj.vector["embedding"] for obj in results.objects])
    y_set_labels = np.array([obj.properties["set_number"] for obj in results.objects])
    X_indices_in_insample_series = np.array(
        [
            i
            for i, image_metadata in enumerate(results.objects)
            if image_metadata.properties["set_number"] not in heldout_series
        ]
    )
    X_train = X[X_indices_in_insample_series]
    y_train_set_labels = y_set_labels[X_indices_in_insample_series]

    logger.info(f"X.shape: {X.shape}")
    logger.info(f"y_set_labels.shape: {y_set_labels.shape}")
    logger.info(f"X_train.shape: {X_train.shape}")
    logger.info(f"y_train_set_labels.shape: {y_train_set_labels.shape}")

    # Running the test
    cm = ConfusionMatrix()

    for sample_series in [in_sample_series, heldout_series]:
        is_in_sample = sample_series == in_sample_series
        for sample_serie in tqdm(
            sample_series.items(),
            total=len(sample_series),
            desc=f"t={threshold} | In-sample: {is_in_sample}",
        ):
            set_num, images_fps = sample_serie
            for image_fp in images_fps:
                # Get the vector for image_fp
                vector = kv_database[set_num][image_fp]
                threshold_test(vector, set_num, X_train, y_train_set_labels, threshold, cm, is_in_sample)

    return cm

def optuna_objective(trial: optuna.Trial, collection_config: CollectionConfig, client: weaviate.WeaviateClient):
    threshold = trial.suggest_float("threshold", 0, 1)
    logger.info(f"Threshold: {threshold}")
    cm = experiment(client, collection_config, threshold)
    
    trial.set_user_attr("selected_threshold", threshold)
    trial.set_user_attr("f1_score", cm.f1())
    trial.set_user_attr("precision", cm.precision())
    trial.set_user_attr("recall", cm.recall())
    trial.set_user_attr("tp", cm.tp)
    trial.set_user_attr("tn", cm.tn)
    trial.set_user_attr("fp", cm.fp)
    trial.set_user_attr("fn", cm.fn)

    return cm.f1()

if __name__ == "__main__":
    load_dotenv()
    base_path = Path("scripts/cuhk_embeddings")
    results_csv_path = base_path / "threshold_test_results.csv"
    journal_backend = JournalFileBackend(str(base_path / "optuna_journal.log"))
    storage = optuna.storages.JournalStorage(journal_backend)
    N_TRIALS = 20

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_config", type=str)
    args = parser.parse_args()

    if args.collection_config == "cuhk_apple":
        collection_config = CUHK_Apple_Collection
    elif args.collection_config == "cuhk_vit_b_32":
        collection_config = CUHK_ViT_B_32_Collection
    else:
        raise ValueError(f"Invalid collection_config {args.collection_config}")
    
    study = optuna.create_study(direction="maximize", study_name="threshold_test", storage=storage, load_if_exists=True)

    with WeaviateClientEnsureReady() as client:
        logger.info(f"Optimizing using {N_TRIALS} trials...")
        study.optimize(lambda trial: optuna_objective(trial, collection_config, client), n_trials=N_TRIALS)
    
    for trial in study.trials:
        print(f"Trial {trial.number}: Threshold={trial.user_attrs['selected_threshold']}, F1={trial.user_attrs['f1_score']}")
    
    study.trials_dataframe().to_csv(results_csv_path)
