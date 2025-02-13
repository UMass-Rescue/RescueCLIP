import logging.config
from dataclasses import dataclass
from hashlib import md5

import torch
import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5, get_vector

from rescueclip.logging_config import LOGGING_CONFIG
from rescueclip.weaviate import WeaviateClientEnsureReady

logger = logging.getLogger(__name__)

COLLECTION_NAME = "Images"


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


@dataclass
class Metadata:
    file_name: str

    def __repr__(self):
        return self.file_name


def insert_or_ignore_vector(
    collection: weaviate.collections.Collection,
    metadata: Metadata,
    vector: torch.Tensor,
):
    uuid = generate_uuid5(metadata)
    try:
        collection.data.insert(
            properties={
                "file_name": metadata.file_name,
            },
            vector={
                "embedding": get_vector(vector),  # type: ignore
            },
            uuid=uuid,
        )
        logger.info("Inserted new vector with uuid: %s", uuid)
    except weaviate.exceptions.UnexpectedStatusCodeError as e:
        if e.status_code == 422 and "already exists" in e.message:
            logger.info("Image already exists. Skipping creation.")
        else:
            raise

    return uuid


def main(client: weaviate.WeaviateClient):
    collection = create_or_get_collection(client, COLLECTION_NAME)

    logger.info("TEST 1: Trying to insert a single vector...")

    file_name1 = "sample_file_name.txt"
    vector1 = torch.rand(4)
    uuid = insert_or_ignore_vector(collection, Metadata(file_name1), vector1)
    data_object = collection.query.fetch_object_by_id(uuid, include_vector=True)
    logger.info("Retrieved vector: %s\n", data_object)

    logger.info("TEST 2: Trying to insert another vector and perform a query...")

    file_name2 = "sample_file_name2.txt"
    vector2 = torch.rand(4)
    insert_or_ignore_vector(collection, Metadata(file_name2), vector2)

    data_object = collection.query.fetch_objects(
        filters=Filter.by_property("file_name").not_equal(file_name1), include_vector=True
    ).objects[0]

    logger.info("\n")
    logger.info("Retrieved vector: %s\n", data_object)


if __name__ == "__main__":
    logging.config.dictConfig(LOGGING_CONFIG)
    with WeaviateClientEnsureReady() as client:
        main(client)
