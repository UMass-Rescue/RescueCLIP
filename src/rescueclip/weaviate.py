import logging.config
from hashlib import md5

import torch
import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.util import generate_uuid5, get_vector

from rescueclip.logging_config import LOGGING_CONFIG

logger = logging.getLogger(__name__)


class WeaviateClientEnsureReady:
    """
    Usage:
    with WeaviateClientEnsureReady() as client:
        main_function(client)
    """
    def __init__(self):
        self.client = None

    def __enter__(self):
        self.client = weaviate.connect_to_local()
        if self.client.is_ready():
            logger.info("Weaviate is ready")
            return self.client
        else:
            self.client.close()
            logger.error("ERROR: Weaviate is not ready!")
            raise ConnectionError("Weaviate is not ready")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed")
