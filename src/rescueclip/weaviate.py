import logging.config

import weaviate


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

    def get_client(self):
        """Returns a ready Weaviate client instance."""
        if self.client is None:
            self.client = weaviate.connect_to_local()

        if self.client.is_ready():
            logger.info("Weaviate is ready")
            return self.client
        else:
            self.client.close()
            logger.error("ERROR: Weaviate is not ready!")
            raise ConnectionError("Weaviate is not ready")
