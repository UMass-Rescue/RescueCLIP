# Getting Started

```bash
conda create -n rescueCLIP python=3.12 -y
conda activate rescueCLIP
pip install -e .
pip install -e ".[dev]"
```

See [Makefile](./Makefile) for commands to run experiments.

# Weaviate

I'm currently using Weaviate as a vector database to store image embeddings. Inspect the [docker-compose.yml](./docker-compose.yml) file for configuration and check the "Weaviate" section in the Makefile
