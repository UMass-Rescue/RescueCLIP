#!/bin/bash

set -eu

PERSISTENCE_DATA_PATH="/scratch3/atharva/workspace/programming/RescueCLIP/weaviate-data"

# Ask for confirmation
read -p "Is '$PERSISTENCE_DATA_PATH' the correct directory for the volume bind mount? (y/n): " CONFIRM

if [[ "$CONFIRM" != "y" ]]; then
    echo "Aborting."
    exit 1
fi

# Start Docker Compose
PERSISTENCE_DATA_PATH=$PERSISTENCE_DATA_PATH docker-compose up -d
