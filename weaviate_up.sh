#!/bin/bash

set -eu

PERSISTENCE_DATA_PATH="/scratch3/atharva/workspace/programming/RescueCLIP/weaviate-data"

# Ask for confirmation
read -p "Is '$PERSISTENCE_DATA_PATH' the correct directory for the volume bind mount? (y/n): " CONFIRM

if [[ "$CONFIRM" != "y" ]]; then
    echo "Aborting."
    exit 1
fi

# Up or down
read -p "Run docker-compose up or down? (up/down): " UP_DOWN

if [[ "$UP_DOWN" == "up" ]]; then
    CURRENT_UID=$(id -u):$(id -g) PERSISTENCE_DATA_PATH=$PERSISTENCE_DATA_PATH docker compose up -d
    exit 0
fi
if [[ "$UP_DOWN" == "down" ]]; then
    CURRENT_UID=$(id -u):$(id -g) PERSISTENCE_DATA_PATH=$PERSISTENCE_DATA_PATH docker compose down
    exit 0
fi

echo "Invalid option $UP_DOWN"
exit 1
