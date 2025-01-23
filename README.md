conda create -n rescueCLIP python=3.12 -y
conda init # for bash, or conda init zsh
conda activate rescueCLIP
pip install -e .
pip install -e ".[dev]"

