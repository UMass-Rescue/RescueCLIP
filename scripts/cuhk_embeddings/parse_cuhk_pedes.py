import argparse
import io
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--raw_folder", type=str, default=Path(os.environ["CUHK_PEDES_DATASET"]) / "raw")
parser.add_argument("--out_folder", type=str, default=Path(os.environ["CUHK_PEDES_DATASET"]) / "out")
args = parser.parse_args()

raw_folder = args.raw_folder
out_folder = args.out_folder

os.makedirs(out_folder, exist_ok=True)

raw_files = os.listdir(raw_folder)
for raw_file in raw_files:
    raw_path = os.path.join(raw_folder, raw_file)
    df = pd.read_parquet(raw_path)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        image = Image.open(io.BytesIO(row["image"]["bytes"]))
        rgb_im = image.convert("RGB")
        base = row["image"]["path"].split(".")[0]
        try:
            rgb_im.save(os.path.join(out_folder, base + ".jpg"))
        except OSError as e:
            print("skipping:", base)
            print("cause: ", e, "\n")
