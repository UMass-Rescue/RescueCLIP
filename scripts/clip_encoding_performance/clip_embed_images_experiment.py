import argparse
import os
import time
from pathlib import Path

import open_clip
import torch
import torchvision
from line_profiler import profile
from PIL import Image
from tqdm import tqdm

from rescueclip.open_clip import (
    ViT_B_32,
    apple_DFN5B_CLIP_ViT_H_14_384,
    load_inference_clip_model,
)

# from memory_profiler import profile




def is_valid_filename(file_name: str) -> bool:
    if file_name.startswith("."):
        print(f"Skipping {file_name} ... Reason: File name starts with a dot")
        return False
    allowed_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    if not any(file_name.endswith(ext) for ext in allowed_extensions):
        print(f"Skipping {file_name} ... Reason: File name does not end with an allowed extension")
        print(f"Allowed extensions: {allowed_extensions}")
        return False
    return True


def should_include_directory_entry(entry: os.DirEntry) -> bool:
    return entry.is_file() and is_valid_filename(entry.name)


def torch_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"

    print("Using device:", device)
    return device


@profile
def load_all_the_images_and_then_encode_them_together(
    files: list[str],
    path: str,
    device: str,
    model: open_clip.CLIP,
    preprocess: torchvision.transforms.Compose,
):
    print(f"Loading all the images first and then encoding images together")
    # Pre-allocate the image tensor
    assert len(preprocess.transforms) > 0, "Expected at least one transform"
    assert isinstance(
        preprocess.transforms[0], torchvision.transforms.transforms.Resize
    ), f"Expected Resize transform, got {type(preprocess.transforms[0])}"

    size = preprocess.transforms[0].size[0]
    x = torch.empty(len(files), 3, size, size, device=device)
    print(f"Pre-allocated image tensor: {x.shape}")

    # Load the images into the tensor
    for i, file in tqdm(enumerate(files), total=len(files)):
        pil_image = Image.open(os.path.join(path, file))
        image = preprocess(pil_image).unsqueeze(0).to(device)  # type: ignore
        pil_image.close()
        x[i] = image
    print(f"Loaded {len(files)} images into tensor")

    # Encode the images
    with torch.no_grad(), torch.amp.autocast(device):  # type: ignore
        images_features = model.encode_image(x)
        print(f"Image features: {images_features.shape}")

    return images_features


@profile
def load_each_image_and_encode_immediately(
    files: list[str],
    path: str,
    device: str,
    model: open_clip.CLIP,
    preprocess: torchvision.transforms.Compose,
):
    print(f"Loading each image and encoding immediately")
    # Pre-allocate the image embedding tensor
    shape = list(model.modules())[-1].normalized_shape[0]
    images_features = torch.empty(len(files), shape, device=device)
    print(f"Pre-allocated image embedding tensor: {images_features.shape}")

    # Encode the images
    with torch.no_grad(), torch.amp.autocast(device):  # type: ignore
        for i, file in tqdm(enumerate(files), total=len(files)):
            pil_image = Image.open(os.path.join(path, file))
            image = preprocess(pil_image).unsqueeze(0).to(device)  # type: ignore
            pil_image.close()
            images_features[i] = model.encode_image(image)
        print(f"Image features: {images_features.shape}")

    return images_features


def main():
    parser = argparse.ArgumentParser(description="Process images with specified parameters.")
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Number of files to process in each batch."
    )
    parser.add_argument(
        "--function",
        choices=[
            "load_all_the_images_and_then_encode_them_together",
            "load_each_image_and_encode_immediately",
        ],
        required=True,
        help="Function to use for image processing.",
    )
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    FUNCTION_ENV = args.function

    # Load the image file paths
    path = "data/training_set/dogs"  # contains 4000 images
    files = os.listdir(path)
    files = list(filter(is_valid_filename, files))
    files.sort()

    # Limit the number of files to BATCH_SIZE
    files = files[:BATCH_SIZE]
    print(f"Found {len(files)} files in {path}")

    # Get the torch device
    device = torch_device()

    # Load the model into memory
    model, preprocess, _ = load_inference_clip_model(apple_DFN5B_CLIP_ViT_H_14_384, device)

    # Choose function based on CLI arg
    if FUNCTION_ENV == "load_all_the_images_and_then_encode_them_together":
        fn = load_all_the_images_and_then_encode_them_together
    elif FUNCTION_ENV == "load_each_image_and_encode_immediately":
        fn = load_each_image_and_encode_immediately
    else:
        raise ValueError(
            f"Invalid function: {FUNCTION_ENV} | Possible values: 'load_all_the_images_and_then_encode_them_together', 'load_each_image_and_encode_immediately'"
        )

    # Run the function
    start = time.time()
    embeddings = fn(files, path, device, model, preprocess)
    end = time.time()
    print(f"Time taken for loop: {end - start:.2f}")
    print(f"Embeddings: {embeddings.shape}")

    # Save results to CSV
    results_csv = Path("./scripts/clip_encoding_performance/clip_embed_images_experiment.csv")
    pre_existing = results_csv.exists()
    with open(results_csv, "a") as f:
        if not pre_existing:
            f.write("function,batch_size,time_taken_secs\n")
        f.write(f"{FUNCTION_ENV},{BATCH_SIZE},{end - start}\n")


if __name__ == "__main__":
    main()