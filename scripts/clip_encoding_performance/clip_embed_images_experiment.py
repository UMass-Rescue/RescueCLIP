import argparse
import os
import time
from pathlib import Path

import open_clip
import torch
import torchvision
from dotenv import load_dotenv
from line_profiler import profile
from PIL import Image
from tqdm import tqdm

from rescueclip.ml_model import (
    CLIPModel,
    ViT_B_32,
    apple_DFN5B_CLIP_ViT_H_14_384,
    encode_image,
    load_inference_clip_model,
    torch_device,
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


@profile
def load_all_the_images_and_then_encode_them_together(
    files: list[str],
    path: str,
    device: str,
    m: CLIPModel,
):
    print(f"Loading all the images first and then encoding images together")
    # Pre-allocate the image tensor
    assert len(m.preprocess.transforms) > 0, "Expected at least one transform"
    assert isinstance(
        m.preprocess.transforms[0], torchvision.transforms.transforms.Resize
    ), f"Expected Resize transform, got {type(m.preprocess.transforms[0])}"

    size = m.preprocess.transforms[0].size[0]
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
        images_features = m.model.encode_image(x)
        print(f"Image features: {images_features.shape}")

    return images_features


@profile
def load_each_image_and_encode_immediately(
    file_basenames: list[str],
    base_dir: str,
    device: str,
    m: CLIPModel,
):
    print(f"Loading each image and encoding immediately")
    # Pre-allocate the image embedding tensor
    shape = list(model.modules())[-1].normalized_shape[0]  # type: ignore
    images_features = torch.empty(len(file_basenames), shape, device=device)  # type: ignore
    print(f"Pre-allocated image embedding tensor: {images_features.shape}")

    # Encode the images
    with torch.no_grad(), torch.amp.autocast(device):  # type: ignore
        for i, file in tqdm(enumerate(file_basenames), total=len(file_basenames)):
            images_features[i] = encode_image(base_dir, file, device, m)
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
    parser.add_argument(
        "--results_csv",
        type=str,
        help="Path to the CSV file to store the results.",
    )
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    FUNCTION_ENV = args.function
    RESULTS_CSV = args.results_csv

    # Load the image file paths
    load_dotenv()
    base_dir = Path(os.environ["CATS_AND_DOGS_TRAINING_DATASET"]) / "dogs"  # contains 4000 images
    file_basenames = os.listdir(base_dir)
    file_basenames = list(filter(is_valid_filename, file_basenames))
    file_basenames.sort()

    # Limit the number of files to BATCH_SIZE
    file_basenames = file_basenames[:BATCH_SIZE]
    print(f"Found {len(file_basenames)} files in {base_dir}")

    # Get the torch device
    device = torch_device()

    # Load the model into memory
    m = load_inference_clip_model(apple_DFN5B_CLIP_ViT_H_14_384, device)
    assert isinstance(m, CLIPModel)

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
    embeddings = fn(file_basenames, base_dir, device, m)
    end = time.time()
    print(f"Time taken for loop: {end - start:.2f}")
    print(f"Embeddings: {embeddings.shape}")

    # Save results to CSV
    results_csv = Path(RESULTS_CSV)
    pre_existing = results_csv.exists()
    with open(results_csv, "a") as f:
        if not pre_existing:
            f.write("function,batch_size,time_taken_secs\n")
        f.write(f"{FUNCTION_ENV},{BATCH_SIZE},{end - start}\n")


if __name__ == "__main__":
    main()
