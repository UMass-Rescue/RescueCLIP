"""
This script is used to test if the OpenCLIP installation is working correctly.
"""

from line_profiler import profile
import torch
from PIL import Image

from rescueclip.open_clip import (
    apple_DFN5B_CLIP_ViT_H_14_384,
    load_inference_clip_model,
)

@profile
def main():
    TEST_IMAGE_PATH = "scripts/small-test-data/CLIP.png"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"

    print("Using device:", device)

    model, preprocess, tokenizer = load_inference_clip_model(apple_DFN5B_CLIP_ViT_H_14_384, device)

    image = preprocess(Image.open(TEST_IMAGE_PATH)).unsqueeze(0).to(device)  # type: ignore
    # images = torch.stack([image, image, image, image, image]).to(device).squeeze(dim=1)
    text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad(), torch.amp.autocast(device):  # type: ignore
        image_features = model.encode_image(image) # [1, 1024]
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

if __name__ == "__main__":
    main()
