"""
This script is used to test if the OpenCLIP installation is working correctly.
"""

import torch
from line_profiler import profile
from PIL import Image

from rescueclip.open_clip import (
    apple_DFN5B_CLIP_ViT_H_14_384,
    load_inference_clip_model,
)


@profile
def main():
    TEST_IMAGE_PATH = "scripts/small_test_data/CLIP.png"

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
        image_features = model.encode_image(image)  # [1, 1024]
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]


if __name__ == "__main__":
    main()

"""
(rescueCLIP) ➜  RescueCLIP git:(main) ✗ time m line_profile
kernprof -lv scripts/open_clip_smoke_test.py
Using device: mps
Label probs: tensor([[1.0000e+00, 2.9189e-07, 5.9983e-10]], device='mps:0')
Wrote profile results to open_clip_smoke_test.py.lprof
Timer unit: 1e-06 s

Total time: 13.624 s
File: scripts/open_clip_smoke_test.py
Function: main at line 14

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    14                                           @profile
    15                                           def main():
    16         1          1.0      1.0      0.0      TEST_IMAGE_PATH = "scripts/small_test_data/CLIP.png"
    17
    18         1          0.0      0.0      0.0      device = "cpu"
    19         1          7.0      7.0      0.0      if torch.cuda.is_available():
    20                                                   device = "cuda"
    21         1       9212.0   9212.0      0.1      if torch.backends.mps.is_available():
    22         1          1.0      1.0      0.0          device = "mps"
    23
    24         1         17.0     17.0      0.0      print("Using device:", device)
    25
    26         1   13098226.0    1e+07     96.1      model, preprocess, tokenizer = load_inference_clip_model(apple_DFN5B_CLIP_ViT_H_14_384, device)
    27
    28         1      18747.0  18747.0      0.1      image = preprocess(Image.open(TEST_IMAGE_PATH)).unsqueeze(0).to(device)  # type: ignore
    29                                               # images = torch.stack([image, image, image, image, image]).to(device).squeeze(dim=1)
    30         1        635.0    635.0      0.0      text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)
    31
    32         2        397.0    198.5      0.0      with torch.no_grad(), torch.amp.autocast(device):  # type: ignore
    33         1     304325.0 304325.0      2.2          image_features = model.encode_image(image) # [1, 1024]
    34         1     142569.0 142569.0      1.0          text_features = model.encode_text(text)
    35         1       5825.0   5825.0      0.0          image_features /= image_features.norm(dim=-1, keepdim=True)
    36         1       4858.0   4858.0      0.0          text_features /= text_features.norm(dim=-1, keepdim=True)
    37
    38         1       6095.0   6095.0      0.0          text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    39
    40         1      33108.0  33108.0      0.2      print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

make line_profile  11.54s user 2.62s system 94% cpu 15.057 total
"""
