import torch
from PIL import Image
import open_clip

TEST_IMAGE_PATH = "src/rescueclip/small-test-data/CLIP.png"
CACHE_DIR = "./.cache/clip"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

print("Using device:", device)

model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", device=device, cache_dir=CACHE_DIR
)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer("ViT-H-14", cache_dir=CACHE_DIR)

image = preprocess(Image.open(TEST_IMAGE_PATH)).unsqueeze(0).to(device)  # type: ignore
text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad(), torch.amp.autocast(device):  # type: ignore
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
