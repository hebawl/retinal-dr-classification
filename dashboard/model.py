"""Load and run the saved IDRiD demo candidate."""

from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "IDRiD_demo_candidate_300px.pt"
INPUT_SIZE = 300
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = {0: "Normal", 1: "Diabetic retinopathy"}


def check_fundus_appearance(image: Image.Image, minimum_fraction=0.30):
    """Conservative color check for IDRiD-like fundus photos.

    This rejects obvious non-retinal uploads; it is not a validated image-type
    classifier and cannot prove that an image is a retinal photograph.
    """
    sample = image.convert("RGB").resize((96, 96))
    pixels = np.asarray(sample, dtype=np.float32)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    orange_pixels = (
        (red > green * 1.05)
        & (green > blue * 1.05)
        & (red > 45)
        & (green > 25)
    )
    orange_fraction = float(orange_pixels.mean())
    return orange_fraction >= minimum_fraction, orange_fraction


def make_transform():
    """Match the saved 300px validation preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE + 32),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_model(checkpoint_path=DEFAULT_CHECKPOINT):
    """Load the local checkpoint without downloading pretrained weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. "
            "Keep the model file in the project's models folder."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_image(model, device, image: Image.Image):
    """Return the model's DR probability for one PIL image."""
    image = image.convert("RGB")
    image_tensor = make_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(image_tensor)
        probability_dr = torch.softmax(logits, dim=1)[0, 1].item()
    return float(probability_dr)
