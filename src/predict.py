import argparse

import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

from model import get_model
from utils import load_config

CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Effusion",
    "Nodule",
    "Mass",
    "Cardiomegaly",
    "Fibrosis",
    "Edema",
]


def load_model(config, model_path, device):
    model = get_model(
        config,
        load_v1=False,
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    return model


def get_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    confidence, pred_idx = torch.max(
        probs,
        dim=1,
    )

    predicted_class = CLASS_NAMES[pred_idx.item()]

    return (
        predicted_class,
        confidence.item(),
        probs.squeeze().cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser(description="Chest X-Ray Disease Classification")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="best_model.pth",
        help="Path to model file",
    )

    args = parser.parse_args()

    config = load_config("configs/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = load_model(
        config,
        args.model,
        device,
    )

    transform = get_transform(config["data"]["img_size"])

    label, confidence, probs = predict(
        args.image,
        model,
        transform,
        device,
    )

    print("\n=== Prediction ===")
    print(f"Image      : {args.image}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.4f}")

    print("\nClass Probabilities:")

    for class_name, prob in zip(
        CLASS_NAMES,
        probs,
    ):
        print(f"{class_name:<15} {prob:.4f}")


if __name__ == "__main__":
    main()
