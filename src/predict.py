import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import get_model
from utils import load_config


def load_model(config, model_path, device):
    model = get_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict(image_path, model, transform, device, threshold=0.6):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    pneumonia_prob = probs[0][1].item()
    normal_prob = probs[0][0].item()

    label = "PNEUMONIA" if pneumonia_prob > threshold else "NORMAL"

    return label, pneumonia_prob, normal_prob


def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detection CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model", type=str, default="best_model.pth", help="Path to model file"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Decision threshold"
    )

    args = parser.parse_args()

    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config, args.model, device)
    transform = get_transform(config["data"]["img_size"])

    label, pneu_prob, norm_prob = predict(
        args.image, model, transform, device, args.threshold
    )

    print("\n=== Prediction ===")
    print(f"Image       : {args.image}")
    print(f"Prediction  : {label}")
    print(f"PNEUMONIA   : {pneu_prob:.4f}")
    print(f"NORMAL      : {norm_prob:.4f}")


if __name__ == "__main__":
    main()
