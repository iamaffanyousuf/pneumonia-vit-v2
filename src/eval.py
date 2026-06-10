import torch
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from utils import load_config
from dataset_v2 import get_dataloaders
from model import get_model

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


def evaluate():
    config = load_config("configs/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(config)

    model = get_model(config, load_v1=False)

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(test_loader)}")

            images = images.to(device)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES,
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            all_labels,
            all_preds,
        )
    )


if __name__ == "__main__":
    evaluate()
