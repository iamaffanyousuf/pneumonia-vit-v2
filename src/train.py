import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from collections import Counter

from utils import load_config
from dataset import get_dataloaders
from model import get_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return total_loss / len(loader), acc


def main():
    config = load_config("configs/config.yaml")

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(config)

    model = get_model(config).to(device)

    # imbalance fix
    labels = train_loader.dataset.targets
    class_counts = Counter(labels)

    total = sum(class_counts.values())

    num_classes = len(class_counts)
    weights = [1.0 / class_counts.get(i, 1) for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=config["train"].get("weight_decay", 0.01),
    )

    # Save directory (Colab + local safe)
    SAVE_DIR = config.get("save_dir", "outputs")
    os.makedirs(SAVE_DIR, exist_ok=True)

    best_acc = 0
    patience = 5
    no_improve = 0

    for epoch in range(config["train"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save LAST model
        last_path = os.path.join(SAVE_DIR, "last_model.pth")
        torch.save(model.state_dict(), last_path)

        # Save BEST model
        if val_acc >= best_acc:
            best_acc = val_acc
            no_improve = 0

            best_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"✅ Best model saved at {best_path}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Save logs
        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        with open(os.path.join(SAVE_DIR, "log.json"), "a") as f:
            f.write(json.dumps(log_data) + "\n")


if __name__ == "__main__":
    main()
