import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from utils import load_config
from dataset import get_dataloaders
from model import get_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

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
            images, labels = images.to(device), labels.to(device)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=float(config["train"]["lr"]))

    best_acc = 0

    for epoch in range(config["train"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved")


if __name__ == "__main__":
    main()
