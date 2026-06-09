import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def get_dataloaders(config):
    img_size = config["data"]["img_size"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_tf, val_tf = get_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        root=config["data"]["train_dir"], transform=train_tf
    )

    val_dataset = datasets.ImageFolder(root=config["data"]["val_dir"], transform=val_tf)

    test_dataset = datasets.ImageFolder(
        root=config["data"]["test_dir"], transform=val_tf
    )

    # gpu optimizer
    pin_memory = True if config["device"] == "cuda" else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader
