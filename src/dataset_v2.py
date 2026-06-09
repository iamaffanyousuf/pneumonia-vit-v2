import pandas as pd

from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

LABEL_MAP = {
    "Normal": 0,
    "Pneumonia": 1,
    "Effusion": 2,
    "Nodule": 3,
    "Mass": 4,
    "Cardiomegaly": 5,
    "Fibrosis": 6,
    "Edema": 7,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        self.targets = [LABEL_MAP[label] for label in self.df["label"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")

        label = LABEL_MAP[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# gettransform fnc
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


# dataloader fnc
def get_dataloaders(config):
    img_size = config["data"]["img_size"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_tf, val_tf = get_transforms(img_size)

    # imagefolder logic/master_dataset lofgic
    train_dataset = ChestXrayDataset(
        config["data"]["train_csv"],
        transform=train_tf,
    )

    val_dataset = ChestXrayDataset(
        config["data"]["val_csv"],
        transform=val_tf,
    )

    test_dataset = ChestXrayDataset(
        config["data"]["test_csv"],
        transform=val_tf,
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
