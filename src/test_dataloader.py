from utils import load_config
from dataset_v2 import get_dataloaders


def main():
    config = load_config("configs/config.yaml")

    train_loader, val_loader, test_loader = get_dataloaders(config)

    images, labels = next(iter(train_loader))

    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Labels sample:", labels[:10])


if __name__ == "__main__":
    main()
