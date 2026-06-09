from utils import load_config
from dataset import get_dataloaders


def main():
    config = load_config("configs/config.yaml")

    train_loader, val_loader, test_loader = get_dataloaders(config)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # Check one batch
    images, labels = next(iter(train_loader))
    print("Image shape:", images.shape)
    print("Labels:", labels[:10])


if __name__ == "__main__":
    main()
