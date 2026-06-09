from src.dataset_v2 import ChestXrayDataset, get_transforms

train_tf, _ = get_transforms(224)

dataset = ChestXrayDataset(
    "data_v2/metadata/train.csv",
    transform=train_tf,
)

print("Dataset size:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
