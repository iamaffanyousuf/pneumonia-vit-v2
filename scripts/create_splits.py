from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data_v2/metadata/master_dataset.csv"

OUTPUT_DIR = Path("data_v2/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print(f"Total images: {len(df)}")

# Train (70%) / Temp (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42,
)

# Val (15%) / Test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)

# Save
train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False,
)

val_df.to_csv(
    OUTPUT_DIR / "val.csv",
    index=False,
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False,
)

print("\nTrain Distribution:")
print(train_df["label"].value_counts())

print("\nVal Distribution:")
print(val_df["label"].value_counts())

print("\nTest Distribution:")
print(test_df["label"].value_counts())

print("\nDone!")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")
