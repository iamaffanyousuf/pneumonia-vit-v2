from pathlib import Path
import pandas as pd

rows = []

# V1 DATASET
V1_ROOT = Path("data_v2/raw/pneumonia_v1")

for split in ["train", "val"]:
    for class_name in ["NORMAL", "PNEUMONIA"]:

        class_dir = V1_ROOT / split / class_name

        if not class_dir.exists():
            continue

        label = "Normal" if class_name == "NORMAL" else "Pneumonia"

        for image_path in class_dir.iterdir():
            if image_path.is_file():
                rows.append(
                    {
                        "image_path": str(image_path),
                        "label": label,
                        "source": "pneumonia_v1",
                    }
                )

# NIH DATASET
NIH_ROOT = Path("data_v2/raw/nih")

nih_df = pd.read_csv("data_v2/metadata/filtered_v2_verified.csv")

print("Building NIH image lookup...")

image_lookup = {}

for image_file in NIH_ROOT.rglob("*.png"):
    image_lookup[image_file.name] = str(image_file)

print(f"Indexed {len(image_lookup)} NIH images")

for _, row in nih_df.iterrows():

    filename = row["Image Index"]

    if filename not in image_lookup:
        continue

    rows.append(
        {
            "image_path": image_lookup[filename],
            "label": row["Finding Labels"],
            "source": "nih",
        }
    )

# SAVE
master_df = pd.DataFrame(rows)

output_file = "data_v2/metadata/master_dataset.csv"

master_df.to_csv(output_file, index=False)

print("\nMaster Dataset Created")
print(master_df["label"].value_counts())
print(f"\nTotal rows: {len(master_df)}")
print(f"\nSaved to: {output_file}")
