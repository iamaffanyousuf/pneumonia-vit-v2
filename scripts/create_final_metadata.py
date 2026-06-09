from pathlib import Path
import pandas as pd

FINAL_ROOT = Path("data_v2/raw/final_dataset")

FILES = [
    "train",
    "val",
    "test",
]

for split in FILES:
    csv_path = f"data_v2/metadata/{split}.csv"

    df = pd.read_csv(csv_path)

    new_paths = []

    for _, row in df.iterrows():
        old_path = Path(row["image_path"])

        new_path = FINAL_ROOT / row["label"] / old_path.name

        new_paths.append(str(new_path))

    df["image_path"] = new_paths

    output_path = f"data_v2/metadata/{split}_final.csv"

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")
    print(f"Rows: {len(df)}")
