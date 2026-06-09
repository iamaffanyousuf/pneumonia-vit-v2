from pathlib import Path
import pandas as pd
import shutil

MASTER_CSV = "data_v2/metadata/master_dataset.csv"

OUTPUT_ROOT = Path("data_v2/raw/final_dataset")

df = pd.read_csv(MASTER_CSV)

print(f"Total rows: {len(df)}")

# Create class folders
for label in sorted(df["label"].unique()):
    (OUTPUT_ROOT / label).mkdir(
        parents=True,
        exist_ok=True,
    )

copied = 0

for _, row in df.iterrows():
    src = Path(row["image_path"])
    dst = OUTPUT_ROOT / row["label"] / src.name

    if not dst.exists():
        shutil.copy2(src, dst)

    copied += 1

    if copied % 1000 == 0:
        print(f"Copied {copied}/{len(df)}")

print("\nDone!")
print(f"Total copied: {copied}")
