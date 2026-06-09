from pathlib import Path
import pandas as pd

# Paths
NIH_ROOT = Path("data_v2/raw/nih")
FILTERED_CSV = Path("data_v2/metadata/filtered_v2.csv")

print("Loading filtered metadata...")
df = pd.read_csv(FILTERED_CSV)

print(f"Rows in filtered dataset: {len(df)}")

print("\nScanning NIH image folders...")

image_index = {}

for image_file in NIH_ROOT.rglob("*.png"):
    image_index[image_file.name] = str(image_file)

print(f"Indexed {len(image_index)} images")

print("\nChecking filtered dataset images...")

missing = []

for filename in df["Image Index"]:
    if filename not in image_index:
        missing.append(filename)

print(f"Found: {len(df) - len(missing)}")
print(f"Missing: {len(missing)}")

if missing:
    print("\nFirst 20 missing files:")
    for file in missing[:20]:
        print(file)
else:
    print("\n✅ All filtered images were found.")


valid_filenames = set(image_index.keys())

clean_df = df[df["Image Index"].isin(valid_filenames)]

print(f"\nRows after removing missing images: {len(clean_df)}")

clean_df.to_csv(
    "data_v2/metadata/filtered_v2_verified.csv",
    index=False,
)

print("\nSaved verified metadata to " "data_v2/metadata/filtered_v2_verified.csv")
