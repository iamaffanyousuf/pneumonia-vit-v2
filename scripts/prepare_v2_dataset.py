import pandas as pd
from pathlib import Path

TARGET_CLASSES = {
    "Effusion",
    "Fibrosis",
    "Cardiomegaly",
    "Mass",
    "Nodule",
    "Edema",
}

INPUT_FILE = "data_v2/raw/nih/Data_Entry_2017.xlsx"
OUTPUT_FILE = "data_v2/metadata/filtered_v2.csv"

df = pd.read_excel(INPUT_FILE)

print(f"Total rows before filtering: {len(df)}")

# Remove multi-label rows
df = df[~df["Finding Labels"].str.contains(r"\|", regex=True)]

print(f"Total rows after removing multi-label entries: {len(df)}")

# Keep only target classes
df = df[df["Finding Labels"].isin(TARGET_CLASSES)]

print("\nClass Distribution:\n")
print(df["Finding Labels"].value_counts())

print(f"\nTotal Images: {len(df)}")

# Save filtered metadata
Path("data_v2/metadata").mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved filtered dataset to: {OUTPUT_FILE}")
