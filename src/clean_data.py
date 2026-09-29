from pathlib import Path
from PIL import Image
import pandas as pd


# Project paths
DATA_DIR = Path("data")

SPLITS = {
    "Training": DATA_DIR / "Training_set",
    "Validation": DATA_DIR / "Validation_set",
    "Test": DATA_DIR / "Test_set",
}


def inspect_split(name, folder):
    print("\n" + "=" * 60)
    print(f"{name.upper()} SET")
    print("=" * 60)

    # Find images
    image_files = [
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    # Find CSV
    csv_files = list(folder.glob("*.csv"))

    print(f"Images found: {len(image_files)}")
    print(f"CSV files found: {len(csv_files)}")

    if not csv_files:
        print("ERROR: No label CSV found.")
        return

    csv_path = csv_files[0]
    labels = pd.read_csv(csv_path, encoding="latin1")

    print(f"Labels found: {len(labels)}")
    print(f"CSV file: {csv_path.name}")

    print("\nColumns:")
    print(list(labels.columns))

    # Check image IDs against CSV IDs
    image_ids = {p.stem for p in image_files}
    label_ids = set(labels["ID"].astype(str))

    missing_labels = image_ids - label_ids
    missing_images = label_ids - image_ids

    print(f"\nImages without labels: {len(missing_labels)}")
    print(f"Labels without images: {len(missing_images)}")

    # DR distribution
    if "DR" in labels.columns:
        print("\nDR distribution:")
        print(labels["DR"].value_counts().sort_index())

    # WNL distribution
    if "WNL" in labels.columns:
        print("\nWNL distribution:")
        print(labels["WNL"].value_counts().sort_index())

    # Check image readability
    bad_images = []
    sizes = {}

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                sizes[img.size] = sizes.get(img.size, 0) + 1

        except Exception as e:
            bad_images.append((image_path.name, str(e)))

    print(f"\nUnreadable/corrupted images: {len(bad_images)}")

    if bad_images:
        for filename, error in bad_images[:10]:
            print(f"  {filename}: {error}")

    print("\nImage dimensions:")
    for size, count in sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {size}: {count} images")


def main():
    print("RFMiD 2.0 DATASET INSPECTION")

    for name, folder in SPLITS.items():
        if folder.exists():
            inspect_split(name, folder)
        else:
            print(f"\nWARNING: {folder} does not exist.")


if __name__ == "__main__":
    main()