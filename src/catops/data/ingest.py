# src/catops/data/ingest.py
"""Data ingestion & preprocessing.

- Extracts images.tar.gz
- Organizes images into binary folders: cat/ and not_cat/
- Prepares data for training pipelines.
"""

from pathlib import Path
import tarfile
import shutil
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def extract_dataset() -> None:
    """Extract the tar.gz if not already done."""
    tar_path = RAW_DIR / "images.tar.gz"
    extract_dir = RAW_DIR / "images"

    if not tar_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found at {tar_path}")

    if not extract_dir.exists():
        print("📦 Extracting Oxford-IIIT Pet images...")
        with tarfile.open(tar_path) as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting"):
                tar.extract(member, path=RAW_DIR)
        print("✅ Extraction complete")
    else:
        print("✅ Images already extracted")


def organize_for_binary_classification() -> None:
    """Organize images into cat / not_cat folders."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cat_dir = PROCESSED_DIR / "cat"
    not_cat_dir = PROCESSED_DIR / "not_cat"

    cat_dir.mkdir(exist_ok=True)
    not_cat_dir.mkdir(exist_ok=True)

    images_dir = RAW_DIR / "images"

    print("🔄 Organizing images into cat / not_cat...")
    image_list = list(images_dir.glob("*.jpg"))

    for img_path in tqdm(image_list, desc="Organizing"):
        # Oxford-IIIT Pet: uppercase first letter = cat
        if img_path.name[0].isupper():
            shutil.copy(img_path, cat_dir / img_path.name)
        else:
            shutil.copy(img_path, not_cat_dir / img_path.name)

    cat_count = len(list(cat_dir.glob("*.jpg")))
    not_cat_count = len(list(not_cat_dir.glob("*.jpg")))

    print("Dataset organized:")
    print(f"   → {cat_count} cat images")
    print(f"   → {not_cat_count} not_cat images")
    print(f"   → Total: {cat_count + not_cat_count} images")


if __name__ == "__main__":
    extract_dataset()
    organize_for_binary_classification()
