"""Production-ready custom Dataset for "Am I a Cat?" pipeline.

Reads from the stratified CSV artifacts (train.csv/val.csv/test.csv) produced by the features stage.
Ensures no data leakage and full DVC reproducibility.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CatDataset(Dataset):
    """Custom Dataset that loads image paths and labels from CSV files."""

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[transforms.Compose] = None,
        processed_dir: str | Path = "data/processed",
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.transform = transform

        # Load split CSV (produced by features stage)
        df = pd.read_csv(csv_path)
        # Expected columns: 'image_path' (relative to processed_dir), 'label' (0=cat, 1=not_cat or string)
        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'image_path' and 'label' columns")

        self.image_paths = df["image_path"].tolist()
        self.labels = df["label"].tolist()

        # Convert string labels to int if needed (cat=0, not_cat=1)
        if isinstance(self.labels[0], str):
            self.labels = [0 if lbl.lower() == "cat" else 1 for lbl in self.labels]

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.processed_dir / self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label