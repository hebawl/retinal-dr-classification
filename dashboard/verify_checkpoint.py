"""Check the saved checkpoint against saved validation predictions (no training)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from model import DEFAULT_CHECKPOINT, PROJECT_ROOT, load_model, make_transform


class ValidationImages(Dataset):
    def __init__(self, manifest):
        self.rows = manifest.reset_index(drop=True)
        self.transform = make_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        relative_path = str(row["relative_path"]).replace("\\", "/")
        image_path = PROJECT_ROOT / Path(relative_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), int(row["label"]), row["image_id"]


def main():
    report_dir = PROJECT_ROOT / "reports" / "IDRiD_experiments"
    manifest = pd.read_csv(
        PROJECT_ROOT / "reports" / "IDRiD_clean_run" / "validation_manifest.csv"
    )
    saved = pd.read_csv(
        report_dir / "larger_input_300_weighted_validation_predictions.csv"
    )
    model, device = load_model(DEFAULT_CHECKPOINT)
    loader = DataLoader(ValidationImages(manifest), batch_size=8, shuffle=False)

    probabilities = []
    labels = []
    image_ids = []
    with torch.inference_mode():
        for images, batch_labels, batch_ids in loader:
            logits = model(images.to(device))
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.numpy())
            image_ids.extend(batch_ids)

    saved = saved.set_index("image_id").loc[image_ids]
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    saved_probabilities = saved["dr_probability"].to_numpy(dtype=float)
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    max_probability_difference = float(np.max(np.abs(probabilities - saved_probabilities)))
    same_classifications = bool(
        np.array_equal(predicted, saved["predicted_label"].to_numpy(dtype=int))
    )

    result = {
        "checkpoint": str(DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)),
        "images_checked": int(len(labels)),
        "validation_only": True,
        "training_performed": False,
        "same_predicted_classes_as_saved_file": same_classifications,
        "max_probability_difference": max_probability_difference,
        "accuracy": float(accuracy_score(labels, predicted)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "interpretation": (
            "Thresholded classes match; probabilities differ slightly. Use metrics recalculated from this checkpoint."
            if same_classifications
            else "Checkpoint thresholded classes differ from the saved validation predictions."
        ),
    }
    output = report_dir / "checkpoint_validation_verification.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved verification record: {output}")


if __name__ == "__main__":
    main()
