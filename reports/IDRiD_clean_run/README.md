# Historical IDRiD baseline record

This folder contains the earlier 224-pixel baseline artifacts. The current internal demo candidate is documented in `reports/IDRiD_experiments/current_candidate_summary.json` and saved as `models/IDRiD_demo_candidate_300px.pt`.

## Data and labels

- Binary task: grade 0 = Normal; grades 1-4 = DR.
- Original training source: 413 images (134 Normal, 279 DR).
- Two exact-copy records with conflicting grades were omitted from the modeling manifests. The original images were not deleted.
- Eligible training-source images: 412, split into 329 training and 83 validation images.
- Original official test source: 103 images; 102 eligible after excluding the conflicting-copy record.
- Exact copies were grouped during splitting. Near-match candidates were recorded for manual review and were not automatically deleted.
- The baseline used class-weighted cross-entropy, with weights derived from training data only.

## Baseline metrics and record conflict

The current saved baseline test predictions calculate to 77.5% accuracy, 77.9% sensitivity, 76.5% specificity, and 0.877 AUC (TN=26, FP=8, FN=15, TP=53). An older project summary claimed 81.4% accuracy, 76.5% sensitivity, 91.2% specificity, and 0.876 AUC. Those records disagree; do not present either as an independent final result.

The official test set was evaluated earlier in the project. It is no longer an untouched final test set. The validation and test outputs in this folder are retained as historical evidence, not as the result of the current demo candidate.

## Current folder contents

Manifests, predictions, training history, duplicate review files, and the baseline checkpoint are retained as historical evidence. Current experiment comparisons and the candidate selection are summarized under `reports/IDRiD_experiments/`.
