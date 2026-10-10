# IDRiD diabetic retinopathy project report

## Project goal
The project classifies retinal photographs into two groups: grade 0 is Normal; grades 1-4 are grouped as Diabetic Retinopathy (DR). It does not predict the exact grade.

## Data checks and preparation
The IDRiD training source has 413 images: 134 Normal and 279 DR, so the classes are imbalanced. Two records for an exact-copy pair had conflicting grades. Both records were excluded from modeling manifests, but the original image files were kept. This left 412 eligible training-source images: 329 for training and 83 for validation. The official test source had 103 images, with 102 eligible after excluding the conflicting-copy record. Exact-copy groups were kept together during splitting. Near-match candidates were recorded for review and not automatically removed. Training used class-weighted loss calculated from the training split only.

## Current demo candidate
The packaged candidate is EfficientNet-B0 at 300 x 300 pixels, initialized from ImageNet weights and trained with class-weighted loss. Its checkpoint is `models/IDRiD_demo_candidate_300px.pt`. It is for supervised internal demonstration, not clinical diagnosis.

## Saved validation results
The packaged checkpoint was run on the same 83 validation images without training. Its thresholded predictions match the saved validation file: 92.8% accuracy, 92.9% DR sensitivity, 92.6% specificity, and 96.3% precision. The checkpoint AUC recalculates to 0.981. The confusion counts are TN=25, FP=2, FN=4, TP=52. It missed 4 of 56 DR images and incorrectly flagged 2 of 27 Normal images in this sample.

Four completed patient-grouped validation folds averaged 88.8% accuracy, 87.4% sensitivity, 91.6% specificity, and 0.963 AUC at 224 pixels. At 300 pixels, the means were 91.8%, 91.5%, 92.5%, and 0.970. Fold five was stopped before evaluation because training took too long. These are preliminary four-fold averages, not five-fold results.

## Important result mismatch
Older saved notebook/report outputs show different single-split 300-pixel validation results: 91.6% accuracy, 94.6% sensitivity, 85.2% specificity, and 0.983 AUC. The packaged checkpoint now confirms the same predicted class for all 83 images, but its probability scores differ from the saved prediction file by up to 5.6 percentage points. Its recalculated AUC is 0.981. Use the checkpoint-recalculated AUC and treat all results as provisional because the historical records disagree and the sample is small.

## What the measures mean
- Accuracy: fraction of all images classified correctly.
- Sensitivity: fraction of DR images detected. Missing DR can be serious.
- Specificity: fraction of Normal images correctly left unflagged.
- Precision: fraction of images flagged as DR that truly have DR.
- AUC: how well the model ranks DR above Normal across thresholds. It does not choose a safe threshold.

## Test-set history and limitations
The current saved baseline test record is 77.5% accuracy, 77.9% sensitivity, 76.5% specificity, and 0.877 AUC on 102 images. An earlier project document claimed 81.4% accuracy, 76.5% sensitivity, 91.2% specificity, and 0.876 AUC, so the historical test records disagree. An earlier 300-pixel test re-check recorded 79.4% accuracy, 75.0% sensitivity, 88.2% specificity, and 0.884 AUC.

The official test set has been examined in earlier project work and is not an untouched independent final test. Do not use these figures to claim clinical accuracy.

The dataset is small and from a limited source. Class weighting cannot replace more diverse DR examples. Only four grouped folds finished. Results may change across clinics, cameras, and patient groups. No outside clinical dataset has been used.

## Recommended next steps
1. The local Streamlit prototype shows the image, model output, uncalibrated score, model name, and a warning for small images. Keep it restricted to supervised research or education.
2. Before sharing it with a community, choose an access-controlled host and decide how images will be protected, retained, and deleted.
3. Require a qualified reviewer for every case. Do not present the result as a diagnosis or replace an eye examination.
4. Before any clinical claim or real screening use, evaluate on a new independent labeled cohort from the intended community.

## Current files
The readable workflow is in five notebooks: EDA, preprocessing, baseline training, experiments, and grouped validation. The grouped-validation notebook records four completed folds and the stopped fifth fold. The local candidate checkpoint and settings are in `models/`. Source IDRiD images remain in `data/IDRiD/` and were not deleted.
