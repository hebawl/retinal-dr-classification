# IDRiD diabetic retinopathy project

This project uses the IDRiD retinal image dataset. The model predicts **two classes**:

- `0` - Normal (grade 0)
- `1` - Diabetic retinopathy (grades 1, 2, 3, or 4)

It does not predict the four disease grades separately.

## Notebooks

Run these in order when you want to repeat the analysis:

1. `notebooks/01_IDRiD_EDA.ipynb` - inspect labels, class balance, image sizes, exact duplicates, and possible near duplicates.
2. `notebooks/02_IDRiD_Preprocessing.ipynb` - create the binary labels and clean train/validation/test manifests.
3. `notebooks/03_IDRiD_Training.ipynb` - train the 224-pixel baseline.
4. `notebooks/04_IDRiD_Experiments.ipynb` - compare the saved 224-pixel and 300-pixel experiments.
5. `notebooks/05_IDRiD_Grouped_CV.ipynb` - compare input sizes on patient-grouped validation folds. Four of five planned folds finished; fold five was stopped early.

The short project report is in `output/pdf/IDRiD_project_report.pdf`. The key current metrics are in `reports/IDRiD_experiments/current_candidate_summary.json`.

## Current model candidate

`models/IDRiD_demo_candidate_300px.pt` is the current model checkpoint for an **internal demonstration**. Its settings are in the same folder. The four completed grouped validation folds favored 300-pixel input on average. This is preliminary: one fold did not finish, and the existing official test set has been used in earlier project work. Do not describe these results as an independent final test.

The packaged checkpoint was run on the 83 validation images without training. It made the same class predictions as the saved file: 92.8% accuracy, 92.9% sensitivity, and 92.6% specificity. Its AUC recalculates to 0.981. Across four completed grouped validation folds, the 300-pixel mean was 91.8% accuracy, 91.5% sensitivity, 92.5% specificity, and 0.970 AUC. These numbers describe different validation comparisons and should not be combined. The checkpoint's scores differ by up to 5.6 percentage points from the saved prediction file; its own AUC is used here.

## Data and safety

The local `data/IDRiD/` folder contains the source IDRiD images and is not included in Git. The old RFMiD data files and the superseded single-file notebook are no longer part of the working project. Do not delete or rename the IDRiD source data if you need to run the notebooks.

This model is a learning and demonstration project, not a medical device. The dashboard must be restricted to a supervised research or educational setting, label its score as uncalibrated, and require a qualified clinician to review every case. It must not give patients a diagnosis or replace an eye examination.

## Running locally

Install packages from `requirements.txt`, select the project Python environment in Jupyter, then open this folder as the notebook working directory. The model checkpoint is kept locally and is ignored by Git because it is a generated binary artifact.

## Local dashboard prototype

From the project folder, run `python -m streamlit run dashboard/app.py --server.address=127.0.0.1 --server.maxUploadSize=20`. It opens on this computer only. See `dashboard/README.md` for setup and the steps required before any internal hosting. The model file must be copied securely to the host because it is intentionally excluded from Git. Do not expose this prototype publicly or use it for clinical decisions.
