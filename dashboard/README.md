# Local dashboard prototype

The Streamlit app is a local research/education demonstration. It runs the saved 300-pixel EfficientNet-B0 checkpoint and shows a binary model output: Normal or possible DR. It does not predict the DR grade.

## Run it locally

1. Open a terminal in the project folder and activate the Python environment that has PyTorch and torchvision installed.
2. If Streamlit is missing, install the project dependencies with `python -m pip install -r requirements.txt`.
3. Confirm `models/IDRiD_demo_candidate_300px.pt` exists. The checkpoint is local and ignored by Git.
4. Start the app with:

   ```powershell
   python -m streamlit run dashboard/app.py --server.address=127.0.0.1 --server.maxUploadSize=20
   ```

5. Use the local address printed by Streamlit, usually `http://127.0.0.1:8501`.

The app does not save uploaded images to disk. While running locally, images are sent to the local app process. If hosted on another computer, uploads will be sent to that host. Do not use patient-identifiable images until privacy, access control, retention, and deletion practices have been reviewed.

## Upload image check

Before prediction, the app applies a simple orange-red color check intended to reject obvious non-fundus images. It rejects the outdoor photo shown in the project discussion and passes the IDRiD sample checked during implementation. This is only a basic guard: orange-colored non-eye images could pass, and unusual retinal images could be rejected. It has not been validated as an eye-image detector. A reliable deployment needs a separately trained and independently evaluated fundus-versus-non-fundus detector with representative positive and negative images.

## Before sharing with a community

- Choose an internal host that supports access control; do not make the prototype public.
- Keep the model checkpoint on the host in a protected location and pin compatible dependency versions.
- Decide how users are authorized and how any uploaded data are handled.
- Keep the warning that scores are uncalibrated and not diagnoses.
- Require qualified human review. Do not use the output to make care decisions.
- Get an independent labeled dataset from the intended community before making performance claims.

## Checkpoint verification

`python dashboard/verify_checkpoint.py` reruns inference on the saved validation images only. It does not train a model or read the test images. Its output is saved at `reports/IDRiD_experiments/checkpoint_validation_verification.json`.
