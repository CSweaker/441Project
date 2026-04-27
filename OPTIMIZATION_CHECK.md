# Optimized Package Check

This package is an optimized version of `network_intrusion_ids_project_ready.zip`.

## Fixes Applied

- Added the GitHub repository link to `README.md`.
- Added `seaborn>=0.13` to `requirements.txt` because the legacy baseline script imports seaborn.
- Updated `app.py` to use script-relative artifact paths instead of relying on the current working directory.
- Updated `app.py` to read uploaded files from memory instead of writing leftover temporary files.
- Updated `src/nids/data.py` so zip extraction only accepts `KDDTrain+.txt` and `KDDTest+.txt`.
- Updated `train_pipeline.py` and `run_sanity_check.py` to resolve paths relative to the project folder.
- Updated stratified quick sampling to return the exact requested row count when possible.
- Added an explicit artifact note in `README.md`.

## Validation Performed

- Source-level Python compile check: passed for all `.py` files.
- Data loading check: passed for `data/KDDTrain+.txt`.
- Exact stratified sample check: passed with 1,000 requested rows.
- Safe zip extraction check: extracted only `KDDTrain+.txt` and `KDDTest+.txt` from the original zip.
- App prediction logic check: passed using a minimal Streamlit stub and the included `artifacts/ids_artifacts.joblib`.
- Optimized quick training check: passed with 500 train rows and 200 test rows.

## Optimized Quick Training Result

- Train rows: 500
- Test rows: 200
- Best binary model: gaussian_nb_pca
- End-to-end accuracy: 0.74
- End-to-end macro F1: 0.45840758270237414

## Known Limitations

- The included default artifact is mainly for quick demo use. Re-run full training for final reproducible metrics.
- Streamlit is required to launch the UI. If `streamlit run app.py` fails, first run `pip install -r requirements.txt`.
- R2L and U2R are rare in NSL-KDD, so their recall may remain weaker than normal/DoS/Probe.
- Do not load `.joblib` artifacts from untrusted sources.
