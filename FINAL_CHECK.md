# Final Package Check

## Status

The package was generated and a quick training run was completed successfully.

## Actual smoke-run result included in this package

- Train sample rows: 10000
- Test sample rows: 3000
- Best Stage-1 binary model: random_forest
- End-to-end accuracy: 0.768
- End-to-end macro F1: 0.4242028881510154

## Checked items

- Python files compile successfully.
- `train_pipeline.py` runs successfully in quick mode.
- Reports are generated under `reports/`.
- Model artifact is generated under `artifacts/ids_artifacts.joblib`.
- Streamlit app code is included. To run it locally, install requirements first.

## Important note

The included artifact was trained in quick mode on a sample so that the app can be opened immediately.
For final submission or best results, rerun:

```bash
python train_pipeline.py --train data/KDDTrain+.txt --test data/KDDTest+.txt
```

This will overwrite the sample artifact with a full-data artifact.
