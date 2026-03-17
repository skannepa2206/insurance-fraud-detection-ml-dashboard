# Insurance Fraud Detection ML Dashboard

This repository contains the modeling workflow, generated evaluation artifacts, and a Streamlit dashboard for an auto insurance fraud detection project focused on imbalanced classification, precision-recall analysis, and threshold selection.

## Repository contents

- `streamlit_app.py`: dashboard for viewing the assignment results
- `scripts/fraud_detection_assignment.py`: local training and evaluation workflow
- `scripts/generate_submission_visuals.py`: script that generates supporting charts
- `artifacts/`: saved metrics, figures, and model output files used in the paper and dashboard

## What is intentionally not included

- The original course-provided `train.csv` and `test.csv` files are not included by default.
- The local virtual environment and unrelated project files are excluded.

If the instructor needs full reproduction, the course datasets should be placed in:

```text
data/
  train.csv
  test.csv
```

Then the training script can be adapted to point at that location or the original local dataset path can be changed.

## Dashboard

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The dashboard presents:

- benchmark comparison across multiple classifiers
- validation-based threshold selection
- precision-recall evidence for the XGBoost submission model
- feature importance and downloadable output artifacts

## Recreate the analysis artifacts

If the datasets are available locally, the analysis and figure scripts can be run with:

```powershell
python scripts/fraud_detection_assignment.py --model xgboost
python scripts/generate_submission_visuals.py
```

## Project note

The Streamlit dashboard is designed as a concise review layer so the results, threshold logic, model comparison, and generated visuals can be inspected quickly without reading through raw output files first.
