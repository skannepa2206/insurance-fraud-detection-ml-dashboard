# Insurance Fraud Detection Dashboard

This project explores auto insurance fraud screening with tree-based and linear classifiers, threshold tuning, and an interactive Streamlit dashboard for reviewing model behavior.

Live dashboard: https://insurance-fraud-detection-ml-dashboard.streamlit.app/  
Source code: https://github.com/skannepa2206/insurance-fraud-detection-ml-dashboard

## Repository structure

- `streamlit_app.py` - interactive dashboard built on the saved artifacts
- `scripts/train_models.py` - trains the models, selects thresholds, and writes output files
- `scripts/build_visuals.py` - builds the summary charts used in the dashboard and write-up
- `artifacts/` - saved summaries, threshold tables, feature importance files, and figures
- `data/` - local copy of `train.csv` and `test.csv`

## Modeling approach

The workflow trains and compares:

- XGBoost
- Random Forest
- CatBoost
- Logistic Regression

The evaluation focuses on precision-recall behavior because fraud is a rare event in the dataset. Threshold selection is based on the validation split and can be optimized with either `F1` or `F2`.

## Run the dashboard

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Refresh the model artifacts

```powershell
pip install -r requirements-modeling.txt
python scripts/train_models.py
python scripts/build_visuals.py
```

Useful options:

```powershell
python scripts/train_models.py --model xgboost
python scripts/train_models.py --model best
python scripts/train_models.py --threshold-objective f2
```

## Notes

- `requirements.txt` is kept minimal for the deployed Streamlit app.
- `requirements-modeling.txt` includes the extra libraries needed to rerun the training workflow locally.
- The dashboard reads from the files in `artifacts/`, so the app does not need to retrain models at runtime.
