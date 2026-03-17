# Insurance Fraud Detection Dashboard

## Project overview

This project focuses on auto insurance fraud detection as an imbalanced binary classification problem. The goal is to identify suspicious claims more effectively than random screening and to translate model output into an operational review workflow. Instead of treating the exercise as a pure modeling task, the project emphasizes the full decision flow: train a model, evaluate it with the right metrics, choose a practical score threshold, and present the results in a dashboard that is easy to interpret.

The repository includes a local training workflow, saved evaluation artifacts, and a Streamlit dashboard that summarizes model quality, threshold tradeoffs, and the strongest fraud signals. The dashboard is designed for quick review of model behavior rather than model training.

## Learning objectives

This project was built around four main objectives:

1. Apply machine learning to a real fraud-detection use case using structured claims data.
2. Evaluate rare-event classification with precision-recall analysis instead of relying on accuracy.
3. Select a score threshold that balances fraud capture against manual review workload.
4. Present the output in a clear interface that supports business interpretation.

## What the project covers

The modeling workflow compares:

- XGBoost
- Random Forest
- CatBoost
- Logistic Regression

The analysis is centered on precision, recall, F1, threshold behavior, and feature importance. Since fraud is a low-frequency class in this dataset, the workflow prioritizes ranking quality and threshold selection over plain accuracy.

## Key learnings

- Imbalanced classification needs the right evaluation lens. Precision-recall analysis is much more informative than accuracy for fraud problems.
- Threshold selection is part of the model design, not just a default setting. A score cutoff directly changes review volume and fraud capture.
- Benchmarking more than one algorithm is useful even when there is a preferred model going in.
- A dashboard is valuable when it explains the model in operational terms, not just technical metrics.

## Repository structure

- `streamlit_app.py` - interactive dashboard built on saved outputs
- `scripts/train_models.py` - local training, threshold selection, and artifact generation
- `scripts/build_visuals.py` - chart generation for saved outputs
- `artifacts/` - summaries, threshold tables, feature importance files, and figures
- `data/` - local copy of `train.csv` and `test.csv`

## Running the project

To run the dashboard:

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To refresh the modeling outputs:

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

- `requirements.txt` contains only the packages needed for the deployed dashboard.
- `requirements-modeling.txt` includes the additional libraries needed for local model training.
- The dashboard reads from `artifacts/`, so it does not retrain models at runtime.
