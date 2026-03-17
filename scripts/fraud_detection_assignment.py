from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


DEFAULT_DATA_DIR = Path(r"C:\Users\Veena\Downloads\AISystems_Mod4\Auto_fraud_detection_dataset")
DEFAULT_OUTPUT_DIR = Path("assignment_outputs") / "fraud_detection"
MODEL_ORDER = ("xgboost", "random_forest", "logistic", "catboost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a local fraud-detection model with threshold analysis."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        choices=("xgboost", "random_forest", "logistic", "catboost", "best"),
        default="xgboost",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=("f1", "f2"),
        default="f1",
        help="Metric used on the validation split to pick the score threshold.",
    )
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Could not find train.csv and test.csv under {data_dir}."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def available_model_builders(scale_pos_weight: float) -> dict[str, object]:
    builders: dict[str, object] = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
        "logistic": lambda: LogisticRegression(
            max_iter=4000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
    }

    if XGBClassifier is not None:
        builders["xgboost"] = lambda: XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
        )

    if CatBoostClassifier is not None:
        builders["catboost"] = lambda: CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            auto_class_weights="Balanced",
            thread_count=1,
        )

    return builders


def threshold_scores(
    precision: np.ndarray, recall: np.ndarray, objective: str
) -> np.ndarray:
    precision = precision[:-1]
    recall = recall[:-1]

    if objective == "f2":
        beta_sq = 4.0
        return (1 + beta_sq) * precision * recall / np.clip(
            beta_sq * precision + recall, 1e-12, None
        )

    return 2 * precision * recall / np.clip(precision + recall, 1e-12, None)


def select_threshold(
    y_true: pd.Series, probabilities: np.ndarray, objective: str
) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    scores = threshold_scores(precision, recall, objective)
    best_idx = int(np.nanargmax(scores))
    threshold = float(thresholds[best_idx])
    return {
        "threshold": threshold,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(
            2
            * precision[best_idx]
            * recall[best_idx]
            / max(precision[best_idx] + recall[best_idx], 1e-12)
        ),
        "f2": float(
            5
            * precision[best_idx]
            * recall[best_idx]
            / max(4 * precision[best_idx] + recall[best_idx], 1e-12)
        ),
    }


def metrics_at_threshold(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "threshold": float(threshold),
        "flagged_claims": int(predictions.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
    }


def build_threshold_table(
    y_true: pd.Series, probabilities: np.ndarray, chosen_threshold: float
) -> pd.DataFrame:
    grid = [0.2, 0.3, 0.4, 0.5, chosen_threshold, 0.7]
    rows = [metrics_at_threshold(y_true, probabilities, threshold) for threshold in grid]
    table = pd.DataFrame(rows).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    return table.reset_index(drop=True)


def extract_feature_importance(model: object, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        scores = np.abs(np.asarray(model.coef_).ravel())
    elif hasattr(model, "get_feature_importance"):
        scores = np.asarray(model.get_feature_importance())
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_precision_recall(
    y_val: pd.Series,
    val_probabilities: np.ndarray,
    y_test: pd.Series,
    test_probabilities: np.ndarray,
    chosen_threshold: float,
    output_path: Path,
) -> None:
    val_precision, val_recall, _ = precision_recall_curve(y_val, val_probabilities)
    test_precision, test_recall, _ = precision_recall_curve(y_test, test_probabilities)
    val_ap = average_precision_score(y_val, val_probabilities)
    test_ap = average_precision_score(y_test, test_probabilities)
    baseline = float(y_test.mean())

    plt.figure(figsize=(9, 6))
    plt.plot(val_recall, val_precision, label=f"Validation PR (AP={val_ap:.3f})", linewidth=2)
    plt.plot(test_recall, test_precision, label=f"Test PR (AP={test_ap:.3f})", linewidth=2)
    plt.axhline(baseline, color="gray", linestyle="--", label=f"No-skill baseline ({baseline:.3f})")
    plt.axvline(
        metrics_at_threshold(y_test, test_probabilities, chosen_threshold)["recall"],
        color="black",
        linestyle=":",
        linewidth=1,
        label=f"Chosen threshold={chosen_threshold:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Auto Fraud Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compare_models(
    builders: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_objective: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    comparison_rows: list[dict[str, float | str]] = []
    artifacts: dict[str, dict[str, object]] = {}

    for model_name in MODEL_ORDER:
        if model_name not in builders:
            continue

        model = builders[model_name]()
        model.fit(X_train, y_train)
        val_probabilities = model.predict_proba(X_val)[:, 1]
        test_probabilities = model.predict_proba(X_test)[:, 1]
        threshold_choice = select_threshold(y_val, val_probabilities, threshold_objective)
        val_metrics = metrics_at_threshold(y_val, val_probabilities, threshold_choice["threshold"])
        test_metrics = metrics_at_threshold(y_test, test_probabilities, threshold_choice["threshold"])

        comparison_rows.append(
            {
                "model": model_name,
                "validation_average_precision": average_precision_score(y_val, val_probabilities),
                "validation_roc_auc": roc_auc_score(y_val, val_probabilities),
                "validation_f1_at_selected_threshold": val_metrics["f1"],
                "test_average_precision": average_precision_score(y_test, test_probabilities),
                "test_roc_auc": roc_auc_score(y_test, test_probabilities),
                "test_f1_at_selected_threshold": test_metrics["f1"],
                "selected_threshold": threshold_choice["threshold"],
            }
        )

        artifacts[model_name] = {
            "model": model,
            "val_probabilities": val_probabilities,
            "test_probabilities": test_probabilities,
            "threshold_choice": threshold_choice,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_average_precision", "test_average_precision"], ascending=False
    )
    return comparison.reset_index(drop=True), artifacts


def selected_model_name(requested_model: str, comparison: pd.DataFrame) -> str:
    if requested_model == "best":
        return str(comparison.iloc[0]["model"])
    if requested_model not in comparison["model"].tolist():
        available = ", ".join(comparison["model"].tolist())
        raise ValueError(f"Requested model '{requested_model}' is unavailable. Available models: {available}")
    return requested_model


def write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(args.data_dir)
    X = train_df.drop(columns=["fraud"])
    y = train_df["fraud"]
    X_test = test_df.drop(columns=["fraud"])
    y_test = test_df["fraud"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    builders = available_model_builders(scale_pos_weight)

    comparison, artifacts = compare_models(
        builders=builders,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        threshold_objective=args.threshold_objective,
    )

    chosen_name = selected_model_name(args.model, comparison)
    chosen_artifacts = artifacts[chosen_name]
    chosen_threshold = float(chosen_artifacts["threshold_choice"]["threshold"])

    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    build_threshold_table(
        y_true=y_test,
        probabilities=chosen_artifacts["test_probabilities"],
        chosen_threshold=chosen_threshold,
    ).to_csv(args.output_dir / f"{chosen_name}_threshold_table.csv", index=False)
    extract_feature_importance(
        chosen_artifacts["model"], X.columns.tolist()
    ).to_csv(args.output_dir / f"{chosen_name}_feature_importance.csv", index=False)

    plot_precision_recall(
        y_val=y_val,
        val_probabilities=chosen_artifacts["val_probabilities"],
        y_test=y_test,
        test_probabilities=chosen_artifacts["test_probabilities"],
        chosen_threshold=chosen_threshold,
        output_path=args.output_dir / f"{chosen_name}_precision_recall_curve.png",
    )

    summary = {
        "data_dir": str(args.data_dir),
        "selected_model": chosen_name,
        "threshold_objective": args.threshold_objective,
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "train_fraud_rate": float(y.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "validation_threshold_choice": chosen_artifacts["threshold_choice"],
        "validation_metrics_at_selected_threshold": chosen_artifacts["val_metrics"],
        "test_metrics_at_selected_threshold": chosen_artifacts["test_metrics"],
        "test_average_precision": float(
            average_precision_score(y_test, chosen_artifacts["test_probabilities"])
        ),
        "test_roc_auc": float(roc_auc_score(y_test, chosen_artifacts["test_probabilities"])),
        "top_features": extract_feature_importance(
            chosen_artifacts["model"], X.columns.tolist()
        ).head(10).to_dict(orient="records"),
    }
    write_json(summary, args.output_dir / f"{chosen_name}_summary.json")

    print("Saved outputs to:", args.output_dir.resolve())
    print("Available models:")
    print(comparison.to_string(index=False))
    print("\nSelected model:", chosen_name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
