from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "artifacts"


def plot_model_comparison() -> None:
    df = pd.read_csv(OUTPUT_DIR / "model_comparison.csv").sort_values(
        "test_average_precision", ascending=False
    )
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["model"], df["test_average_precision"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"])
    plt.ylabel("Test Average Precision")
    plt.title("Model Comparison on Auto Fraud Test Set")
    plt.ylim(0, max(df["test_average_precision"]) * 1.2)
    for bar, value in zip(bars, df["test_average_precision"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison_test_average_precision.png", dpi=200)
    plt.close()


def plot_threshold_tradeoff() -> None:
    df = pd.read_csv(OUTPUT_DIR / "xgboost_threshold_table.csv").sort_values("threshold")
    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["precision"], marker="o", linewidth=2, label="Precision")
    plt.plot(df["threshold"], df["recall"], marker="o", linewidth=2, label="Recall")
    plt.plot(df["threshold"], df["f1"], marker="o", linewidth=2, label="F1")
    selected = df.loc[df["f1"].idxmax()] if "f1" in df.columns else df.iloc[0]
    plt.axvline(selected["threshold"], color="black", linestyle="--", label=f"Chosen threshold {selected['threshold']:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("XGBoost Threshold Tradeoff")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xgboost_threshold_tradeoff.png", dpi=200)
    plt.close()


def plot_feature_importance() -> None:
    df = pd.read_csv(OUTPUT_DIR / "xgboost_feature_importance.csv").head(10)
    df = df.sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 5.5))
    plt.barh(df["feature"], df["importance"], color="#1f77b4")
    plt.xlabel("Importance")
    plt.title("Top 10 XGBoost Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xgboost_top10_feature_importance.png", dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison()
    plot_threshold_tradeoff()
    plot_feature_importance()
    print("Saved visuals to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
