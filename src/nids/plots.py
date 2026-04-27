"""Plotting helpers for reports."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_distribution(df: pd.DataFrame, out_path: str | Path) -> None:
    counts = df["final_target"].value_counts().sort_index()
    fig = plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("NSL-KDD Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metrics_table(metrics_df: pd.DataFrame, out_path: str | Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    labels = metrics_df["model"]
    values = metrics_df["macro_f1"]
    plt.bar(labels, values)
    plt.title("Binary IDS Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion(cm, labels, title: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importances: pd.DataFrame, out_path: str | Path) -> None:
    if importances.empty:
        return
    top = importances.sort_values("importance", ascending=True)
    fig = plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
