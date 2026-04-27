"""Train the multi-stage NSL-KDD intrusion detection project.

Example:
    python train_pipeline.py --train data/KDDTrain+.txt --test data/KDDTest+.txt

Fast smoke test:
    python train_pipeline.py --quick --sample-size 10000 --test-sample-size 3000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.nids.config import RANDOM_STATE
from src.nids.data import add_targets, extract_zip_if_needed, feature_frame, load_nsl_kdd
from src.nids.modeling import (
    evaluate_classifier,
    get_feature_importance_from_pipeline,
    make_attack_family_model,
    save_artifacts,
    train_and_compare_binary_models,
)
from src.nids.plots import (
    plot_class_distribution,
    plot_confusion,
    plot_feature_importance,
    plot_metrics_table,
)
from src.nids.stats_analysis import run_statistical_analysis


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-stage IDS model on NSL-KDD.")
    parser.add_argument("--train", default="data/KDDTrain+.txt", help="Path to KDDTrain+.txt")
    parser.add_argument("--test", default="data/KDDTest+.txt", help="Path to KDDTest+.txt")
    parser.add_argument("--zip", default=None, help="Optional zip file containing NSL-KDD text files")
    parser.add_argument("--out", default="reports", help="Output report directory")
    parser.add_argument("--artifacts", default="artifacts/ids_artifacts.joblib", help="Model artifact path")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional exact stratified train sample size")
    parser.add_argument("--test-sample-size", type=int, default=None, help="Optional exact stratified test sample size")
    parser.add_argument("--quick", action="store_true", help="Use faster models for smoke testing")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def resolve_path(path_value: str | Path, *, base_dir: Path = BASE_DIR) -> Path:
    """Resolve user paths robustly whether the script is run from project root or elsewhere."""
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def stratified_sample(df: pd.DataFrame, target: str, n: int | None, random_state: int) -> pd.DataFrame:
    """Return an exact-size stratified sample whenever n is smaller than len(df)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    if n <= 0:
        raise ValueError("Sample size must be positive.")

    counts = df[target].value_counts(sort=False)
    if n < len(counts):
        selected_classes = counts.sort_values(ascending=False).head(n).index
        parts = [
            df[df[target] == cls].sample(n=1, random_state=random_state)
            for cls in selected_classes
        ]
        return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)

    raw = counts * (n / len(df))
    allocated = np.floor(raw).astype(int)
    allocated[allocated == 0] = 1

    while int(allocated.sum()) > n:
        reducible = allocated[allocated > 1]
        cls = reducible.idxmax()
        allocated.loc[cls] -= 1

    remainders = (raw - np.floor(raw)).sort_values(ascending=False)
    while int(allocated.sum()) < n:
        changed = False
        for cls in remainders.index:
            if allocated.loc[cls] < counts.loc[cls]:
                allocated.loc[cls] += 1
                changed = True
                if int(allocated.sum()) == n:
                    break
        if not changed:
            break

    parts = []
    for cls, group in df.groupby(target, group_keys=False):
        group_n = int(min(allocated.loc[cls], len(group)))
        parts.append(group.sample(n=group_n, random_state=random_state))
    sampled = pd.concat(parts, axis=0).sample(frac=1, random_state=random_state)
    return sampled.reset_index(drop=True)


def main() -> None:
    args = parse_args()

    out_dir = resolve_path(args.out)
    artifact_path = resolve_path(args.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.zip:
        extract_zip_if_needed(resolve_path(args.zip), data_dir=BASE_DIR / "data")

    train_path = resolve_path(args.train)
    test_path = resolve_path(args.test)

    print("[1/7] Loading data...")
    train_df = add_targets(load_nsl_kdd(train_path))
    test_df = add_targets(load_nsl_kdd(test_path))

    train_df = stratified_sample(train_df, "final_target", args.sample_size, args.random_state)
    test_df = stratified_sample(test_df, "final_target", args.test_sample_size, args.random_state)

    X_train = feature_frame(train_df)
    X_test = feature_frame(test_df)
    y_train_binary = train_df["binary_target"]
    y_test_binary = test_df["binary_target"]

    print(f"Train rows: {len(train_df):,}; Test rows: {len(test_df):,}")
    print("Class counts:")
    print(train_df["final_target"].value_counts())

    print("[2/7] Running statistical analysis...")
    stats_summary = run_statistical_analysis(train_df, out_dir=out_dir, random_state=args.random_state)
    plot_class_distribution(train_df, out_dir / "class_distribution.png")

    print("[3/7] Training binary baseline and stronger models...")
    binary_results, best_binary_name, fitted_binary = train_and_compare_binary_models(
        X_train,
        y_train_binary,
        X_test,
        y_test_binary,
        random_state=args.random_state,
        quick=args.quick,
    )
    binary_results.to_csv(out_dir / "binary_model_comparison.csv", index=False)
    plot_metrics_table(binary_results, out_dir / "binary_model_comparison.png")
    best_binary_model = fitted_binary[best_binary_name]["model"]
    best_binary_metrics = fitted_binary[best_binary_name]["metrics"]

    print("[4/7] Training stage-2 attack family model...")
    attack_train = train_df[train_df["binary_target"] == 1].copy()
    attack_test = test_df[test_df["binary_target"] == 1].copy()

    X_attack_train = feature_frame(attack_train)
    y_attack_train = attack_train["attack_family"]
    X_attack_test = feature_frame(attack_test)
    y_attack_test = attack_test["attack_family"]

    attack_family_model = make_attack_family_model(
        X_attack_train,
        random_state=args.random_state,
        quick=args.quick,
    )
    attack_family_model.fit(X_attack_train, y_attack_train)

    family_labels = ["DoS", "Probe", "R2L", "U2R", "Other"]
    family_metrics = evaluate_classifier(attack_family_model, X_attack_test, y_attack_test, labels=family_labels)

    print("[5/7] Evaluating end-to-end two-stage predictions...")
    binary_pred = best_binary_model.predict(X_test)
    final_pred = pd.Series(["normal"] * len(test_df), index=test_df.index, dtype=object)
    attack_idx = test_df.index[binary_pred == 1]
    if len(attack_idx) > 0:
        final_pred.loc[attack_idx] = attack_family_model.predict(feature_frame(test_df.loc[attack_idx]))

    final_labels = ["normal", "DoS", "Probe", "R2L", "U2R", "Other"]
    final_report = classification_report(test_df["final_target"], final_pred, labels=final_labels, zero_division=0)
    final_cm = confusion_matrix(test_df["final_target"], final_pred, labels=final_labels)
    final_metrics = {
        "accuracy": float(accuracy_score(test_df["final_target"], final_pred)),
        "macro_f1": float(
            f1_score(test_df["final_target"], final_pred, labels=final_labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(test_df["final_target"], final_pred, labels=final_labels, average="weighted", zero_division=0)
        ),
        "classification_report": final_report,
        "confusion_matrix": final_cm.tolist(),
    }

    print("[6/7] Saving plots and reports...")
    (out_dir / "binary_best_classification_report.txt").write_text(
        f"Best binary model: {best_binary_name}\n\n{best_binary_metrics['classification_report']}",
        encoding="utf-8",
    )
    (out_dir / "attack_family_classification_report.txt").write_text(
        family_metrics["classification_report"],
        encoding="utf-8",
    )
    (out_dir / "end_to_end_classification_report.txt").write_text(final_report, encoding="utf-8")

    plot_confusion(
        best_binary_metrics["confusion_matrix"],
        ["Normal", "Attack"],
        "Stage 1 Confusion Matrix",
        out_dir / "stage1_confusion_matrix.png",
    )
    plot_confusion(
        family_metrics["confusion_matrix"],
        family_labels,
        "Stage 2 Attack Family Confusion Matrix",
        out_dir / "stage2_confusion_matrix.png",
    )
    plot_confusion(
        final_metrics["confusion_matrix"],
        final_labels,
        "End-to-End Confusion Matrix",
        out_dir / "end_to_end_confusion_matrix.png",
    )

    importances = get_feature_importance_from_pipeline(best_binary_model, top_n=25)
    importances.to_csv(out_dir / "feature_importance.csv", index=False)
    plot_feature_importance(importances, out_dir / "feature_importance.png")

    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "best_binary_model": best_binary_name,
        "binary_results": binary_results.to_dict(orient="records"),
        "statistical_summary": stats_summary,
        "end_to_end_metrics": final_metrics,
        "artifact_note": "Included default artifact may be a quick-run artifact. Re-run full training for final results.",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[7/7] Saving model artifacts...")
    save_artifacts(
        artifact_path,
        binary_model=best_binary_model,
        attack_family_model=attack_family_model,
        feature_columns=list(X_train.columns),
        final_labels=final_labels,
        family_labels=family_labels,
        best_binary_model_name=best_binary_name,
        summary=summary,
    )

    print("\nDone.")
    print(f"Best binary model: {best_binary_name}")
    print(f"End-to-end accuracy: {final_metrics['accuracy']:.4f}")
    print(f"End-to-end macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"Reports saved to: {out_dir.resolve()}")
    print(f"Artifacts saved to: {artifact_path.resolve()}")


if __name__ == "__main__":
    main()
