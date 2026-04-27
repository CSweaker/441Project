"""Model creation, evaluation, and persistence helpers."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from .features import make_full_preprocess_pipeline


def make_binary_model_specs(X: pd.DataFrame, random_state: int = 42, quick: bool = False) -> dict:
    """Return baseline and stronger binary IDS model pipelines."""
    rf_trees = 60 if not quick else 25
    return {
        "dummy_majority": Pipeline([
            ("prep", make_full_preprocess_pipeline(X, dense=True)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "gaussian_nb_pca": Pipeline([
            ("prep", make_full_preprocess_pipeline(X, dense=True)),
            ("pca", PCA(n_components=20, random_state=random_state)),
            ("model", GaussianNB()),
        ]),
        "logistic_pca": Pipeline([
            ("prep", make_full_preprocess_pipeline(X, dense=True)),
            ("pca", PCA(n_components=25, random_state=random_state)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("prep", make_full_preprocess_pipeline(X, dense=True)),
            ("model", RandomForestClassifier(
                n_estimators=rf_trees,
                max_depth=22,
                min_samples_leaf=2,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=random_state,
            )),
        ]),
    }


def make_attack_family_model(X: pd.DataFrame, random_state: int = 42, quick: bool = False) -> Pipeline:
    """Model for stage 2: attack family classification."""
    trees = 80 if not quick else 30
    return Pipeline([
        ("prep", make_full_preprocess_pipeline(X, dense=True)),
        ("model", ExtraTreesClassifier(
            n_estimators=trees,
            max_depth=24,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])


def evaluate_classifier(model, X_test, y_test, labels=None) -> dict:
    """Evaluate a fitted classifier and return standard metrics."""
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist() if labels else confusion_matrix(y_test, pred).tolist(),
        "predictions": pred,
    }


def train_and_compare_binary_models(X_train, y_train, X_test, y_test, random_state=42, quick=False):
    """Fit all binary models and choose the strongest one by macro F1."""
    specs = make_binary_model_specs(X_train, random_state=random_state, quick=quick)
    rows = []
    fitted = {}
    for name, model in specs.items():
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        metrics = evaluate_classifier(fitted_model, X_test, y_test, labels=[0, 1])
        rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })
        fitted[name] = {"model": fitted_model, "metrics": metrics}

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "model"]
    return results, best_name, fitted


def save_artifacts(path: str | Path, **objects) -> None:
    """Save model artifacts with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(objects, path)


def load_artifacts(path: str | Path) -> dict:
    """Load model artifacts saved by save_artifacts."""
    return joblib.load(path)


def get_feature_importance_from_pipeline(model: Pipeline, top_n: int = 25) -> pd.DataFrame:
    """Extract tree-model feature importances after preprocessing."""
    if "model" not in model.named_steps:
        return pd.DataFrame()
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    prep = model.named_steps["prep"]
    feature_names = prep.named_steps["preprocess"].get_feature_names_out()
    importances = estimator.feature_importances_
    out = pd.DataFrame({"feature": feature_names, "importance": importances})
    return out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
