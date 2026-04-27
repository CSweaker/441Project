"""Feature engineering and preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, LOG_TRANSFORM_COLUMNS


def clean_feature_values(X: pd.DataFrame) -> pd.DataFrame:
    """Copy input features and apply safe numeric cleaning/log transforms.

    NSL-KDD contains a few very skewed byte/count variables. log1p reduces their
    scale while preserving zeros. This helps linear baseline models.
    """
    X = X.copy()
    for col in X.columns:
        if col not in CATEGORICAL_COLUMNS:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in LOG_TRANSFORM_COLUMNS:
        if col in X.columns:
            X[col] = np.log1p(np.clip(X[col].fillna(0), a_min=0, a_max=None))

    return X


def make_preprocessor(X: pd.DataFrame, dense: bool = True) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for mixed numerical/categorical features."""
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def make_full_preprocess_pipeline(X: pd.DataFrame, dense: bool = True) -> Pipeline:
    """Create a reusable preprocessing pipeline.

    The first step applies custom feature cleaning. The second step handles
    missing values, scaling, and one-hot encoding.
    """
    return Pipeline(
        steps=[
            ("clean", FunctionTransformer(clean_feature_values, validate=False)),
            ("preprocess", make_preprocessor(X, dense=dense)),
        ]
    )
