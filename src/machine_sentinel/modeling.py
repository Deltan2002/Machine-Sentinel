"""Model training and evaluation without CLI or visualization concerns."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EDGE_FEATURES = ("rms", "ptp")
EXPERIMENTS = {
    "rms_only": ("rms",),
    "time_domain": EDGE_FEATURES,
    "frequency_domain": (
        "dominant_freq",
        "energy_low",
        "energy_mid",
        "energy_high",
    ),
    "all_features": (
        "rms",
        "ptp",
        "dominant_freq",
        "energy_low",
        "energy_mid",
        "energy_high",
    ),
}


def split_by_recording(
    features: pd.DataFrame,
    test_suffix: str = "_3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep whole recordings together when creating train and test sets."""
    is_test = features["recording_id"].astype(str).str.endswith(test_suffix)
    train = features.loc[~is_test].copy()
    test = features.loc[is_test].copy()
    if train.empty or test.empty:
        raise ValueError(f"The '{test_suffix}' split produced an empty train or test set")
    return train, test


def train_classifier(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Pipeline:
    """Fit the scaler and logistic regression as one deployable pipeline."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    return model.fit(train.loc[:, feature_columns], train["label"])


def evaluate_classifier(
    model: Pipeline,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, object]:
    """Return JSON-safe classification metrics."""
    predictions = model.predict(test.loc[:, feature_columns])
    return {
        "accuracy": float(accuracy_score(test["label"], predictions)),
        "confusion_matrix": confusion_matrix(test["label"], predictions).tolist(),
    }


def deployment_constants(
    model: Pipeline,
    feature_columns: Sequence[str],
) -> dict[str, object]:
    """Extract scaler and classifier values needed by the firmware."""
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "features": list(feature_columns),
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "weights": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
    }
