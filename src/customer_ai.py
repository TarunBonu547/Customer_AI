"""Reusable data validation and clustering helpers for Customer AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]


def validate_model_artifacts(
    scaler: Any,
    pca: Any,
    model: Any,
    expected_features: list[str] = FEATURES,
) -> None:
    """Validate compatibility between scaler, PCA, and clustering artifacts."""
    errors = []
    expected_count = len(expected_features)

    scaler_count = getattr(scaler, "n_features_in_", None)
    if scaler_count != expected_count:
        errors.append(
            f"scaler expects {scaler_count} features; expected {expected_count}"
        )
    scaler_names = getattr(scaler, "feature_names_in_", None)
    if scaler_names is not None and list(scaler_names) != expected_features:
        errors.append("scaler feature order does not match the required schema")

    pca_input_count = getattr(pca, "n_features_in_", None)
    if pca_input_count != expected_count:
        errors.append(
            f"PCA expects {pca_input_count} input features; expected {expected_count}"
        )
    pca_components = getattr(pca, "components_", None)
    if pca_components is None or pca_components.ndim != 2:
        errors.append("PCA components are missing or malformed")
        pca_component_count = None
    else:
        pca_component_count = pca_components.shape[0]
        if pca_components.shape[1] != expected_count:
            errors.append("PCA component width does not match the feature count")

    model_count = getattr(model, "n_features_in_", None)
    if pca_component_count is not None and model_count != pca_component_count:
        errors.append(
            f"clustering model expects {model_count} transformed features; "
            f"PCA produces {pca_component_count}"
        )
    centers = getattr(model, "cluster_centers_", None)
    cluster_count = getattr(model, "n_clusters", None)
    if centers is None or centers.ndim != 2:
        errors.append("clustering centers are missing or malformed")
    elif cluster_count != centers.shape[0]:
        errors.append("declared cluster count does not match cluster centers")
    if cluster_count is None or cluster_count < 2:
        errors.append("clustering model must define at least two clusters")

    if errors:
        raise ValueError("Incompatible model artifacts: " + "; ".join(errors))


def load_model_artifacts(
    model_dir: str | Path,
    expected_features: list[str] = FEATURES,
) -> tuple[Any, Any, Any]:
    """Load and validate the saved scaler, PCA, and clustering artifacts."""
    artifact_dir = Path(model_dir)
    paths = {
        "scaler": artifact_dir / "scaler.pkl",
        "PCA": artifact_dir / "pca.pkl",
        "K-Means": artifact_dir / "kmeans.pkl",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): " + ", ".join(missing)
        )

    try:
        scaler = joblib.load(paths["scaler"])
        pca = joblib.load(paths["PCA"])
        model = joblib.load(paths["K-Means"])
    except Exception as exc:
        raise RuntimeError(
            "Could not load the saved model artifacts. "
            "They may be corrupted or incompatible with this environment."
        ) from exc

    validate_model_artifacts(scaler, pca, model, expected_features)
    return scaler, pca, model


def validate_customer_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a customer spending dataframe.

    Extra columns are retained, while the six required spending columns are
    converted to numeric values and returned in the original column order.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Customer data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Customer data must contain at least one row.")

    missing_columns = [feature for feature in FEATURES if feature not in data]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    validated = data.copy()
    numeric_data = validated[FEATURES].apply(pd.to_numeric, errors="coerce")
    invalid_mask = numeric_data.isna() | ~np.isfinite(numeric_data)
    if invalid_mask.any().any():
        invalid_count = int(invalid_mask.sum().sum())
        raise ValueError(
            f"Customer data contains {invalid_count} missing, non-numeric, "
            "or non-finite spending values."
        )
    if (numeric_data < 0).any().any():
        raise ValueError("Spending values cannot be negative.")

    validated[FEATURES] = numeric_data.astype(float)
    return validated


def transform_customer_data(
    data: pd.DataFrame,
    scaler: Any,
    pca: Any,
) -> np.ndarray:
    """Apply the training log, scaling, and PCA transformations."""
    validated = validate_customer_data(data)
    transformed = pca.transform(scaler.transform(np.log1p(validated[FEATURES])))
    if not np.isfinite(transformed).all():
        raise ValueError("The transformed customer data contains non-finite values.")
    return transformed


def predict_customers(
    data: pd.DataFrame,
    scaler: Any,
    pca: Any,
    model: Any,
) -> np.ndarray:
    """Validate customer data, transform it, and return cluster labels."""
    transformed = transform_customer_data(data, scaler, pca)
    return model.predict(transformed)


def evaluate_clustering(data: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calculate clustering metrics from transformed data and model labels."""
    values = np.asarray(data)
    cluster_labels = np.asarray(labels)
    if values.ndim != 2 or len(values) != len(cluster_labels):
        raise ValueError("Data and labels must contain the same number of rows.")
    unique_labels = np.unique(cluster_labels)
    if len(values) < 3 or len(unique_labels) < 2 or len(unique_labels) >= len(values):
        raise ValueError(
            "Clustering metrics require at least two non-singleton clusters."
        )

    return {
        "silhouette": float(silhouette_score(values, cluster_labels)),
        "davies_bouldin": float(davies_bouldin_score(values, cluster_labels)),
        "calinski_harabasz": float(
            calinski_harabasz_score(values, cluster_labels)
        ),
    }
