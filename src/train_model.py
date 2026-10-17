"""Train and persist the Customer AI preprocessing and clustering models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.customer_ai import FEATURES, evaluate_clustering, validate_customer_data

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "wholesale_customers.csv"
DEFAULT_MODEL_DIR = BASE_DIR / "models"


def train_models(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    n_clusters: int = 3,
    n_components: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit and persist the preprocessing and K-Means pipeline."""
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if n_components < 1:
        raise ValueError("n_components must be positive.")

    data = validate_customer_data(pd.read_csv(data_path))
    if n_clusters >= len(data):
        raise ValueError("n_clusters must be smaller than the number of rows.")
    if n_components > min(len(data), len(FEATURES)):
        raise ValueError(
            "n_components cannot exceed the number of rows or features."
        )

    X_log = np.log1p(data[FEATURES])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X_pca)
    metrics = evaluate_clustering(X_pca, labels)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(pca, output_dir / "pca.pkl")
    joblib.dump(kmeans, output_dir / "kmeans.pkl")
    joblib.dump(
        {
            "features": FEATURES,
            "n_clusters": n_clusters,
            "n_components": n_components,
            "random_state": random_state,
            "training_rows": len(data),
            "variance_retained": float(pca.explained_variance_ratio_.sum()),
            "metrics": metrics,
        },
        output_dir / "metadata.pkl",
    )

    return {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse optional command-line training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Train models from command-line options and print verified results."""
    args = parse_args()
    result = train_models(
        args.data,
        args.model_dir,
        n_clusters=args.clusters,
        n_components=args.components,
        random_state=args.random_state,
    )
    pca = result["pca"]
    kmeans = result["kmeans"]

    print("=" * 50)
    print("REAL-TIME ML PIPELINE TRAINED")
    print("=" * 50)
    print(f"Features             : {len(FEATURES)}")
    print(f"PCA components       : {pca.n_components_}")
    print(f"Variance retained    : {pca.explained_variance_ratio_.sum() * 100:.2f}%")
    print(f"K-Means clusters     : {kmeans.n_clusters}")
    print(f"K-Means inertia      : {kmeans.inertia_:.4f}")
    print(f"Metrics              : {result['metrics']}")
    print(f"Saved models         : {args.model_dir}")


if __name__ == "__main__":
    main()
