import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.customer_ai import (
    FEATURES,
    evaluate_clustering,
    load_model_artifacts,
    predict_customers,
    transform_customer_data,
    validate_model_artifacts,
    validate_customer_data,
)
from src.train_model import train_models


def make_customer_data(rows=12):
    values = np.arange(rows * len(FEATURES), dtype=float).reshape(
        rows, len(FEATURES)
    )
    return pd.DataFrame(values + 1, columns=FEATURES)


def fit_models(data):
    transformed = np.log1p(data[FEATURES])
    scaler = StandardScaler().fit(transformed)
    scaled = scaler.transform(transformed)
    pca = PCA(n_components=2, random_state=7).fit(scaled)
    reduced = pca.transform(scaled)
    model = KMeans(n_clusters=3, random_state=7, n_init=10).fit(reduced)
    return scaler, pca, model


def test_validate_customer_data_converts_numeric_strings_and_keeps_extra_columns():
    data = make_customer_data(3).astype(object)
    data["customer_id"] = ["a", "b", "c"]
    data["Fresh"] = ["1", "2", "3"]

    validated = validate_customer_data(data)

    assert validated[FEATURES].dtypes.eq(float).all()
    assert validated["customer_id"].tolist() == ["a", "b", "c"]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.drop(columns="Milk"), "Missing required columns"),
        (lambda frame: frame.assign(Fresh=[1, np.inf] + [1] * 10), "non-finite"),
        (lambda frame: frame.assign(Grocery=[1, -1] + [1] * 10), "negative"),
        (lambda frame: frame.assign(Frozen=[1, "bad"] + [1] * 10), "non-numeric"),
    ],
)
def test_validate_customer_data_rejects_invalid_values(mutator, message):
    with pytest.raises(ValueError, match=message):
        validate_customer_data(mutator(make_customer_data()))


def test_validate_customer_data_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one row"):
        validate_customer_data(pd.DataFrame(columns=FEATURES))


def test_prediction_uses_training_feature_order_and_returns_labels():
    data = make_customer_data()
    scaler, pca, model = fit_models(data)
    reordered = data[[*reversed(FEATURES)]]

    transformed = transform_customer_data(reordered, scaler, pca)
    predictions = predict_customers(reordered, scaler, pca, model)

    assert transformed.shape == (len(data), 2)
    assert predictions.shape == (len(data),)
    assert set(predictions).issubset(set(model.labels_))


def test_evaluate_clustering_returns_metrics_from_labels():
    values = np.array([[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]])
    labels = np.array([0, 0, 1, 1, 2, 2])

    metrics = evaluate_clustering(values, labels)

    assert set(metrics) == {"silhouette", "davies_bouldin", "calinski_harabasz"}
    assert 0 < metrics["silhouette"] <= 1
    assert metrics["davies_bouldin"] > 0
    assert metrics["calinski_harabasz"] > 0


@pytest.mark.parametrize(
    "values, labels",
    [
        (np.ones((2, 2)), np.array([0, 1])),
        (np.ones((4, 2)), np.array([0, 0, 0, 0])),
        (np.ones((4, 2)), np.array([0, 1, 2, 3])),
    ],
)
def test_evaluate_clustering_rejects_unsuitable_inputs(values, labels):
    with pytest.raises(ValueError, match="require"):
        evaluate_clustering(values, labels)


def test_train_models_uses_configured_paths_and_writes_metadata(tmp_path):
    data_path = tmp_path / "customers.csv"
    model_dir = tmp_path / "models"
    make_customer_data(18).to_csv(data_path, index=False)

    result = train_models(
        data_path,
        model_dir,
        n_clusters=3,
        n_components=3,
        random_state=19,
    )

    assert result["kmeans"].random_state == 19
    assert set(result["metrics"]) == {
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    }
    assert all(
        (model_dir / filename).exists()
        for filename in ["scaler.pkl", "pca.pkl", "kmeans.pkl", "metadata.pkl"]
    )

    metadata = joblib.load(model_dir / "metadata.pkl")
    assert metadata["features"] == FEATURES
    assert metadata["random_state"] == 19


def test_valid_saved_artifacts_load_and_validate(tmp_path):
    data = make_customer_data(18)
    scaler, pca, model = fit_models(data)
    for name, artifact in {
        "scaler.pkl": scaler,
        "pca.pkl": pca,
        "kmeans.pkl": model,
    }.items():
        joblib.dump(artifact, tmp_path / name)

    loaded = load_model_artifacts(tmp_path)

    validate_model_artifacts(*loaded)
    assert loaded[2].n_clusters == 3


def test_artifact_validation_rejects_feature_order_mismatch():
    data = make_customer_data(18)
    scaler, pca, model = fit_models(data)
    reordered = data[list(reversed(FEATURES))]
    reordered_scaler = StandardScaler().fit(np.log1p(reordered))

    with pytest.raises(ValueError, match="feature order"):
        validate_model_artifacts(reordered_scaler, pca, model)


def test_artifact_loader_reports_missing_and_corrupt_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing model artifact"):
        load_model_artifacts(tmp_path)

    for name in ["scaler.pkl", "pca.pkl", "kmeans.pkl"]:
        (tmp_path / name).write_bytes(b"not a pickle")

    with pytest.raises(RuntimeError, match="Could not load"):
        load_model_artifacts(tmp_path)
