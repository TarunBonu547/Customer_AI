import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/wholesale_customers.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X = df[features].copy()


# -----------------------------
# Preprocessing
# -----------------------------
# Reduce right-skewness
X_log = np.log1p(X)


# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)


# -----------------------------
# PCA
# -----------------------------
# Keep 4 components (~92% variance)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)


# -----------------------------
# K-Means
# -----------------------------
n_clusters = 3
kmeans = KMeans(
    n_clusters=n_clusters,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(X_pca)


# -----------------------------
# Save models
# -----------------------------
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans.pkl"))


# -----------------------------
# Display results
# -----------------------------
print("=" * 50)
print("REAL-TIME ML PIPELINE TRAINED")
print("=" * 50)

print(f"Dataset shape       : {df.shape}")
print(f"Features             : {len(features)}")
print(f"PCA components       : {pca.n_components_}")
print(
    f"Variance retained    : "
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
)
print(f"K-Means clusters     : {n_clusters}")
print(f"K-Means inertia      : {kmeans.inertia_:.4f}")

print("\nSaved models:")

for filename in [
    "scaler.pkl",
    "pca.pkl",
    "kmeans.pkl"
]:
    print(f"  models/{filename}")

print("\nTraining complete!")