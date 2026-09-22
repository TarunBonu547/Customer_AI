import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==================================================
# 1. Load PCA Dataset
# ==================================================

df = pd.read_csv(
    "data/pca_4_customers.csv"
)

X = df.values

print("Dataset Shape:", X.shape)


# ==================================================
# 2. Create K-Means Model
# ==================================================

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42
)


# ==================================================
# 3. Train Model
# ==================================================

kmeans.fit(X)


# ==================================================
# 4. Get Cluster Labels
# ==================================================

labels = kmeans.labels_


# ==================================================
# 5. Get Centroids
# ==================================================

centroids = kmeans.cluster_centers_

print("\nFinal Cluster Centroids:")
print(centroids)


# ==================================================
# 6. Cluster Sizes
# ==================================================

print("\nCluster Sizes:")

unique, counts = pd.Series(
    labels
).value_counts().sort_index().index, pd.Series(
    labels
).value_counts().sort_index().values

for cluster, count in zip(
    unique,
    counts
):

    print(
        f"Cluster {cluster}: "
        f"{count} customers"
    )


# ==================================================
# 7. Calculate Inertia
# ==================================================

inertia = kmeans.inertia_

print(
    f"\nFinal Inertia: {inertia:.4f}"
)


# ==================================================
# 8. Calculate Silhouette Score
# ==================================================

silhouette = silhouette_score(
    X,
    labels
)

print(
    f"Silhouette Score: "
    f"{silhouette:.4f}"
)


# ==================================================
# 9. Save Results
# ==================================================

result = df.copy()

result["Cluster"] = labels

result.to_csv(
    "data/kmeans_library_results.csv",
    index=False
)

print(
    "\nK-Means library results "
    "saved successfully!"
)


# ==================================================
# 10. Visualize Clusters
# ==================================================

plt.figure(figsize=(9, 6))

for cluster in range(3):

    cluster_points = X[
        labels == cluster
    ]

    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {cluster}"
    )


# Plot centroids

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "K-Means Clustering Using Scikit-learn"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/kmeans_library.png",
    dpi=300
)

plt.show()