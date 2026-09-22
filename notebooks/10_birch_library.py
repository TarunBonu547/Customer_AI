import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import Birch
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
# 2. Create BIRCH Model
# ==================================================

birch = Birch(
    threshold=1.0,
    branching_factor=50,
    n_clusters=3
)


# ==================================================
# 3. Train Model
# ==================================================

birch.fit(X)


# ==================================================
# 4. Get Cluster Labels
# ==================================================

labels = birch.labels_


# ==================================================
# 5. Cluster Sizes
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
# 6. Silhouette Score
# ==================================================

silhouette = silhouette_score(
    X,
    labels
)

print(
    f"\nSilhouette Score: "
    f"{silhouette:.4f}"
)


# ==================================================
# 7. Number of Subclusters
# ==================================================

subcluster_labels = (
    birch.subcluster_labels_
)

number_of_subclusters = len(
    set(subcluster_labels)
)

print(
    "\nNumber of BIRCH Subclusters:",
    number_of_subclusters
)


# ==================================================
# 8. Save Results
# ==================================================

result = df.copy()

result["Cluster"] = labels

result.to_csv(
    "data/birch_library_results.csv",
    index=False
)

print(
    "\nBIRCH library results "
    "saved successfully!"
)


# ==================================================
# 9. Visualization
# ==================================================

plt.figure(
    figsize=(9, 6)
)

for cluster in sorted(
    set(labels)
):

    points = X[
        labels == cluster
    ]

    plt.scatter(
        points[:, 0],
        points[:, 1],
        label=f"Cluster {cluster}"
    )


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "BIRCH Clustering Using Scikit-learn"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/birch_library.png",
    dpi=300
)

plt.show()