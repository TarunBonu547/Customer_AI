import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. Load PCA Dataset
# ==================================================

df = pd.read_csv("data/pca_4_customers.csv")

X = df.values

print("Dataset Shape:", X.shape)


# ==================================================
# 2. K-Means From Scratch
# ==================================================

class KMeansScratch:

    def __init__(self, k=4, max_iterations=100, random_state=42):

        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    # ------------------------------------------------
    # Calculate Euclidean Distance
    # ------------------------------------------------

    def euclidean_distance(self, point, centroid):

        return np.sqrt(
            np.sum((point - centroid) ** 2)
        )

    # ------------------------------------------------
    # Initialize Centroids
    # ------------------------------------------------

    def initialize_centroids(self, X):

        np.random.seed(self.random_state)

        random_indices = np.random.choice(
            len(X),
            self.k,
            replace=False
        )

        self.centroids = X[random_indices].copy()

    # ------------------------------------------------
    # Assign Points to Nearest Centroid
    # ------------------------------------------------

    def assign_clusters(self, X):

        if self.centroids is None:
            raise RuntimeError(
                "Centroids must be initialized before assigning clusters."
            )

        labels = []

        for point in X:

            distances = [
                self.euclidean_distance(point, centroid)
                for centroid in self.centroids
            ]

            nearest_cluster = np.argmin(distances)

            labels.append(nearest_cluster)

        return np.array(labels)

    # ------------------------------------------------
    # Update Centroids
    # ------------------------------------------------

    def update_centroids(self, X, labels):

        if self.centroids is None:
            raise RuntimeError(
                "Centroids must be initialized before updating them."
            )

        new_centroids = []

        for cluster in range(self.k):

            cluster_points = X[labels == cluster]

            if len(cluster_points) > 0:

                new_centroid = cluster_points.mean(
                    axis=0
                )

            else:

                new_centroid = self.centroids[cluster]

            new_centroids.append(new_centroid)

        return np.array(new_centroids)

    # ------------------------------------------------
    # Calculate Within-Cluster Sum of Squares
    # ------------------------------------------------

    def calculate_inertia(self, X, labels):

        if self.centroids is None:
            raise RuntimeError(
                "Centroids must be initialized before calculating inertia."
            )

        inertia = 0

        for cluster in range(self.k):

            cluster_points = X[labels == cluster]

            if len(cluster_points) > 0:

                distances = np.sum(
                    (cluster_points - self.centroids[cluster]) ** 2
                )

                inertia += distances

        return inertia

    # ------------------------------------------------
    # Fit K-Means
    # ------------------------------------------------

    def fit(self, X):

        self.initialize_centroids(X)

        print("\nInitial Centroids:")
        print(self.centroids)

        for iteration in range(
            self.max_iterations
        ):

            # Assign clusters
            labels = self.assign_clusters(X)

            # Calculate new centroids
            new_centroids = self.update_centroids(
                X,
                labels
            )

            # Check convergence
            centroid_shift = np.linalg.norm(
                new_centroids - self.centroids
            )

            self.centroids = new_centroids

            print(
                f"Iteration {iteration + 1}: "
                f"Centroid Shift = "
                f"{centroid_shift:.6f}"
            )

            if centroid_shift < 0.0001:

                print(
                    "\nK-Means converged!"
                )

                break

        self.labels = self.assign_clusters(X)

        return self

    # ------------------------------------------------
    # Predict New Data
    # ------------------------------------------------

    def predict(self, X):

        return self.assign_clusters(X)


# ==================================================
# 3. Train K-Means
# ==================================================

kmeans = KMeansScratch(
    k=3,
    max_iterations=100,
    random_state=42
)

kmeans.fit(X)


# ==================================================
# 4. Get Cluster Labels
# ==================================================

labels = kmeans.labels

if labels is None:
    raise RuntimeError(
        "Cluster labels are unavailable. Fit the model before using them."
    )

print("\nFinal Cluster Centroids:")
print(kmeans.centroids)


# ==================================================
# 5. Display Cluster Sizes
# ==================================================

print("\nCluster Sizes:")

unique, counts = np.unique(
    labels,
    return_counts=True
)

for cluster, count in zip(unique, counts):

    print(
        f"Cluster {cluster}: "
        f"{count} customers"
    )


# ==================================================
# 6. Calculate Inertia
# ==================================================

inertia = kmeans.calculate_inertia(
    X,
    labels
)

print(
    f"\nFinal Inertia: {inertia:.4f}"
)


# ==================================================
# 7. Create Result Dataset
# ==================================================

result = df.copy()

result["Cluster"] = labels

result.to_csv(
    "data/kmeans_scratch_results.csv",
    index=False
)

print(
    "\nK-Means results saved successfully!"
)


# ==================================================
# 8. Visualize Clusters
# ==================================================

plt.figure(figsize=(9, 6))

for cluster in range(kmeans.k):

    cluster_points = X[
        labels == cluster
    ]

    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {cluster}"
    )


# Plot centroids

if kmeans.centroids is None:
    raise RuntimeError("K-Means centroids are unavailable after fitting.")

plt.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title(
    "K-Means Clustering From Scratch"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/kmeans_scratch.png",
    dpi=300
)

plt.show()