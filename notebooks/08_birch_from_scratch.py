import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. Load PCA Dataset
# ==================================================

df = pd.read_csv(
    "data/pca_4_customers.csv"
)

X = df.values

print("Dataset Shape:", X.shape)


# ==================================================
# 2. Clustering Feature
# ==================================================

class ClusteringFeature:

    def __init__(self, point):

        self.n = 1

        self.linear_sum = point.copy()

        self.squared_sum = point ** 2

    def centroid(self):

        return self.linear_sum / self.n

    def radius_if_added(self, point):

        new_n = self.n + 1

        new_ls = self.linear_sum + point

        new_ss = self.squared_sum + point ** 2

        new_centroid = new_ls / new_n

        variance = (
            new_ss / new_n
            - new_centroid ** 2
        )

        variance = np.maximum(
            variance,
            0
        )

        radius = np.sqrt(
            np.sum(variance)
        )

        return radius

    def add_point(self, point):

        self.n += 1

        self.linear_sum += point

        self.squared_sum += point ** 2


# ==================================================
# 3. Simplified BIRCH Algorithm
# ==================================================

class BirchScratch:

    def __init__(
        self,
        n_clusters=3,
        threshold=1.0
    ):

        self.n_clusters = n_clusters

        self.threshold = threshold

        self.cf_entries = []

        self.labels = None

        self.centroids = None


    # ------------------------------------------------
    # Find Nearest CF
    # ------------------------------------------------

    def find_nearest_cf(self, point):

        if not self.cf_entries:

            return None

        distances = []

        for cf in self.cf_entries:

            distance = np.linalg.norm(
                point - cf.centroid()
            )

            distances.append(distance)

        return np.argmin(distances)


    # ------------------------------------------------
    # Build CF Structure
    # ------------------------------------------------

    def build_cf_tree(self, X):

        for point in X:

            nearest_index = (
                self.find_nearest_cf(point)
            )

            # First point
            if nearest_index is None:

                self.cf_entries.append(
                    ClusteringFeature(point)
                )

                continue


            nearest_cf = (
                self.cf_entries[
                    nearest_index
                ]
            )

            new_radius = (
                nearest_cf.radius_if_added(
                    point
                )
            )


            # Add point if threshold is satisfied
            if new_radius <= self.threshold:

                nearest_cf.add_point(point)

            else:

                self.cf_entries.append(
                    ClusteringFeature(point)
                )


    # ------------------------------------------------
    # Get CF Centroids
    # ------------------------------------------------

    def get_cf_centroids(self):

        return np.array(
            [
                cf.centroid()
                for cf in self.cf_entries
            ]
        )


    # ------------------------------------------------
    # Final K-Means on CF Centroids
    # ------------------------------------------------

    def final_clustering(self):

        cf_centroids = (
            self.get_cf_centroids()
        )

        # Simple K-Means using CF centroids

        np.random.seed(42)

        indices = np.random.choice(
            len(cf_centroids),
            self.n_clusters,
            replace=False
        )

        centroids = (
            cf_centroids[
                indices
            ].copy()
        )

        labels = np.empty(
            len(cf_centroids),
            dtype=int
        )


        for _ in range(100):

            distances = np.sqrt(
                np.sum(
                    (
                        cf_centroids[:, None, :]
                        - centroids[None, :, :]
                    ) ** 2,
                    axis=2
                )
            )

            labels = np.argmin(
                distances,
                axis=1
            )


            new_centroids = []

            for cluster in range(
                self.n_clusters
            ):

                points = (
                    cf_centroids[
                        labels == cluster
                    ]
                )

                if len(points) > 0:

                    new_centroids.append(
                        points.mean(axis=0)
                    )

                else:

                    new_centroids.append(
                        centroids[cluster]
                    )

            new_centroids = np.array(
                new_centroids
            )


            shift = np.linalg.norm(
                new_centroids
                - centroids
            )

            centroids = new_centroids


            if shift < 0.0001:

                break


        self.cf_centroids = cf_centroids

        self.cf_labels = labels

        self.centroids = centroids


    # ------------------------------------------------
    # Fit
    # ------------------------------------------------

    def fit(self, X):

        self.build_cf_tree(X)

        print(
            "\nNumber of CF Entries:",
            len(self.cf_entries)
        )

        self.final_clustering()

        # Assign original points to nearest final centroid

        if self.centroids is None:
            raise RuntimeError(
                "Final clustering did not produce centroids."
            )

        distances = np.sqrt(
            np.sum(
                (
                    X[:, None, :]
                    - self.centroids[None, :, :]
                ) ** 2,
                axis=2
            )
        )

        self.labels = np.argmin(
            distances,
            axis=1
        )

        return self


# ==================================================
# 4. Train BIRCH
# ==================================================

birch = BirchScratch(
    n_clusters=3,
    threshold=1.0
)

birch.fit(X)


# ==================================================
# 5. Display Results
# ==================================================

print("\nFinal BIRCH Centroids:")

print(
    birch.centroids
)


print("\nCluster Sizes:")

labels = birch.labels
if labels is None:
    raise RuntimeError("BIRCH did not produce cluster labels")

unique, counts = np.unique(
    labels,
    return_counts=True
)

for cluster, count in zip(
    unique,
    counts
):

    print(
        f"Cluster {cluster}: "
        f"{count} customers"
    )


# ==================================================
# 6. Save Results
# ==================================================

result = df.copy()

result["Cluster"] = (
    birch.labels
)

result.to_csv(
    "data/birch_scratch_results.csv",
    index=False
)

print(
    "\nBIRCH scratch results "
    "saved successfully!"
)


# ==================================================
# 7. Visualization
# ==================================================

plt.figure(
    figsize=(9, 6)
)

for cluster in range(3):

    points = X[
        birch.labels == cluster
    ]

    plt.scatter(
        points[:, 0],
        points[:, 1],
        label=f"Cluster {cluster}"
    )


if birch.centroids is not None:

    plt.scatter(
        birch.centroids[:, 0],
        birch.centroids[:, 1],
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
    "BIRCH Clustering From Scratch"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/birch_scratch.png",
    dpi=300
)

plt.show()