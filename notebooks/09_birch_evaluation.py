import pandas as pd
import numpy as np

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

        return np.sqrt(
            np.sum(variance)
        )

    def add_point(self, point):

        self.n += 1
        self.linear_sum += point
        self.squared_sum += point ** 2


# ==================================================
# 3. BIRCH From Scratch
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
        self.centroids = None
        self.labels = None

    def find_nearest_cf(self, point):

        if not self.cf_entries:
            return None

        distances = [
            np.linalg.norm(
                point - cf.centroid()
            )
            for cf in self.cf_entries
        ]

        return np.argmin(distances)

    def build_cf_tree(self, X):

        for point in X:

            nearest_index = (
                self.find_nearest_cf(point)
            )

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

            if new_radius <= self.threshold:

                nearest_cf.add_point(point)

            else:

                self.cf_entries.append(
                    ClusteringFeature(point)
                )

    def get_cf_centroids(self):

        return np.array([
            cf.centroid()
            for cf in self.cf_entries
        ])

    def final_clustering(self):

        cf_centroids = (
            self.get_cf_centroids()
        )

        np.random.seed(42)

        indices = np.random.choice(
            len(cf_centroids),
            self.n_clusters,
            replace=False
        )

        centroids = (
            cf_centroids[indices].copy()
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
                new_centroids - centroids
            )

            centroids = new_centroids

            if shift < 0.0001:
                break

        self.centroids = centroids

    def fit(self, X):

        self.build_cf_tree(X)

        self.final_clustering()

        assert self.centroids is not None

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
# 4. Test Different Thresholds
# ==================================================

thresholds = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5
]

print("\nBIRCH Threshold Evaluation")
print("=" * 60)

results = []


for threshold in thresholds:

    model = BirchScratch(
        n_clusters=3,
        threshold=threshold
    )

    model.fit(X)

    labels = model.labels
    if labels is None:
        raise RuntimeError("BIRCH model did not produce cluster labels")

    cf_count = len(
        model.cf_entries
    )

    silhouette = silhouette_score(
        X,
        labels
    )

    results.append({
        "Threshold": threshold,
        "CF Entries": cf_count,
        "Silhouette Score": silhouette
    })

    print(
        f"Threshold = {threshold:.2f} | "
        f"CF Entries = {cf_count} | "
        f"Silhouette = {silhouette:.4f}"
    )


# ==================================================
# 5. Display Results
# ==================================================

results_df = pd.DataFrame(
    results
)

print("\nEvaluation Summary:")
print(
    results_df.to_string(
        index=False
    )
)


# ==================================================
# 6. Best Threshold
# ==================================================

best_index = (
    results_df[
        "Silhouette Score"
    ].idxmax()
)

best_threshold = (
    results_df.loc[
        best_index,
        "Threshold"
    ]
)

best_score = (
    results_df.loc[
        best_index,
        "Silhouette Score"
    ]
)

print(
    f"\nBest Threshold: "
    f"{best_threshold}"
)

print(
    f"Best Silhouette Score: "
    f"{best_score:.4f}"
)