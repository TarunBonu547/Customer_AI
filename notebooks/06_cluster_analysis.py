import pandas as pd
import numpy as np


# ==================================================
# 1. Load Original Dataset
# ==================================================

df = pd.read_csv(
    "data/wholesale_customers.csv"
)


# ==================================================
# 2. Load PCA Dataset
# ==================================================

pca_df = pd.read_csv(
    "data/pca_4_customers.csv"
)

X = pca_df.values


# ==================================================
# 3. K-Means Scratch
# ==================================================

class KMeansScratch:

    def __init__(
        self,
        k=3,
        max_iterations=100,
        random_state=42
    ):

        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):

        np.random.seed(
            self.random_state
        )

        indices = np.random.choice(
            len(X),
            self.k,
            replace=False
        )

        self.centroids = X[
            indices
        ].copy()

    def assign_clusters(self, X):

        if self.centroids is None:

            raise RuntimeError(
                "Centroids must be initialized before assigning clusters."
            )

        distances = np.sqrt(
            np.sum(
                (
                    X[:, np.newaxis, :]
                    - self.centroids[np.newaxis, :, :]
                ) ** 2,
                axis=2
            )
        )

        return np.argmin(
            distances,
            axis=1
        )

    def update_centroids(
        self,
        X,
        labels
    ):

        if self.centroids is None:

            raise RuntimeError(
                "Centroids must be initialized before updating centroids."
            )

        new_centroids = []

        for cluster in range(self.k):

            points = X[
                labels == cluster
            ]

            if len(points) > 0:

                centroid = points.mean(
                    axis=0
                )

            else:

                centroid = self.centroids[
                    cluster
                ]

            new_centroids.append(
                centroid
            )

        return np.array(
            new_centroids
        )

    def fit(self, X):

        self.initialize_centroids(X)

        for _ in range(
            self.max_iterations
        ):

            labels = self.assign_clusters(
                X
            )

            new_centroids = (
                self.update_centroids(
                    X,
                    labels
                )
            )

            shift = np.linalg.norm(
                new_centroids
                - self.centroids
            )

            self.centroids = (
                new_centroids
            )

            if shift < 0.0001:

                break

        self.labels = (
            self.assign_clusters(X)
        )

        return self


# ==================================================
# 4. Analyze K = 2, 3 and 4
# ==================================================

features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]


for k in [2, 3, 4]:

    print("\n")
    print("=" * 60)
    print(f"K-MEANS CLUSTER PROFILE ANALYSIS: K = {k}")
    print("=" * 60)

    model = KMeansScratch(
        k=k,
        max_iterations=100,
        random_state=42
    )

    model.fit(X)

    labels = model.labels

    # Add labels to original data
    analysis_df = df[
        features
    ].copy()

    analysis_df["Cluster"] = labels

    # Cluster sizes
    print("\nCluster Sizes:")

    cluster_sizes = (
        analysis_df[
            "Cluster"
        ]
        .value_counts()
        .sort_index()
    )

    for cluster, size in cluster_sizes.items():

        print(
            f"Cluster {cluster}: "
            f"{size} customers"
        )

    # Average spending
    print("\nAverage Spending by Cluster:")

    profile = (
        analysis_df
        .groupby("Cluster")[features]
        .mean()
        .round(2)
    )

    print(profile)

    # Median spending
    print("\nMedian Spending by Cluster:")

    median_profile = (
        analysis_df
        .groupby("Cluster")[features]
        .median()
        .round(2)
    )

    print(median_profile)