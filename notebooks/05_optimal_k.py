import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# ==================================================
# 1. Load PCA Dataset
# ==================================================

df = pd.read_csv("data/pca_4_customers.csv")

X = df.values

print("Dataset Shape:", X.shape)


# ==================================================
# 2. K-Means Scratch Implementation
# ==================================================

class KMeansScratch:

    def __init__(self, k=4, max_iterations=100, random_state=42):

        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):

        np.random.seed(self.random_state)

        indices = np.random.choice(
            len(X),
            self.k,
            replace=False
        )

        self.centroids = X[indices].copy()

    def assign_clusters(self, X):

        labels = []

        for point in X:

            distances = np.sqrt(
                np.sum(
                    (self.centroids - point) ** 2,
                    axis=1
                )
            )

            labels.append(
                np.argmin(distances)
            )

        return np.array(labels)

    def update_centroids(self, X, labels):

        new_centroids = []

        if self.centroids is None:
            raise RuntimeError("Centroids must be initialized before updating")

        for cluster in range(self.k):

            points = X[
                labels == cluster
            ]

            if len(points) > 0:

                centroid = points.mean(axis=0)

            else:

                centroid = self.centroids[cluster]

            new_centroids.append(centroid)

        return np.array(new_centroids)

    def calculate_inertia(self, X, labels):

        if self.centroids is None:
            raise RuntimeError("Centroids must be initialized before calculating inertia")

        inertia = 0

        for cluster in range(self.k):

            points = X[
                labels == cluster
            ]

            if len(points) > 0:

                inertia += np.sum(
                    (points - self.centroids[cluster]) ** 2
                )

        return inertia

    def fit(self, X):

        self.initialize_centroids(X)

        for _ in range(self.max_iterations):

            labels = self.assign_clusters(X)

            new_centroids = self.update_centroids(
                X,
                labels
            )

            shift = np.linalg.norm(
                new_centroids - self.centroids
            )

            self.centroids = new_centroids

            if shift < 0.0001:

                break

        self.labels = self.assign_clusters(X)

        return self


# ==================================================
# 3. Test Different Values of K
# ==================================================

k_values = range(2, 9)

inertias = []
silhouette_scores = []


print("\nK-Means Evaluation")
print("=" * 45)


for k in k_values:

    model = KMeansScratch(
        k=k,
        max_iterations=100,
        random_state=42
    )

    model.fit(X)

    labels = model.labels
    if labels is None:
        raise RuntimeError("K-Means did not produce labels")

    inertia = model.calculate_inertia(
        X,
        labels
    )

    silhouette = silhouette_score(
        X,
        labels
    )

    inertias.append(inertia)

    silhouette_scores.append(
        silhouette
    )

    print(
        f"K = {k} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette Score = {silhouette:.4f}"
    )


# ==================================================
# 4. Find Best K According to Silhouette Score
# ==================================================

best_index = np.argmax(
    silhouette_scores
)

best_k = list(k_values)[best_index]

best_score = silhouette_scores[best_index]


print("\nBest K According to Silhouette Score:")
print("K =", best_k)
print(
    "Silhouette Score =",
    round(best_score, 4)
)


# ==================================================
# 5. Elbow Plot
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.title(
    "Elbow Method for Optimal K"
)

plt.xticks(list(k_values))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/elbow_method.png",
    dpi=300
)

plt.show()


# ==================================================
# 6. Silhouette Score Plot
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.title(
    "Silhouette Score for Different K Values"
)

plt.xticks(list(k_values))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/silhouette_scores.png",
    dpi=300
)

plt.show()