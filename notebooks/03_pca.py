import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --------------------------------------------------
# 1. Load Preprocessed Dataset
# --------------------------------------------------

df = pd.read_csv("data/preprocessed_customers.csv")

print("Preprocessed Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# 2. Apply PCA
# --------------------------------------------------

pca = PCA()

X_pca = pca.fit_transform(df)


# --------------------------------------------------
# 3. Explained Variance
# --------------------------------------------------

explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:")
for i, variance in enumerate(explained_variance):
    print(f"PC{i + 1}: {variance:.4f} ({variance * 100:.2f}%)")


# --------------------------------------------------
# 4. Cumulative Explained Variance
# --------------------------------------------------

cumulative_variance = explained_variance.cumsum()

print("\nCumulative Explained Variance:")
for i, variance in enumerate(cumulative_variance):
    print(f"First {i + 1} PC(s): {variance:.4f} ({variance * 100:.2f}%)")


# --------------------------------------------------
# 5. Find Number of Components for 90% Variance
# --------------------------------------------------

n_components_90 = (
    cumulative_variance >= 0.90
).argmax() + 1

print(
    f"\nNumber of components required for "
    f"90% variance: {n_components_90}"
)

# --------------------------------------------------
# PCA with 4 Components for Clustering
# --------------------------------------------------

pca_4 = PCA(n_components=4)

X_pca_4 = pca_4.fit_transform(df)

pca_4_df = pd.DataFrame(
    X_pca_4,
    columns=["PC1", "PC2", "PC3", "PC4"]
)

print("\n4-Component PCA Dataset:")
print(pca_4_df.head())

print("\nShape of 4-Component PCA Dataset:")
print(pca_4_df.shape)

# Save 4-component PCA dataset
pca_4_df.to_csv(
    "data/pca_4_customers.csv",
    index=False
)

print("\n4-component PCA dataset saved successfully!")


# --------------------------------------------------
# 6. Create 2-Component PCA
# --------------------------------------------------

pca_2 = PCA(n_components=2)

X_pca_2 = pca_2.fit_transform(df)

pca_df = pd.DataFrame(
    X_pca_2,
    columns=["PC1", "PC2"]
)

print("\nPCA Dataset:")
print(pca_df.head())


# --------------------------------------------------
# 7. Save PCA Dataset
# --------------------------------------------------

pca_df.to_csv(
    "data/pca_customers.csv",
    index=False
)

print("\nPCA dataset saved successfully!")


# --------------------------------------------------
# 8. Plot Explained Variance
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=0.90,
    linestyle="--",
    label="90% Variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Cumulative Explained Variance")

plt.xticks(range(1, len(cumulative_variance) + 1))

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/pca_explained_variance.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 9. Plot 2D PCA
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customers in PCA Space")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/pca_2d.png",
    dpi=300
)

plt.show()