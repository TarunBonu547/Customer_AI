import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingModuleSource]

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/wholesale_customers.csv")

print("Original Dataset Shape:", df.shape)


# --------------------------------------------------
# 2. Check for Missing Values
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# 3. Check for Duplicate Rows
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nNumber of Duplicate Rows:", duplicate_count)


# --------------------------------------------------
# 4. Select Features for Clustering
# --------------------------------------------------

features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X = df[features].copy()

print("\nFeatures Selected for Clustering:")
print(X.head())


# --------------------------------------------------
# 5. Check Skewness
# --------------------------------------------------

print("\nSkewness of Features:")
print(X.skew())


# --------------------------------------------------
# 6. Log Transformation
# --------------------------------------------------

X_log = pd.DataFrame(
    np.log1p(X),
    columns=features,
    index=X.index,
)

print("\nData After Log Transformation:")
print(X_log.head())


# --------------------------------------------------
# 7. Standardization
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=features
)

print("\nStandardized Data:")
print(X_scaled.head())


# --------------------------------------------------
# 8. Verify Standardization
# --------------------------------------------------

print("\nMeans After Standardization:")
print(X_scaled.mean().round(3))

print("\nStandard Deviations After Standardization:")
print(X_scaled.std().round(3))


# --------------------------------------------------
# 9. Save Preprocessed Dataset
# --------------------------------------------------

X_scaled.to_csv(
    "data/preprocessed_customers.csv",
    index=False
)

print("\nPreprocessed dataset saved successfully!")

print("\nFinal Dataset Shape:", X_scaled.shape)