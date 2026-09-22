import joblib
import numpy as np
import pandas as pd


# -----------------------------
# Load saved models
# -----------------------------
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
kmeans = joblib.load("models/kmeans.pkl")


# -----------------------------
# New customer input
# -----------------------------
new_customer = pd.DataFrame([{
    "Fresh": 5000,
    "Milk": 8000,
    "Grocery": 15000,
    "Frozen": 1000,
    "Detergents_Paper": 6000,
    "Delicassen": 1500
}])


# -----------------------------
# Apply same preprocessing
# -----------------------------
X_log = np.log1p(new_customer)

X_scaled = scaler.transform(X_log)

X_pca = pca.transform(X_scaled)


# -----------------------------
# Predict customer segment
# -----------------------------
cluster = kmeans.predict(X_pca)[0]


# -----------------------------
# Display result
# -----------------------------
segment_names = {
    0: "Grocery & Household Focused",
    1: "General / Lower-Spending",
    2: "High-Value Broad Buyer"
}

print("=" * 50)
print("REAL-TIME CUSTOMER ANALYSIS")
print("=" * 50)

print("\nCustomer Spending:")
print(new_customer.to_string(index=False))

print(f"\nPredicted Cluster : {cluster}")
print(f"Customer Profile  : {segment_names.get(cluster, 'Customer Segment')}")

print("\nPCA Representation:")
print(X_pca[0])

print("\nAnalysis completed successfully!")