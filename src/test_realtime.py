"""Run a manual prediction against the checked-in Customer AI artifacts."""

from pathlib import Path

import joblib
import pandas as pd

try:
    from src.customer_ai import predict_customers
except ModuleNotFoundError:
    from customer_ai import predict_customers

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


def main() -> None:
    """Load the saved pipeline and print one example prediction."""
    new_customer = pd.DataFrame(
        [
            {
                "Fresh": 5000,
                "Milk": 8000,
                "Grocery": 15000,
                "Frozen": 1000,
                "Detergents_Paper": 6000,
                "Delicassen": 1500,
            }
        ]
    )
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    pca = joblib.load(MODEL_DIR / "pca.pkl")
    kmeans = joblib.load(MODEL_DIR / "kmeans.pkl")
    cluster = int(predict_customers(new_customer, scaler, pca, kmeans)[0])

    print("=" * 50)
    print("REAL-TIME CUSTOMER ANALYSIS")
    print("=" * 50)
    print("\nCustomer Spending:")
    print(new_customer.to_string(index=False))
    print(f"\nPredicted Cluster : {cluster}")
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
