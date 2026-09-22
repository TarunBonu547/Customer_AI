import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer AI",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Customer AI")
st.subheader("Customer Segmentation & Intelligence System")

st.write(
    "An end-to-end Machine Learning prototype using "
    "Preprocessing, PCA, K-Means and BIRCH clustering."
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    original = pd.read_csv(
        "data/wholesale_customers.csv"
    )

    preprocessed = pd.read_csv(
        "data/preprocessed_customers.csv"
    )

    pca = pd.read_csv(
        "data/pca_4_customers.csv"
    )

    return original, preprocessed, pca


original, preprocessed, pca_df = load_data()


# =========================================================
# DATA PREPARATION
# =========================================================

X = pca_df.values


# =========================================================
# K-MEANS
# =========================================================

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(X)

kmeans_silhouette = silhouette_score(
    X,
    kmeans_labels
)


# =========================================================
# BIRCH
# =========================================================

birch = Birch(
    threshold=1.0,
    branching_factor=50,
    n_clusters=3
)

birch_labels = birch.fit_predict(X)

birch_silhouette = silhouette_score(
    X,
    birch_labels
)


# =========================================================
# KPI SECTION
# =========================================================

st.markdown("## 📊 System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Customers",
        len(original)
    )

with col2:

    st.metric(
        "PCA Components",
        pca_df.shape[1]
    )

with col3:

    st.metric(
        "K-Means Silhouette",
        f"{kmeans_silhouette:.3f}"
    )

with col4:

    st.metric(
        "BIRCH Silhouette",
        f"{birch_silhouette:.3f}"
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Data Exploration",
        "PCA Analysis",
        "K-Means",
        "BIRCH",
        "Customer Profiles"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header("📈 Customer Intelligence Dashboard")

    st.write(
        "This system analyzes wholesale customer purchasing "
        "behavior and identifies groups of customers with "
        "similar spending patterns."
    )

    st.markdown("### ML Pipeline")

    st.code(
        """
Customer Dataset
       ↓
Data Preprocessing
       ↓
Log Transformation
       ↓
Standardization
       ↓
PCA
       ↓
K-Means / BIRCH
       ↓
Customer Segmentation
       ↓
Business Insights
        """
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        original.head(10),
        use_container_width=True
    )


# =========================================================
# DATA EXPLORATION
# =========================================================

elif page == "Data Exploration":

    st.header("🔎 Data Exploration")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Rows",
            original.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            original.shape[1]
        )

    st.markdown("### Dataset")

    st.dataframe(
        original,
        use_container_width=True
    )

    st.markdown("### Missing Values")

    missing = original.isnull().sum()

    st.dataframe(
        missing.to_frame(
            "Missing Values"
        )
    )

    st.markdown("### Statistical Summary")

    st.dataframe(
        original.describe(),
        use_container_width=True
    )


# =========================================================
# PCA ANALYSIS
# =========================================================

elif page == "PCA Analysis":

    st.header("📉 Principal Component Analysis")

    st.write(
        "PCA reduces the dimensionality of the customer "
        "dataset while preserving most of its information."
    )

    st.markdown("### PCA Dataset")

    st.dataframe(
        pca_df.head(10),
        use_container_width=True
    )

    st.markdown("### PCA Visualization")

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        alpha=0.7
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        "Customers in PCA Space"
    )

    ax.grid(True)

    st.pyplot(fig)

    st.info(
        "The PCA representation is used as the input "
        "for the clustering algorithms."
    )


# =========================================================
# K-MEANS
# =========================================================

elif page == "K-Means":

    st.header("🔵 K-Means Customer Segmentation")

    st.write(
        "K-Means divides customers into three groups "
        "based on similarity in PCA space."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Number of Clusters",
            3
        )

    with col2:

        st.metric(
            "Inertia",
            f"{kmeans.inertia_:.2f}"
        )

    with col3:

        st.metric(
            "Silhouette Score",
            f"{kmeans_silhouette:.3f}"
        )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for cluster in range(3):

        points = X[
            kmeans_labels == cluster
        ]

        ax.scatter(
            points[:, 0],
            points[:, 1],
            label=f"Cluster {cluster}"
        )

    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=200,
        label="Centroids"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        "K-Means Customer Segmentation"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    st.markdown("### Cluster Distribution")

    cluster_counts = (
        pd.Series(kmeans_labels)
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts
    )


# =========================================================
# BIRCH
# =========================================================

elif page == "BIRCH":

    st.header("🟢 BIRCH Customer Segmentation")

    st.write(
        "BIRCH creates compact clustering summaries "
        "using a hierarchical clustering structure."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Number of Clusters",
            3
        )

    with col2:

        st.metric(
            "Threshold",
            1.0
        )

    with col3:

        st.metric(
            "Silhouette Score",
            f"{birch_silhouette:.3f}"
        )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for cluster in sorted(
        set(birch_labels)
    ):

        points = X[
            birch_labels == cluster
        ]

        ax.scatter(
            points[:, 0],
            points[:, 1],
            label=f"Cluster {cluster}"
        )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        "BIRCH Customer Segmentation"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    st.markdown("### Cluster Distribution")

    cluster_counts = (
        pd.Series(birch_labels)
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        cluster_counts
    )


# =========================================================
# CUSTOMER PROFILES
# =========================================================

elif page == "Customer Profiles":

    st.header("👥 Customer Profiles")

    features = [
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents_Paper",
        "Delicassen"
    ]

    profile_df = original[
        features
    ].copy()

    profile_df["Cluster"] = (
        kmeans_labels
    )

    st.markdown(
        "### Average Spending by Customer Segment"
    )

    averages = (
        profile_df
        .groupby("Cluster")[features]
        .mean()
        .round(2)
    )

    st.dataframe(
        averages,
        use_container_width=True
    )

    st.markdown(
        "### Select a Customer Segment"
    )

    selected_cluster = st.selectbox(
        "Cluster",
        [0, 1, 2]
    )

    selected_profile = averages.loc[
        selected_cluster
    ]

    st.markdown(
        f"### Cluster {selected_cluster}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Fresh",
            f"{selected_profile['Fresh']:,.0f}"
        )

        st.metric(
            "Grocery",
            f"{selected_profile['Grocery']:,.0f}"
        )

    with c2:

        st.metric(
            "Milk",
            f"{selected_profile['Milk']:,.0f}"
        )

        st.metric(
            "Frozen",
            f"{selected_profile['Frozen']:,.0f}"
        )

    with c3:

        st.metric(
            "Detergents/Paper",
            f"{selected_profile['Detergents_Paper']:,.0f}"
        )

        st.metric(
            "Delicassen",
            f"{selected_profile['Delicassen']:,.0f}"
        )


    # ==============================================
    # BUSINESS INTERPRETATION
    # ==============================================

    st.markdown(
        "### 💡 Business Interpretation"
    )

    total_spending = float(
        selected_profile[features].to_numpy(dtype=float).sum()
    )

    if total_spending >= 45000:

        st.success(
            "High-value customer segment. "
            "Recommended strategies include loyalty "
            "programs, premium offers and cross-selling."
        )

    elif (
        float(selected_profile["Grocery"])
        > float(selected_profile["Fresh"])
        and
        float(selected_profile["Milk"])
        > float(selected_profile["Fresh"])
    ):

        st.info(
            "Grocery-focused customer segment. "
            "Consider grocery bundles and household "
            "product promotions."
        )

    else:

        st.warning(
            "General customer segment. "
            "Targeted discounts and personalized "
            "offers may help increase purchasing."
        )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "Customer AI | ML GLOB Project\n\n"
    "Preprocessing + PCA + K-Means + BIRCH"
)