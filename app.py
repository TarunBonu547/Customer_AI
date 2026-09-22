import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "wholesale_customers.csv"
)

SCALER_PATH = os.path.join(
    BASE_DIR, "models", "scaler.pkl"
)

PCA_PATH = os.path.join(
    BASE_DIR, "models", "pca.pkl"
)

KMEANS_PATH = os.path.join(
    BASE_DIR, "models", "kmeans.pkl"
)

FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f7f9fc;
}

.hero {
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(135deg, #eef4ff, #ffffff);
    border: 1px solid #e2e8f0;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    color: #64748b;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    background: white;
    border: 1px solid #e2e8f0;
    margin: 15px 0;
}

.segment-name {
    font-size: 30px;
    font-weight: 700;
}

.insight-card {
    padding: 18px;
    border-radius: 14px;
    background: white;
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    kmeans = joblib.load(KMEANS_PATH)

    return scaler, pca, kmeans


df = load_data()

scaler, pca, kmeans = load_models()


# ============================================================
# CREATE CLUSTER PROFILES
# ============================================================

@st.cache_data
def create_cluster_profiles():

    X = df[FEATURES].copy()

    X_log = np.log1p(X)

    X_scaled = scaler.transform(X_log)

    X_pca = pca.transform(X_scaled)

    labels = kmeans.predict(X_pca)

    profile_df = df[FEATURES].copy()

    profile_df["Cluster"] = labels

    averages = profile_df.groupby("Cluster")[FEATURES].mean()

    return averages


cluster_averages = create_cluster_profiles()


# ============================================================
# DETERMINE SEGMENT NAMES
# ============================================================

def create_segment_names(averages):

    names = {}

    total_spending = averages.sum(axis=1)

    high_cluster = total_spending.idxmax()

    names[high_cluster] = "High-Value Broad Buyer"

    remaining = [
        c for c in averages.index
        if c != high_cluster
    ]

    household_score = (
        averages["Grocery"]
        + averages["Milk"]
        + averages["Detergents_Paper"]
    )

    household_cluster = household_score.loc[remaining].idxmax()

    names[household_cluster] = (
        "Grocery & Household Focused"
    )

    for cluster in averages.index:

        if cluster not in names:

            names[cluster] = (
                "General / Lower-Spending"
            )

    return names


SEGMENT_NAMES = create_segment_names(
    cluster_averages
)


# ============================================================
# SEGMENT DESCRIPTIONS
# ============================================================

SEGMENT_DESCRIPTIONS = {

    "High-Value Broad Buyer":
        "Your spending is relatively strong across several product categories.",

    "Grocery & Household Focused":
        "Your spending is concentrated around grocery, milk and household products.",

    "General / Lower-Spending":
        "Your overall spending is comparatively lower across the analyzed categories."
}


# ============================================================
# REAL-TIME PREDICTION
# ============================================================

def predict_customer(values):

    customer = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    # Same preprocessing used during training
    X_log = np.log1p(customer)

    # Standardization
    X_scaled = scaler.transform(X_log)

    # PCA
    X_pca = pca.transform(X_scaled)

    # K-Means prediction
    cluster = int(
        kmeans.predict(X_pca)[0]
    )

    segment = SEGMENT_NAMES.get(
        cluster,
        "Customer Segment"
    )

    return customer, X_pca, cluster, segment


# ============================================================
# GENERATE INSIGHTS
# ============================================================

def generate_insights(customer, cluster):

    values = customer.iloc[0]

    averages = cluster_averages.loc[cluster]

    insights = []

    for feature in FEATURES:

        user_value = values[feature]

        average_value = averages[feature]

        if average_value == 0:
            continue

        difference = (
            (user_value - average_value)
            / average_value
        ) * 100

        if difference >= 20:

            insights.append(
                f"Your **{feature.replace('_', ' ')}** "
                f"spending is about **{difference:.0f}% above** "
                f"the average of your customer segment."
            )

        elif difference <= -20:

            insights.append(
                f"Your **{feature.replace('_', ' ')}** "
                f"spending is about **{abs(difference):.0f}% below** "
                f"the average of your customer segment."
            )

    return insights


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(customer):

    values = customer.iloc[0]

    recommendations = []

    highest_category = (
        values[FEATURES]
        .sort_values(ascending=False)
        .index[0]
    )

    category_name = (
        highest_category
        .replace("_", " ")
    )

    recommendations.append(
        f"Your strongest category is **{category_name}**. "
        f"Look for bundle offers or loyalty rewards in this category."
    )

    if values["Grocery"] > values["Milk"]:

        recommendations.append(
            "You have strong Grocery spending. "
            "Grocery bundles and bulk discounts may provide better value."
        )

    if values["Detergents_Paper"] > 5000:

        recommendations.append(
            "Your household-product spending is significant. "
            "Consider household-product bundles."
        )

    if values["Milk"] > 5000:

        recommendations.append(
            "Your Milk spending is relatively high. "
            "Recurring offers or loyalty rewards could be useful."
        )

    return recommendations


# ============================================================
# SMART SAVINGS ADVISOR
# ============================================================

def generate_savings_advisor(customer, cluster):

    values = customer.iloc[0]
    averages = cluster_averages.loc[cluster]

    category_rows = []

    for feature in FEATURES:

        user_value = float(values[feature])
        average_value = float(averages[feature])

        if average_value <= 0:
            continue

        difference = (
            (user_value - average_value)
            / average_value
        ) * 100

        category_rows.append({
            "feature": feature,
            "user_value": user_value,
            "average_value": average_value,
            "difference": difference
        })

    # Categories with the largest spending above the segment average
    above = [
        row for row in category_rows
        if row["difference"] >= 20
    ]

    above.sort(
        key=lambda row: row["difference"],
        reverse=True
    )

    # Categories with the largest spending below the segment average
    below = [
        row for row in category_rows
        if row["difference"] <= -20
    ]

    below.sort(
        key=lambda row: row["difference"]
    )

    priorities = []

    for row in above[:3]:

        readable = row["feature"].replace("_", " ")

        review_amount = max(
            0,
            row["user_value"] - row["average_value"]
        )

        priorities.append({
            "category": readable,
            "difference": row["difference"],
            "review_amount": review_amount
        })

    # Create practical, segment-based advice.
    if priorities:

        top = priorities[0]

        priority_message = (
            f"Your **{top['category']}** spending is the largest "
            f"area above your segment average. Reviewing roughly "
            f"**{top['review_amount']:,.0f}** of spending in this "
            f"category could bring your profile closer to the "
            f"typical pattern of your segment."
        )

    else:

        priority_message = (
            "None of your categories are substantially above your "
            "segment average. Your spending pattern is already "
            "relatively balanced for your customer segment."
        )

    return priorities, below[:3], priority_message



# ============================================================
# DOWNLOADABLE CUSTOMER REPORT
# ============================================================

def create_customer_pdf(customer, cluster, segment):
    """Create a PDF report from the latest customer analysis."""

    from io import BytesIO

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        spaceBefore=12,
        spaceAfter=8,
    )

    story = [
        Paragraph("Customer AI Report", title_style),
        Paragraph(
            "Real-Time Machine Learning Customer Intelligence",
            subtitle_style,
        ),
    ]

    story.append(Paragraph("Customer Profile", heading_style))
    story.append(Paragraph(f"<b>Segment:</b> {segment}", styles["Normal"]))
    story.append(
        Paragraph(
            SEGMENT_DESCRIPTIONS.get(segment, ""),
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    values = customer.iloc[0]
    total_spending = sum(float(values[f]) for f in FEATURES)
    strongest = max(FEATURES, key=lambda f: values[f])

    summary_data = [
        ["Metric", "Value"],
        ["Total Annual Spending", f"{total_spending:,.0f}"],
        ["Strongest Category", strongest.replace("_", " ")],
        ["Customer Segment", segment],
    ]

    summary_table = Table(summary_data, colWidths=[230, 230])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )
    story.append(summary_table)

    story.append(Paragraph("Spending vs Similar Customers", heading_style))

    comparison_data = [
        ["Category", "Your Spending", "Segment Average", "Difference"]
    ]

    for feature in FEATURES:
        user_value = float(values[feature])
        avg_value = float(cluster_averages.loc[cluster, feature])
        diff = ((user_value - avg_value) / avg_value * 100) if avg_value else 0

        comparison_data.append([
            feature.replace("_", " "),
            f"{user_value:,.0f}",
            f"{avg_value:,.0f}",
            f"{diff:+.1f}%",
        ])

    comparison_table = Table(
        comparison_data,
        colWidths=[145, 105, 105, 105],
        repeatRows=1,
    )
    comparison_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(comparison_table)

    story.append(Paragraph("Why This Segment?", heading_style))
    insights = generate_insights(customer, cluster)

    if insights:
        for insight in insights:
            clean = insight.replace("**", "").replace("🔎 ", "")
            story.append(Paragraph("• " + clean, styles["Normal"]))
            story.append(Spacer(1, 4))
    else:
        story.append(
            Paragraph(
                "Your spending pattern is relatively close to "
                "the average of your segment.",
                styles["Normal"],
            )
        )

    story.append(Paragraph("Smart Savings Advisor", heading_style))

    advisor_items = []
    for feature in FEATURES:
        user_value = float(values[feature])
        avg_value = float(cluster_averages.loc[cluster, feature])

        if avg_value <= 0:
            continue

        difference = ((user_value - avg_value) / avg_value) * 100

        if difference >= 20:
            advisor_items.append(
                f"{feature.replace('_', ' ')} is {difference:.0f}% "
                "above your segment average. Review this category "
                "for bundle offers, subscriptions, or unnecessary purchases."
            )
        elif difference <= -20:
            advisor_items.append(
                f"{feature.replace('_', ' ')} is {abs(difference):.0f}% "
                "below your segment average. This is a relatively "
                "lower-spending category in your profile."
            )

    if not advisor_items:
        advisor_items.append(
            "Your spending is relatively balanced compared with "
            "your customer segment."
        )

    for item in advisor_items:
        story.append(Paragraph("• " + item, styles["Normal"]))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Machine Learning Pipeline", heading_style))
    story.append(
        Paragraph(
            "Log Transformation → StandardScaler → PCA (4 components) "
            "→ K-Means prediction.",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            "The trained PCA pipeline retains approximately 92.07% "
            "of the information.",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 18))
    story.append(
        Paragraph(
            "Note: Savings suggestions are segment-based analytical "
            "insights and are not guaranteed financial savings.",
            styles["Normal"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 Customer AI")

st.sidebar.caption(
    "Real-Time Customer Intelligence"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "👤 Analyze Me",
        "💡 My Insights",
        "🏢 Business Mode",
        "🎓 ML Lab",
        "📊 Model Comparison"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Enter a new customer's spending pattern "
    "and let the trained ML model identify "
    "the most similar customer segment."
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">

    <h1>🤖 Customer AI</h1>

    <p>
    Discover your shopping profile using Machine Learning.
    Enter your spending and receive real-time customer insights.
    </p>

    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Training Customers",
            "440"
        )

    with c2:
        st.metric(
            "Shopping Categories",
            "6"
        )

    with c3:
        st.metric(
            "Customer Segments",
            "3"
        )

    with c4:
        st.metric(
            "PCA Information Retained",
            "92.07%"
        )

    st.divider()

    st.header("✨ What can Customer AI do?")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### 👤 For Customers

        **Analyze your shopping behavior**

        Enter your spending and discover:

        - 🎯 Your customer profile
        - 📊 Your spending pattern
        - 👥 Similar customer behavior
        - 💡 Personalized insights
        - 🛍️ Useful shopping suggestions
        """)

    with col2:

        st.markdown("""
        ### 🧠 Machine Learning

        Customer AI uses:

        **1. Data Preprocessing**

        **2. Standardization**

        **3. PCA**

        **4. K-Means Clustering**

        **5. Segment Analysis**
        """)

    st.divider()

    st.success(
        "👉 Open **Analyze Me** to try the real-time system."
    )


# ============================================================
# ANALYZE ME
# ============================================================

elif page == "👤 Analyze Me":

    st.markdown("""
    <div class="hero">

    <h1>👤 Analyze My Shopping Profile</h1>

    <p>
    Enter your approximate annual spending.
    The trained Machine Learning model will analyze
    your profile instantly.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.info(
        "💡 This can be a completely new customer. "
        "The customer does not need to exist in the training dataset."
    )

    st.header("Enter Your Spending")

    col1, col2, col3 = st.columns(3)

    with col1:

        fresh = st.number_input(
            "🌱 Fresh",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        milk = st.number_input(
            "🥛 Milk",
            min_value=0.0,
            value=8000.0,
            step=500.0
        )

    with col2:

        grocery = st.number_input(
            "🛒 Grocery",
            min_value=0.0,
            value=15000.0,
            step=500.0
        )

        frozen = st.number_input(
            "❄️ Frozen",
            min_value=0.0,
            value=1000.0,
            step=500.0
        )

    with col3:

        detergent = st.number_input(
            "🧴 Detergents & Paper",
            min_value=0.0,
            value=6000.0,
            step=500.0
        )

        delicassen = st.number_input(
            "🍱 Delicatessen",
            min_value=0.0,
            value=1500.0,
            step=500.0
        )

    st.write("")

    analyze = st.button(
        "🔍 ANALYZE MY PROFILE",
        type="primary",
        use_container_width=True
    )

    if analyze:

        values = {
            "Fresh": fresh,
            "Milk": milk,
            "Grocery": grocery,
            "Frozen": frozen,
            "Detergents_Paper": detergent,
            "Delicassen": delicassen
        }

        (
            customer,
            X_pca,
            cluster,
            segment
        ) = predict_customer(values)

        # Store result
        st.session_state["customer"] = customer
        st.session_state["cluster"] = cluster
        st.session_state["segment"] = segment

        st.divider()

        st.header("🎯 Your Customer Profile")

        st.markdown(
            f"""
            <div class="result-card">

            <div class="segment-name">
            {segment}
            </div>

            <p>
            {SEGMENT_DESCRIPTIONS.get(segment, "")}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # METRICS
        # ====================================================

        total_spending = sum(
            values.values()
        )

        strongest = max(
            values,
            key=values.get
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "Total Spending",
                f"{total_spending:,.0f}"
            )

        with m2:
            st.metric(
                "Strongest Category",
                strongest.replace("_", " ")
            )

        with m3:
            st.metric(
                "AI Segment",
                segment
            )

        # ====================================================
        # SPENDING CHART
        # ====================================================

        st.header("📊 Your Spending Profile")

        chart_df = customer.T.reset_index()

        chart_df.columns = [
            "Category",
            "Spending"
        ]

        fig = px.bar(
            chart_df,
            x="Category",
            y="Spending",
            text_auto=".0f",
            title="Your Spending Across Categories"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ====================================================
        # COMPARE WITH SIMILAR CUSTOMERS
        # ====================================================

        st.header("👥 You vs. Similar Customers")

        comparison = pd.DataFrame({
            "Category": FEATURES,
            "Your Spending": [
                customer.iloc[0][f]
                for f in FEATURES
            ],
            "Segment Average": [
                cluster_averages.loc[cluster][f]
                for f in FEATURES
            ]
        })

        comparison["Difference (%)"] = (
            (
                comparison["Your Spending"]
                - comparison["Segment Average"]
            )
            / comparison["Segment Average"]
        ) * 100

        st.dataframe(
            comparison.style.format({
                "Your Spending": "{:,.0f}",
                "Segment Average": "{:,.0f}",
                "Difference (%)": "{:+.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # COMPARISON CHART
        # ====================================================

        compare_long = comparison[
            [
                "Category",
                "Your Spending",
                "Segment Average"
            ]
        ].melt(
            id_vars="Category",
            var_name="Type",
            value_name="Spending"
        )

        fig_compare = px.bar(
            compare_long,
            x="Category",
            y="Spending",
            color="Type",
            barmode="group",
            title="Your Spending vs Segment Average"
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True
        )

        # ====================================================
        # WHY AI CHOSE THIS SEGMENT
        # ====================================================

        st.header("🔍 Why did AI choose this segment?")

        st.write(
            "The model compares your transformed spending pattern "
            "with the patterns learned from existing customers."
        )

        insights = generate_insights(
            customer,
            cluster
        )

        if insights:

            for insight in insights:

                st.markdown(
                    f"""
                    <div class="insight-card">
                    🔎 {insight}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "Your spending pattern is relatively close "
                "to the average of your segment."
            )

        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.header("💡 Your Suggestions")

        recommendations = generate_recommendations(
            customer
        )

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="insight-card">
                💡 {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # SMART SAVINGS ADVISOR
        # ====================================================

        st.header("💰 Smart Savings Advisor")

        st.write(
            "This advisor compares your spending with the average "
            "of customers in your predicted segment. It highlights "
            "categories worth reviewing and suggests practical ways "
            "to improve value."
        )

        st.caption(
            "Note: These are segment-based insights, not guaranteed "
            "savings or financial advice."
        )

        priorities, lower_categories, priority_message = (
            generate_savings_advisor(
                customer,
                cluster
            )
        )

        if priorities:

            st.subheader("🎯 Your Top Review Priorities")

            st.markdown(
                f"""
                <div class="result-card">
                💡 {priority_message}
                </div>
                """,
                unsafe_allow_html=True
            )

            for priority in priorities:

                st.markdown(
                    f"""
                    <div class="insight-card">
                    <b>🛒 {priority["category"]}</b><br>
                    About <b>{priority["difference"]:.0f}% above</b>
                    your segment average.<br>
                    <small>
                    Amount above segment average:
                    {priority["review_amount"]:,.0f}
                    </small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.success(
                "✅ Your spending is relatively balanced compared "
                "with customers in your segment."
            )

        if lower_categories:

            st.subheader("📉 Categories You Spend Less On")

            for row in lower_categories:

                readable = row["feature"].replace(
                    "_",
                    " "
                )

                st.write(
                    f"• **{readable}** is about "
                    f"**{abs(row['difference']):.0f}% below** "
                    f"your segment average."
                )

        st.subheader("🧠 Smart Actions")

        if priorities:

            st.write(
                "• Compare prices, bundles and bulk offers in your "
                "highest-priority category."
            )

            st.write(
                "• Before increasing another category, check whether "
                "the additional spending provides enough value."
            )

            st.write(
                "• Use the What-If Simulator below to explore how "
                "different spending patterns affect your segment."
            )

        else:

            st.write(
                "• Keep tracking your category spending regularly."
            )

            st.write(
                "• Compare major purchases against your usual "
                "segment pattern before increasing spending."
            )

            st.write(
                "• Use the What-If Simulator below to explore "
                "alternative spending patterns."
            )

        # ====================================================
        # ML DETAILS
        # ====================================================

        with st.expander("🔬 View Machine Learning Details"):

            st.write(
                "Your data was processed using the same pipeline "
                "used during model training."
            )

            st.write(
                "1️⃣ Log transformation"
            )

            st.write(
                "2️⃣ StandardScaler"
            )

            st.write(
                "3️⃣ PCA with 4 components"
            )

            st.write(
                "4️⃣ K-Means prediction"
            )

            st.write("PCA representation:")

            st.code(
                np.array2string(
                    X_pca[0],
                    precision=4
                )
            )

    # ============================================================
    # DOWNLOAD REPORT
    # ============================================================

    st.divider()

    st.header("📄 Download Your Customer AI Report")

    st.write(
        "Save your customer segment, spending comparison, "
        "AI explanation, and Smart Savings Advisor as a PDF."
    )

    if "customer" not in st.session_state:
        st.info(
            "First analyze your profile above. "
            "Your downloadable report will appear here."
        )
    else:
        report_pdf = create_customer_pdf(
            st.session_state["customer"],
            st.session_state["cluster"],
            st.session_state["segment"],
        )

        st.download_button(
            label="📥 Download My Customer AI Report",
            data=report_pdf,
            file_name="customer_ai_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

    # ============================================================
    # WHAT-IF SIMULATOR
    # ============================================================

    st.markdown("---")

    st.header("🔄 What-If Simulator")

    st.write(
        "Explore how changing your spending pattern could change "
        "your customer profile."
    )
    
    if "customer" not in st.session_state:
    
        st.info(
            "First analyze your profile above. "
            "Your current spending will then appear here."
        )
    
    else:
    
        current_customer = st.session_state["customer"]
        current_cluster = st.session_state["cluster"]
        current_segment = st.session_state["segment"]
    
        st.subheader("Adjust Your Spending")
    
        st.caption(
            "Move the sliders and see how the trained ML model "
            "responds to the new spending pattern."
        )
    
        col1, col2 = st.columns(2)
    
        with col1:
    
            whatif_fresh = st.slider(
                "🌱 Fresh",
                min_value=0,
                max_value=50000,
                value=int(current_customer.iloc[0]["Fresh"]),
                step=500
            )
    
            whatif_milk = st.slider(
                "🥛 Milk",
                min_value=0,
                max_value=50000,
                value=int(current_customer.iloc[0]["Milk"]),
                step=500
            )
    
            whatif_grocery = st.slider(
                "🛒 Grocery",
                min_value=0,
                max_value=100000,
                value=int(current_customer.iloc[0]["Grocery"]),
                step=500
            )
    
        with col2:
    
            whatif_frozen = st.slider(
                "❄️ Frozen",
                min_value=0,
                max_value=50000,
                value=int(current_customer.iloc[0]["Frozen"]),
                step=500
            )
    
            whatif_detergent = st.slider(
                "🧴 Detergents & Paper",
                min_value=0,
                max_value=50000,
                value=int(
                    current_customer.iloc[0]["Detergents_Paper"]
                ),
                step=500
            )
    
            whatif_delicassen = st.slider(
                "🍱 Delicatessen",
                min_value=0,
                max_value=50000,
                value=int(
                    current_customer.iloc[0]["Delicassen"]
                ),
                step=500
            )
    
        whatif_values = {
            "Fresh": whatif_fresh,
            "Milk": whatif_milk,
            "Grocery": whatif_grocery,
            "Frozen": whatif_frozen,
            "Detergents_Paper": whatif_detergent,
            "Delicassen": whatif_delicassen
        }
    
        (
            whatif_customer,
            whatif_pca,
            whatif_cluster,
            whatif_segment
        ) = predict_customer(
            whatif_values
        )
    
        st.divider()
    
        st.subheader("🎯 Simulation Result")
    
        current_col, arrow_col, whatif_col = st.columns([4, 1, 4])
    
        with current_col:
    
            st.markdown("### Current Profile")
    
            st.success(
                f"**{current_segment}**"
            )
    
        with arrow_col:
    
            st.markdown(
                "<h2 style='text-align:center;'>→</h2>",
                unsafe_allow_html=True
            )
    
        with whatif_col:
    
            st.markdown("### What-If Profile")
    
            if whatif_segment == current_segment:
    
                st.success(
                    f"**{whatif_segment}**"
                )
    
            else:
    
                st.warning(
                    f"**{whatif_segment}**"
                )
    
        # --------------------------------------------------------
        # Change detection
        # --------------------------------------------------------
    
        if whatif_segment == current_segment:
    
            st.info(
                "Your simulated spending pattern remains in the "
                "same customer segment."
            )
    
        else:
    
            st.success(
                f"Your simulated spending pattern moved from "
                f"**{current_segment}** to **{whatif_segment}**."
            )
    
        # --------------------------------------------------------
        # Compare spending
        # --------------------------------------------------------
    
        st.subheader("📊 Current vs What-If Spending")
    
        comparison_df = pd.DataFrame({
            "Category": FEATURES,
            "Current": [
                current_customer.iloc[0][feature]
                for feature in FEATURES
            ],
            "What-If": [
                whatif_customer.iloc[0][feature]
                for feature in FEATURES
            ]
        })
    
        comparison_long = comparison_df.melt(
            id_vars="Category",
            var_name="Profile",
            value_name="Spending"
        )
    
        fig_whatif = px.bar(
            comparison_long,
            x="Category",
            y="Spending",
            color="Profile",
            barmode="group",
            title="Current vs What-If Spending"
        )
    
        st.plotly_chart(
            fig_whatif,
            use_container_width=True
        )
    
        # --------------------------------------------------------
        # Explanation
        # --------------------------------------------------------
    
        st.subheader("🔍 What Changed?")
    
        changed_categories = []
    
        for feature in FEATURES:
    
            old_value = current_customer.iloc[0][feature]
    
            new_value = whatif_customer.iloc[0][feature]
    
            if old_value != new_value:
    
                difference = new_value - old_value
    
                changed_categories.append(
                    (
                        feature,
                        difference
                    )
                )
    
        if changed_categories:
    
            for feature, difference in changed_categories:
    
                readable_name = feature.replace(
                    "_",
                    " & "
                )
    
                if difference > 0:
    
                    st.write(
                        f"📈 **{readable_name}** increased by "
                        f"**{difference:,.0f}**."
                    )
    
                else:
    
                    st.write(
                        f"📉 **{readable_name}** decreased by "
                        f"**{abs(difference):,.0f}**."
                    )
    
        else:
    
            st.write(
                "No spending values were changed."
            )
    
        # --------------------------------------------------------
        # ML explanation
        # --------------------------------------------------------
    
        with st.expander(
            "🔬 View What-If ML Processing"
        ):
    
            st.write(
                "The What-If values are passed through exactly "
                "the same preprocessing and trained model used "
                "for the original prediction."
            )
    
            st.write(
                "Log Transformation → StandardScaler → PCA → K-Means"
            )
    
            st.write(
                "What-If PCA representation:"
            )
    
            st.code(
                np.array2string(
                    whatif_pca[0],
                    precision=4
                )
            )
    

# ============================================================
# MY INSIGHTS
# ============================================================

elif page == "💡 My Insights":

    st.markdown("""
    <div class="hero">

    <h1>💡 My Insights</h1>

    <p>
    Review the results from your latest analysis.
    </p>

    </div>
    """, unsafe_allow_html=True)

    if "customer" not in st.session_state:

        st.warning(
            "Please analyze your profile first."
        )

    else:

        customer = st.session_state["customer"]

        cluster = st.session_state["cluster"]

        segment = st.session_state["segment"]

        st.success(
            f"Your profile: **{segment}**"
        )

        st.header("📊 Your Spending")

        st.dataframe(
            customer,
            use_container_width=True,
            hide_index=True
        )

        st.header("💡 Recommendations")

        recommendations = generate_recommendations(
            customer
        )

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="insight-card">
                💡 {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.header("💰 Smart Savings Advisor")

        priorities, lower_categories, priority_message = (
            generate_savings_advisor(
                customer,
                cluster
            )
        )

        if priorities:

            st.info(priority_message)

            for priority in priorities:

                st.markdown(
                    f"""
                    <div class="insight-card">
                    🎯 <b>{priority["category"]}</b>:
                    {priority["difference"]:.0f}% above your
                    segment average.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.success(
                "Your spending is relatively balanced for your "
                "customer segment."
            )



# ============================================================
# BUSINESS MODE
# ============================================================

elif page == "🏢 Business Mode":

    st.markdown("""
    <div class="hero">

    <h1>🏢 Business Customer Intelligence</h1>

    <p>
    Understand customer segments, spending behavior and
    opportunities using the trained Machine Learning model.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Business Mode analyzes the existing customer dataset. "
        "The segment labels are created from the trained K-Means "
        "model and are intended for customer-behavior analysis."
    )

    # --------------------------------------------------------
    # CSV UPLOAD
    # --------------------------------------------------------

    st.header("📁 Analyze Your Own Customer CSV")

    st.write(
        "Upload a CSV containing the six spending categories below. "
        "The same trained preprocessing, PCA and K-Means pipeline "
        "will classify every uploaded customer."
    )

    st.caption(
        "Required columns: Fresh, Milk, Grocery, Frozen, "
        "Detergents_Paper, Delicassen"
    )

    uploaded_file = st.file_uploader(
        "Choose a customer CSV file",
        type=["csv"],
        help="The CSV must contain the six required spending columns."
    )

    uploaded_df = None

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            missing_columns = [
                feature
                for feature in FEATURES
                if feature not in uploaded_df.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded CSV is missing these required columns: "
                    + ", ".join(missing_columns)
                )

                uploaded_df = None

            else:

                numeric_data = uploaded_df[FEATURES].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                invalid_count = int(
                    numeric_data.isna().sum().sum()
                )

                if invalid_count > 0:

                    st.error(
                        f"The uploaded file contains {invalid_count} "
                        "missing or non-numeric spending values. "
                        "Please clean the CSV and upload it again."
                    )

                    uploaded_df = None

                elif (numeric_data < 0).any().any():

                    st.error(
                        "Spending values cannot be negative. "
                        "Please check the uploaded CSV."
                    )

                    uploaded_df = None

                else:

                    uploaded_df = uploaded_df.copy()
                    uploaded_df[FEATURES] = numeric_data

                    st.success(
                        f"CSV loaded successfully: "
                        f"{len(uploaded_df):,} customers."
                    )

        except Exception as exc:

            st.error(
                f"Could not read the CSV file: {exc}"
            )

            uploaded_df = None

    if uploaded_df is not None:

        # Apply the exact trained pipeline.
        upload_log = np.log1p(
            uploaded_df[FEATURES]
        )

        upload_scaled = scaler.transform(
            upload_log
        )

        upload_pca = pca.transform(
            upload_scaled
        )

        upload_labels = kmeans.predict(
            upload_pca
        )

        analyzed_upload = uploaded_df.copy()

        analyzed_upload["Cluster"] = upload_labels

        analyzed_upload["Segment"] = [
            SEGMENT_NAMES[int(label)]
            for label in upload_labels
        ]

        st.subheader("🎯 Uploaded Customer Segments")

        upload_counts = (
            analyzed_upload["Segment"]
            .value_counts()
            .reset_index()
        )

        upload_counts.columns = [
            "Segment",
            "Customers"
        ]

        upload_counts["Share (%)"] = (
            upload_counts["Customers"]
            / len(analyzed_upload)
            * 100
        )

        c1, c2, c3 = st.columns(3)

        for column, segment_name in zip(
            [c1, c2, c3],
            SEGMENT_NAMES.values()
        ):

            count = int(
                (
                    analyzed_upload["Segment"]
                    == segment_name
                ).sum()
            )

            with column:
                st.metric(
                    segment_name,
                    f"{count:,}"
                )

        st.dataframe(
            upload_counts.style.format({
                "Customers": "{:,.0f}",
                "Share (%)": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        fig_upload = px.pie(
            upload_counts,
            names="Segment",
            values="Customers",
            title="Uploaded Customer Segment Distribution"
        )

        st.plotly_chart(
            fig_upload,
            use_container_width=True
        )

        st.subheader("💰 Uploaded Customer Spending Profile")

        upload_profile = (
            analyzed_upload
            .groupby("Segment")[FEATURES]
            .mean()
            .reset_index()
        )

        upload_long = upload_profile.melt(
            id_vars="Segment",
            var_name="Category",
            value_name="Average Spending"
        )

        fig_upload_profile = px.bar(
            upload_long,
            x="Category",
            y="Average Spending",
            color="Segment",
            barmode="group",
            title="Average Spending by Uploaded Customer Segment"
        )

        st.plotly_chart(
            fig_upload_profile,
            use_container_width=True
        )

        with st.expander("📋 View Classified Uploaded Customers"):

            st.dataframe(
                analyzed_upload,
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            label="📥 Download Classified CSV",
            data=analyzed_upload.to_csv(index=False).encode("utf-8"),
            file_name="customer_ai_classified_customers.csv",
            mime="text/csv",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREPARE BUSINESS DATA
    # --------------------------------------------------------

    business_df = df[FEATURES].copy()

    X_business_log = np.log1p(business_df)
    X_business_scaled = scaler.transform(X_business_log)
    X_business_pca = pca.transform(X_business_scaled)

    business_labels = kmeans.predict(X_business_pca)

    business_df["Cluster"] = business_labels
    business_df["Segment"] = [
        SEGMENT_NAMES[int(label)]
        for label in business_labels
    ]

    # --------------------------------------------------------
    # OVERVIEW METRICS
    # --------------------------------------------------------

    st.header("📊 Customer Base Overview")

    segment_counts = (
        business_df["Segment"]
        .value_counts()
    )

    total_customers = len(business_df)

    high_value_count = int(
        (business_df["Segment"] == "High-Value Broad Buyer").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with c2:
        st.metric(
            "Customer Segments",
            "3"
        )

    with c3:
        st.metric(
            "High-Value Customers",
            f"{high_value_count:,}"
        )

    with c4:
        st.metric(
            "Categories",
            "6"
        )

    # --------------------------------------------------------
    # SEGMENT DISTRIBUTION
    # --------------------------------------------------------

    st.header("🎯 Customer Segment Distribution")

    distribution_df = pd.DataFrame({
        "Segment": segment_counts.index,
        "Customers": segment_counts.values
    })

    distribution_df["Share (%)"] = (
        distribution_df["Customers"]
        / total_customers
        * 100
    )

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            distribution_df.style.format({
                "Customers": "{:,.0f}",
                "Share (%)": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    with col2:

        fig_distribution = px.pie(
            distribution_df,
            names="Segment",
            values="Customers",
            title="Customer Segment Share"
        )

        st.plotly_chart(
            fig_distribution,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SEGMENT SPENDING PROFILE
    # --------------------------------------------------------

    st.header("💰 Segment Spending Profile")

    segment_profile = (
        business_df
        .groupby("Segment")[FEATURES]
        .mean()
        .reset_index()
    )

    display_profile = segment_profile.copy()

    for feature in FEATURES:
        display_profile[feature] = display_profile[feature].round(0)

    st.dataframe(
        display_profile.style.format({
            feature: "{:,.0f}"
            for feature in FEATURES
        }),
        use_container_width=True,
        hide_index=True
    )

    profile_long = segment_profile.melt(
        id_vars="Segment",
        var_name="Category",
        value_name="Average Spending"
    )

    fig_profile = px.bar(
        profile_long,
        x="Category",
        y="Average Spending",
        color="Segment",
        barmode="group",
        title="Average Spending by Customer Segment"
    )

    st.plotly_chart(
        fig_profile,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TOTAL SPENDING BY SEGMENT
    # --------------------------------------------------------

    st.header("📈 Total Spending by Segment")

    segment_totals = (
        segment_profile
        .set_index("Segment")[FEATURES]
        .sum(axis=1)
        .sort_values(ascending=False)
        .reset_index()
    )

    segment_totals.columns = [
        "Segment",
        "Average Total Spending"
    ]

    fig_totals = px.bar(
        segment_totals,
        x="Segment",
        y="Average Total Spending",
        text_auto=".0f",
        title="Average Total Spending per Customer"
    )

    st.plotly_chart(
        fig_totals,
        use_container_width=True
    )

    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.header("💡 Business Insights")

    total_by_segment = (
        segment_profile
        .set_index("Segment")[FEATURES]
        .sum(axis=1)
    )

    highest_value_segment = (
        total_by_segment.idxmax()
    )

    lowest_value_segment = (
        total_by_segment.idxmin()
    )

    highest_category = (
        segment_profile
        .set_index("Segment")[FEATURES]
        .mean()
        .idxmax()
    )

    insight_items = [
        (
            "🏆",
            f"**{highest_value_segment}** has the highest "
            "average total spending among the identified segments."
        ),
        (
            "📉",
            f"**{lowest_value_segment}** has the lowest "
            "average total spending among the identified segments."
        ),
        (
            "🛒",
            f"Across the customer base, **{highest_category.replace('_', ' ')}** "
            "has the highest average spending among the six analyzed categories."
        ),
        (
            "🎯",
            "Businesses can use these segments to understand "
            "different spending patterns and design segment-specific "
            "marketing or loyalty strategies."
        ),
    ]

    for icon, text in insight_items:

        st.markdown(
            f"""
            <div class="insight-card">
            {icon} {text}
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # SEGMENT FOCUS
    # --------------------------------------------------------

    st.header("🔎 Explore a Segment")

    selected_segment = st.selectbox(
        "Select a customer segment",
        list(SEGMENT_NAMES.values())
    )

    selected_rows = business_df[
        business_df["Segment"] == selected_segment
    ]

    selected_average = (
        selected_rows[FEATURES]
        .mean()
    )

    st.write(
        SEGMENT_DESCRIPTIONS.get(
            selected_segment,
            ""
        )
    )

    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric(
            "Customers in Segment",
            f"{len(selected_rows):,}"
        )

    with a2:
        st.metric(
            "Segment Share",
            f"{len(selected_rows) / total_customers * 100:.1f}%"
        )

    with a3:
        st.metric(
            "Average Total Spending",
            f"{selected_average.sum():,.0f}"
        )

    selected_chart = pd.DataFrame({
        "Category": FEATURES,
        "Average Spending": [
            selected_average[feature]
            for feature in FEATURES
        ]
    })

    fig_selected = px.bar(
        selected_chart,
        x="Category",
        y="Average Spending",
        text_auto=".0f",
        title=f"{selected_segment} — Average Spending"
    )

    st.plotly_chart(
        fig_selected,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CUSTOMER DATA
    # --------------------------------------------------------

    with st.expander("📋 View Classified Customer Data"):

        display_customers = business_df.copy()

        st.dataframe(
            display_customers,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# ML LAB
# ============================================================

elif page == "🎓 ML Lab":

    st.markdown("""
    <div class="hero">

    <h1>🎓 Machine Learning Lab</h1>

    <p>
    Explore the technical components behind Customer AI.
    </p>

    </div>
    """, unsafe_allow_html=True)

    section = st.selectbox(
        "Choose Component",
        [
            "Data Exploration",
            "Preprocessing",
            "PCA Analysis",
            "K-Means",
            "BIRCH"
        ]
    )

    # --------------------------------------------------------
    # DATA EXPLORATION
    # --------------------------------------------------------

    if section == "Data Exploration":

        st.header("🔎 Data Exploration")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Rows", df.shape[0])

        with c2:
            st.metric("Columns", df.shape[1])

        with c3:
            st.metric(
                "Duplicates",
                int(df.duplicated().sum())
            )

        with c4:
            st.metric(
                "Missing Values",
                int(df.isna().sum().sum())
            )

        st.subheader("Dataset")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Statistics")

        st.dataframe(
            df.describe(),
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    elif section == "Preprocessing":

        st.header("🧹 Data Preprocessing")

        X = df[FEATURES].copy()

        st.subheader("Feature Skewness")

        skewness = X.skew()

        skew_df = pd.DataFrame({
            "Feature": skewness.index,
            "Skewness": skewness.values
        })

        st.dataframe(
            skew_df,
            use_container_width=True,
            hide_index=True
        )

        st.write(
            "Log transformation is applied to reduce strong "
            "right-skewness. StandardScaler then places features "
            "on a comparable scale."
        )

        X_log = np.log1p(X)

        X_scaled = scaler.transform(X_log)

        scaled_df = pd.DataFrame(
            X_scaled,
            columns=FEATURES
        )

        st.subheader("Standardized Data")

        st.dataframe(
            scaled_df.head(),
            use_container_width=True
        )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    elif section == "PCA Analysis":

        st.header("📉 PCA Analysis")

        X = df[FEATURES]

        X_log = np.log1p(X)

        X_scaled = scaler.transform(X_log)

        X_pca = pca.transform(X_scaled)

        explained = pca.explained_variance_ratio_

        pca_table = pd.DataFrame({
            "Component": [
                f"PC{i + 1}"
                for i in range(len(explained))
            ],
            "Variance": explained,
            "Cumulative Variance":
                np.cumsum(explained)
        })

        st.dataframe(
            pca_table,
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Variance Retained",
            f"{explained.sum() * 100:.2f}%"
        )

        plot_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1]
        })

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            title="Customers in PCA Space"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    elif section == "K-Means":

        st.header("🔵 K-Means")

        X = df[FEATURES]

        X_log = np.log1p(X)

        X_scaled = scaler.transform(X_log)

        X_pca = pca.transform(X_scaled)

        labels = kmeans.predict(X_pca)

        plot_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Segment": [
                SEGMENT_NAMES[int(x)]
                for x in labels
            ]
        })

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Segment",
            title="K-Means Customer Segments"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Cluster Sizes")

        counts = pd.Series(
            labels
        ).value_counts().sort_index()

        for cluster, count in counts.items():

            st.write(
                f"**{SEGMENT_NAMES[cluster]}:** {count} customers"
            )

    # --------------------------------------------------------
    # BIRCH
    # --------------------------------------------------------

    elif section == "BIRCH":

        st.header("🟢 BIRCH")

        st.write(
            "BIRCH is used as a second clustering approach "
            "for comparison with K-Means."
        )

        from sklearn.cluster import Birch
        from sklearn.metrics import silhouette_score

        X = df[FEATURES]

        X_log = np.log1p(X)

        X_scaled = scaler.transform(X_log)

        X_pca = pca.transform(X_scaled)

        birch = Birch(
            threshold=1.0,
            branching_factor=50,
            n_clusters=3
        )

        labels = birch.fit_predict(X_pca)

        score = silhouette_score(
            X_pca,
            labels
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Clusters",
                3
            )

        with c2:
            st.metric(
                "Silhouette Score",
                f"{score:.4f}"
            )

        plot_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": labels
        })

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title="BIRCH Clusters"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":

    st.markdown("""
    <div class="hero">

    <h1>📊 Model Comparison</h1>

    <p>
    Compare the clustering methods used in this project.
    </p>

    </div>
    """, unsafe_allow_html=True)

    from sklearn.cluster import Birch
    from sklearn.metrics import silhouette_score

    X = df[FEATURES]

    X_log = np.log1p(X)

    X_scaled = scaler.transform(X_log)

    X_pca = pca.transform(X_scaled)

    # K-Means
    km_labels = kmeans.predict(X_pca)

    km_score = silhouette_score(
        X_pca,
        km_labels
    )

    # BIRCH
    birch = Birch(
        threshold=1.0,
        branching_factor=50,
        n_clusters=3
    )

    birch_labels = birch.fit_predict(
        X_pca
    )

    birch_score = silhouette_score(
        X_pca,
        birch_labels
    )

    comparison = pd.DataFrame({
        "Model": [
            "K-Means Scratch",
            "K-Means sklearn",
            "BIRCH Scratch",
            "BIRCH sklearn"
        ],
        "Silhouette Score": [
            0.2857,
            km_score,
            0.3122,
            birch_score
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        comparison,
        x="Model",
        y="Silhouette Score",
        text_auto=".3f",
        title="Clustering Model Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "K-Means is used for real-time customer prediction "
        "because the trained model directly supports prediction "
        "for new customer inputs."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer AI • Real-Time Machine Learning Customer Segmentation"
)
