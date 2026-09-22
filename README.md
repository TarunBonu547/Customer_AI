# 🤖 Customer AI

**Customer AI** is a machine learning application for customer segmentation and spending-pattern analysis. It uses unsupervised learning techniques to identify customer segments and provides interactive insights for individual customers as well as businesses.

## 🚀 Features

### 👤 Customer Analysis
- Analyze a new customer's spending pattern in real time
- Predict the customer's segment using a trained K-Means model
- Compare spending with similar customers
- Generate personalized insights and recommendations

### 💰 Smart Savings Advisor
- Compare spending against the average of the predicted customer segment
- Identify categories with significantly higher or lower spending
- Provide segment-based spending suggestions

### 🔄 What-If Simulator
- Modify spending categories interactively
- Re-run the trained ML pipeline on the simulated values
- Observe how changes in spending affect the predicted customer segment

### 🏢 Business Mode
- Analyze an existing customer dataset
- Upload a custom CSV containing customer spending data
- Classify uploaded customers into segments
- Visualize segment distribution and spending patterns
- Download classified customer data

### 📄 Customer Reports
- Generate downloadable PDF reports
- Include customer segment, spending profile, comparisons, insights, and recommendations

### 🎓 ML Lab
Interactive exploration of:
- Data exploration
- Data preprocessing
- PCA
- K-Means clustering
- BIRCH clustering

### 📊 Model Comparison
Compare clustering approaches using silhouette scores and visualize their performance.

---

## 🧠 Machine Learning Pipeline

The application uses the following preprocessing and clustering workflow:

```text
Customer Spending Data
        ↓
Log Transformation
        ↓
StandardScaler
        ↓
PCA
        ↓
K-Means Clustering
        ↓
Customer Segment
```

The application also includes **BIRCH** as an alternative clustering approach for model comparison.

---

## 📊 Customer Segmentation

The trained K-Means model identifies three customer segments:

| Segment | Description |
|---|---|
| **High-Value Broad Buyer** | Relatively strong spending across several product categories |
| **Grocery & Household Focused** | Spending concentrated around grocery, milk, and household products |
| **General / Lower-Spending** | Comparatively lower overall spending across the analyzed categories |

These segment names are descriptive labels assigned to the clusters based on their spending characteristics.

---

## 📁 Project Structure

```text
Customer_AI/
│
├── data/
│   └── wholesale_customers.csv
│
├── models/
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── kmeans.pkl
│
├── notebooks/
│   └── Machine learning experiments
│
├── outputs/
│   └── Generated results
│
├── src/
│   ├── train_model.py
│   └── test_realtime.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computation
- **Scikit-learn** — machine learning and clustering
- **Joblib** — model persistence
- **Streamlit** — interactive web application
- **Plotly** — interactive visualizations
- **ReportLab** — PDF report generation
- **Jupyter Notebook** — experimentation and analysis

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/TarunBonu547/Customer_AI.git
cd Customer_AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📂 Input Data

The application works with customer spending data containing the following six features:

```text
Fresh
Milk
Grocery
Frozen
Detergents_Paper
Delicassen
```

Business Mode also allows users to upload their own CSV files containing these required columns.

---

## 🔬 Machine Learning Details

### Preprocessing

The customer spending features are processed using:

1. Log transformation
2. StandardScaler
3. Principal Component Analysis (PCA)

### Clustering

The primary customer segmentation model is **K-Means clustering**.

The application also implements **BIRCH clustering** for comparison.

### Real-Time Prediction

A new customer's spending values are passed through the same preprocessing pipeline used during training:

```text
New Customer
     ↓
Log Transformation
     ↓
StandardScaler
     ↓
PCA
     ↓
Trained K-Means Model
     ↓
Customer Segment
```

This allows the application to analyze customers who were not part of the original training dataset.

---

## 📈 Model Evaluation

The project includes model comparison using the **Silhouette Score** to evaluate clustering quality.

The results can be viewed from the **Model Comparison** section of the application.

---

## 🖥️ Application Sections

The Streamlit application contains:

```text
🏠 Home
👤 Analyze Me
💡 My Insights
🏢 Business Mode
🎓 ML Lab
📊 Model Comparison
```

---

## 📊 Sample Business Mode Result

Customer AI can process an external customer CSV and classify customers using the trained ML pipeline.

### Test Dataset

[Download Sample Customer CSV](test_data/customer_ai_test_business.csv)

### Generated Report

[View Sample Customer AI Report](outputs/customer_ai_report.pdf)

---

## 📌 Project Status

**Completed**

The current version includes the core machine learning pipeline, interactive customer analysis, business analysis, model comparison, What-If simulation, Smart Savings Advisor, and downloadable customer reports.

---

## 👨‍💻 Author

**Tarun Bonu**

B-Tech Student | Machine Learning

---

## 📜 License

This project is developed for educational and academic purposes.