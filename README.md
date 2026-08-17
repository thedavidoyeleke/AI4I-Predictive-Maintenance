````markdown
# 🛠️ AI4I 2020 Predictive Maintenance Dashboard

An end-to-end, enterprise-grade Machine Learning solution and interactive web application designed to predict equipment failures, analyze root causes, estimate Remaining Useful Life (RUL), and quantify the financial impact of maintenance decisions in industrial manufacturing environments.

Built using **Python**, **XGBoost**, **Imbalanced-learn (SMOTE)**, **SHAP (Explainable AI)**, and **Streamlit**.

---

## 🌟 Key Features & Industry Use Cases

- **🔮 Failure Risk Prediction:** Real-time machine status classification (`Normal` vs. `Failure Risk`) powered by a hyperparameter-tuned **XGBoost Classifier**.
- **🧠 Explainable AI (SHAP):** Transparent, feature-level model explanations via SHAP waterfall charts and plain-English breakdowns to help plant operators understand _why_ predictions were made.
- **⏳ Remaining Useful Life (RUL) Estimation:** Computes the remaining tool lifespan in minutes based on physical degradation thresholds (`Tool wear [min]`) and operating strain.
- **🛠️ Automated Root Cause Diagnosis:** Identifies specific physical failure modes (e.g., Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure).
- **💰 Financial Impact & ROI Quantification:** Calculates real-time cost trade-offs between proactive scheduled maintenance versus reactive emergency downtime losses.
- **🎨 Interactive Streamlit Interface:** Modern, clean UI featuring custom status-coded visualizations, health probability charts, and interactive input parameter tuning.

---

## 🏗️ Technical Architecture & Pipeline

1. **Preprocessing & Feature Engineering:**
   - Numerical Scaling: `StandardScaler` applied to continuous sensor measurements (`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`).
   - Categorical Encoding: `OneHotEncoder` applied to equipment quality variants (`Type`: L, M, H).
   - Class Imbalance Handling: Integrated `SMOTE` within an `ImbPipeline` to handle sparse failure cases safely during cross-validation.
2. **Model Optimization:**
   - Hyperparameter tuning conducted using `RandomizedSearchCV` across tree depth, learning rate, subsample ratios, and estimator counts.
3. **Model Interpretability:**
   - Extracted transformer pipeline steps dynamically to feed clean feature sets into SHAP `TreeExplainer` for zero-friction interpretation.

---

## 📁 Repository Structure

```text
├── .ipynb_checkpoints/
├── data/
│   └── ai4i2020.csv                          # AI4I 2020 Predictive Maintenance Dataset
├── AI4I Predictive Maintenance.ipynb          # Model training, EDA, and hyperparameter tuning notebook
├── AI4I_Predictive_Maintenance_XGB_model_1.joblib # Serialized XGBoost pipeline model
├── app.py                                    # Interactive Streamlit web application
├── requirements.txt                          # Python package dependencies
└── README.md                                 # Project documentation
```
````

---

## 🚀 Quickstart Guide

### Prerequisites

- **Python 3.9+**
- Git installed on your local machine

### 1. Clone the Repository

```bash
git clone [https://github.com/thedavidoyeleke/AI4I-Predictive-Maintenance.git](https://github.com/thedavidoyeleke/AI4I-Predictive-Maintenance.git)
cd AI4I-Predictive-Maintenance

```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Streamlit Application

```bash
streamlit run app.py

```

Open `http://localhost:8501` in your browser to interact with the dashboard.

---

## 📦 Dependencies

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `xgboost`
- `joblib`
- `shap`
- `matplotlib`

---

## 📊 Dataset Reference

The dataset used in this project is the **AI4I 2020 Predictive Maintenance Dataset**, sourced from the UCI Machine Learning Repository. It reflects real-world synthetic industrial data gathered from a milling machine with 10,000 data points and 14 operational features.

---

## 👤 Author

**David Olayimika Oyeleke**

- **GitHub:** [@thedavidoyeleke](https://www.google.com/search?q=https://github.com/thedavidoyeleke)
- **Role:** Data Scientist & Machine Learning Engineer

```

```
