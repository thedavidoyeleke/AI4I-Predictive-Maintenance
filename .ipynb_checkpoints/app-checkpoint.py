import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="AI4I Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

# Load the saved pipeline model
@st.cache_resource
def load_model():
    return joblib.load("AI4I_Predictive_Maintenance_XGB_model_1.joblib")

model = load_model()

# App Header
st.title("⚙️ AI4I Predictive Maintenance Dashboard")
st.markdown("Monitor industrial machine health, predict potential failures, and prevent unexpected downtime using machine learning.")

# Sidebar Input Controls for Machine Parameters
st.sidebar.header("🔧 Machine Operating Parameters")

def user_input_features():
    # Categorical feature
    machine_type = st.sidebar.selectbox("Machine Type (L/M/H)", ["L", "M", "H"])
    
    # Numerical features (using typical ranges from the AI4I dataset)
    air_temp = st.sidebar.number_input("Air temperature [K]", min_value=290.0, max_value=310.0, value=298.0, step=0.1)
    process_temp = st.sidebar.number_input("Process temperature [K]", min_value=300.0, max_value=320.0, value=308.6, step=0.1)
    rotational_speed = st.sidebar.number_input("Rotational speed [rpm]", min_value=1100, max_value=3000, value=1500, step=10)
    torque = st.sidebar.number_input("Torque [Nm]", min_value=3.8, max_value=80.0, value=40.0, step=0.5)
    tool_wear = st.sidebar.number_input("Tool wear [min]", min_value=0, max_value=250, value=0, step=1)
    
    # Bundle into a DataFrame matching the training structure
    data = {
        'Type': [machine_type],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    }
    return pd.DataFrame(data)

input_df = user_input_features()

# Main Display Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Current Machine Inputs")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("Action Center")
    predict_btn = st.button("Run Failure Prediction", type="primary", use_container_width=True)

# Prediction Logic
if predict_btn:
    # Predict class and probability using the pipeline
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    st.markdown("---")
    st.subheader("📊 Diagnostic Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.error("⚠️ **ALERT: High Risk of Machine Failure Predicted!**")
            st.markdown("Recommended Action: Schedule immediate inspection for thermal stress or tool wear.")
        else:
            st.success("✅ **Status: Machine Operating Normally**")
            st.markdown("No immediate failure indicators detected.")
            
    with res_col2:
        failure_prob = prediction_proba[1] * 100
        st.metric(label="Failure Probability", value=f"{failure_prob:.2f}%")

# Footer / Project Info
st.markdown("---")
st.markdown("Built with Python, XGBoost, and Streamlit.")