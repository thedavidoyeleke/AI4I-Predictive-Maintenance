import streamlit as st
import pandas as pd
import joblib
import datetime
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="AI4I Predictive Maintenance System",
    page_icon="⚙️",
    layout="centered"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'Timestamp', 'Machine Type', 'Air Temp [K]', 'Process Temp [K]', 
        'Rotational Speed [rpm]', 'Torque [Nm]', 'Tool Wear [min]', 
        'Prediction', 'Failure Probability [%]'
    ])

# Load model
@st.cache_resource
def load_model():
    return joblib.load("AI4I_Predictive_Maintenance_XGB_model_1.joblib")

model = load_model()

# --- DIAGNOSTIC & FINANCIAL FUNCTIONS ---
def diagnose_failure_mode(row):
    failures = []
    tool_wear = row['Tool wear [min]']
    if 200 <= tool_wear <= 240:
        failures.append("Tool Wear Failure (TWF): Tool exceeded operational wear limit.")
        
    air_temp = row['Air temperature [K]']
    process_temp = row['Process temperature [K]']
    rotational_speed = row['Rotational speed [rpm]']
    temp_diff = process_temp - air_temp
    if temp_diff < 8.6 and rotational_speed < 1380:
        failures.append("Heat Dissipation Failure (HDF): Insufficient cooling and low speed.")
        
    torque = row['Torque [Nm]']
    power = torque * rotational_speed * (2 * 3.14159 / 60)
    if power < 3500 or power > 9000:
        failures.append(f"Power Failure (PWF): Abnormal power load ({power:.1f}W detected).")
        
    wear_torque_product = tool_wear * torque
    machine_type = row['Type']
    limit = 11000 if machine_type == 'L' else (12000 if machine_type == 'M' else 13000)
    if wear_torque_product > limit:
        failures.append("Overstrain Failure (OSF): Excessive mechanical stress on tool.")
        
    if not failures:
        failures.append("Random/Multi-factor Failure: Inspect overall assembly.")
    return failures

def calculate_financial_impact(is_failure_predicted, emergency_cost=12000, scheduled_cost=1500):
    if is_failure_predicted:
        return {
            "cost_if_ignored": emergency_cost,
            "cost_if_addressed": scheduled_cost,
            "net_savings": emergency_cost - scheduled_cost
        }
    return {"cost_if_ignored": 0, "cost_if_addressed": 0, "net_savings": 0}

def strict_validate_batch(df):
    expected_columns = [
        'Type', 'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        return None, None, [f"Missing required columns in CSV: {missing_cols}"]

    df_work = df.copy()
    numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    row_errors = {}

    for idx, row in df_work.iterrows():
        errs = []
        m_type = str(row['Type']).strip().upper()
        if pd.isnull(row['Type']) or m_type not in ['L', 'M', 'H']:
            errs.append(f"Invalid or missing machine Type ('{row['Type']}'). Must be L, M, or H.")
            
        for col in numeric_cols:
            val = row[col]
            if pd.isnull(val):
                errs.append(f"Missing value in '{col}'.")
            else:
                try:
                    float(val)
                except ValueError:
                    errs.append(f"Corrupted non-numeric value in '{col}' ('{val}').")
                    
        if errs:
            row_errors[idx] = errs

    invalid_indices = list(row_errors.keys())
    valid_indices = [i for i in df_work.index if i not in invalid_indices]

    df_valid = df_work.loc[valid_indices, expected_columns].copy()
    df_invalid = df_work.loc[invalid_indices].copy() if invalid_indices else pd.DataFrame(columns=df_work.columns)
    
    for col in numeric_cols:
        df_valid[col] = df_valid[col].astype(float)
    df_valid['Type'] = df_valid['Type'].astype(str).str.upper().str.strip()

    return df_valid, df_invalid, row_errors


# ==========================================
# PAGE 1: WELCOME & LANDING PAGE
# ==========================================
if st.session_state.page == 'welcome':
    st.title("⚙️ AI4I Predictive Maintenance Intelligence")
    st.markdown("""
    Welcome to the **David Oyeleke** industrial monitoring dashboard. 
    This application utilizes an advanced machine learning pipeline powered by **XGBoost** and **SMOTE** to forecast industrial machinery failures before they occur.
    """)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Prediction Dashboard", type="primary", use_container_width=True):
            st.session_state.page = 'app'
            st.rerun()

# ==========================================
# PAGE 2: PREDICTION WORKSPACE
# ==========================================
elif st.session_state.page == 'app':
    if st.button("⬅️ Back to Home"):
        st.session_state.page = 'welcome'
        st.rerun()

    st.title("🔧 Industrial Maintenance Control Center")
    tab1, tab2 = st.tabs(["🔍 Single Machine Analysis & Explainable AI", "📁 Batch CSV Processing"])
    
    # --- TAB 1: SINGLE MACHINE FORM & SHAP ---
    with tab1:
        st.markdown("Enter live sensor metrics below to evaluate individual machine status and feature attribution.")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                machine_type = st.selectbox("Machine Type (L/M/H)", ["L", "M", "H"])
                air_temp = st.number_input("Air temperature [K]", min_value=290.0, max_value=310.0, value=298.0, step=0.1)
                process_temp = st.number_input("Process temperature [K]", min_value=300.0, max_value=320.0, value=308.6, step=0.1)
            with col2:
                rotational_speed = st.number_input("Rotational speed [rpm]", min_value=1100, max_value=3000, value=1500, step=10)
                torque = st.number_input("Torque [Nm]", min_value=3.8, max_value=80.0, value=40.0, step=0.5)
                tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=250, value=0, step=1)
                
            submitted = st.form_submit_button("Run Failure Analysis & Explain", type="primary", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame({
                'Type': [machine_type],
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear]
            })
            
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            failure_prob = proba[1] * 100
            safe_prob = proba[0] * 100
            
            # --- RUL (Remaining Useful Life) Calculation ---
            max_wear_limit = 240.0 # Standard AI4I dataset failure threshold for tool wear
            current_wear = float(tool_wear)
            rul_minutes = max(0.0, max_wear_limit - current_wear)
            wear_percentage = min(100.0, (current_wear / max_wear_limit) * 100)
            
            st.markdown("---")
            st.subheader("📊 Diagnostic Results & Remaining Useful Life (RUL)")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction == 1:
                    st.error("⚠️ **ALERT: High Risk of Machine Failure!**")
                    detected_failures = diagnose_failure_mode(input_df.iloc[0])
                    st.markdown("**Identified Root Causes:**")
                    for cause in detected_failures:
                        st.write(f"- {cause}")
                else:
                    st.success("✅ **Status: Machine Operating Normally**")
                    st.markdown("No immediate failure indicators identified.")
                    
            with res_col2:
                st.metric(label="Calculated Failure Risk", value=f"{failure_prob:.2f}%")
                st.metric(label="Estimated Remaining Useful Life (RUL)", value=f"{rul_minutes:.1f} mins", delta=f"-{current_wear:.1f} min wear used")
                
            # --- RUL VISUAL PROGRESS RUNWAY ---
            st.markdown("### ⏳ Tool Degradation & Operational Runway")
            st.progress(wear_percentage / 100.0, text=f"Tool Wear Progress: {current_wear} / {max_wear_limit} mins ({wear_percentage:.1f}% consumed)")
            
            if rul_minutes < 30:
                st.warning("⚠️ **Recommendation:** RUL is critically low. Schedule tool replacement before the next operational cycle.")
            else:
                st.info("💡 **Recommendation:** Tool operating within acceptable lifespan limits.")

            financials = calculate_financial_impact(prediction == 1)
            if prediction == 1:
                st.warning(f"💰 **Financial Impact Analysis:**\n"
                           f"- Emergency Breakdown Cost: **${financials['cost_if_ignored']:,}**\n"
                           f"- Scheduled Maintenance Cost: **${financials['cost_if_addressed']:,}**\n"
                           f"- **Estimated Savings:** **${financials['net_savings']:,}**")

            st.markdown("### 📈 Health Probability Breakdown")
            
            fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
            statuses = ['Safe / Normal', 'Failure Risk']
            probabilities = [safe_prob, failure_prob]
            bar_colors = ['#21c354', '#ff4b4b'] # Green for Safe, Red for Risk
            
            bars = ax_prob.bar(statuses, probabilities, color=bar_colors, width=0.5)
            ax_prob.set_ylim(0, 100)
            ax_prob.set_ylabel("Probability [%]")
            
            plt.xticks(rotation=0, fontweight='bold')
            
            for bar in bars:
                height = bar.get_height()
                ax_prob.text(bar.get_x() + bar.get_width()/2., height + 2, f'{height:.1f}%', 
                             ha='center', va='bottom', fontsize=10, fontweight='bold')
                             
            st.pyplot(fig_prob)
            plt.clf()

           # --- EXPLAINABLE AI (SHAP) INTEGRATION ---
            st.markdown("---")
            st.subheader("🧠 Explainable AI (SHAP Feature Contribution)")
            st.markdown("Understanding *why* the model made this prediction based on feature importance weights.")
            
            try:
                # Unwrap best estimator from RandomizedSearchCV
                active_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
                
                # Extract preprocessor and XGBoost model using pipeline step names
                preprocessor = active_model.named_steps['preprocessor']
                xgb_model = active_model.named_steps['model']
                
                # Transform input dataframe and fetch feature names
                transformed_input = preprocessor.transform(input_df)
                if hasattr(preprocessor, "get_feature_names_out"):
                    raw_feature_names = preprocessor.get_feature_names_out()
                    clean_feature_names = [
                        name.replace("[", "").replace("]", "").replace("<", "").replace(">", "")
                        for name in raw_feature_names
                    ]
                else:
                    clean_feature_names = input_df.columns.tolist()
                    
                transformed_df = pd.DataFrame(transformed_input, columns=clean_feature_names)
                
                # Calculate and plot SHAP values
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer(transformed_df)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
                plt.clf()

                # --- PLAIN ENGLISH INTERPRETATION ---
                st.markdown("#### 📖 Plain English Breakdown")
                st.info(
                    "**How to read this chart:**\n"
                    "- **Blue bars** push the machine *away* from failure (indicating safe operating conditions).\n"
                    "- **Red bars** push the machine *toward* a potential failure risk.\n"
                    "- The length of each bar shows how strongly that specific sensor metric influenced the final decision."
                )
                
            except Exception as e:
                st.error(f"Could not render SHAP plot directly: {e}")

            # Log to session history
            new_entry = pd.DataFrame({
                'Timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'Machine Type': [machine_type],
                'Air Temp [K]': [air_temp],
                'Process Temp [K]': [process_temp],
                'Rotational Speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool Wear [min]': [tool_wear],
                'Prediction': ['Failure' if prediction == 1 else 'Normal'],
                'Failure Probability [%]': [round(failure_prob, 2)]
            })
            st.session_state.history = pd.concat([new_entry, st.session_state.history], ignore_index=True)

    # --- TAB 2: BATCH CSV PROCESSING ---
    with tab2:
        st.markdown("Upload a CSV file containing multiple machine logs to run bulk failure predictions with strict error handling.")
        uploaded_file = st.file_uploader("Choose a CSV file for batch analysis", type="csv")

        if uploaded_file is not None:
            try:
                raw_batch_df = pd.read_csv(uploaded_file)
                st.write("Preview of Uploaded Data:")
                st.dataframe(raw_batch_df.head(), use_container_width=True)
                
                if st.button("Validate & Run Batch Analysis", type="primary"):
                    df_valid, df_invalid, row_errors = strict_validate_batch(raw_batch_df)
                    st.session_state.df_valid = df_valid
                    st.session_state.df_invalid = df_invalid
                    st.session_state.row_errors = row_errors
                    st.session_state.raw_batch_df = raw_batch_df

                if 'row_errors' in st.session_state:
                    row_errors = st.session_state.row_errors
                    df_valid = st.session_state.df_valid
                    df_invalid = st.session_state.df_invalid
                    
                    if row_errors:
                        st.warning(f"⚠️ **Validation Alert:** Found errors in **{len(row_errors)} out of {len(st.session_state.raw_batch_df)}** rows.")
                        
                        error_summary = []
                        for idx, errs in row_errors.items():
                            error_summary.append({
                                'Row Number (1-indexed)': idx + 2,
                                'Errors Found': "; ".join(errs)
                            })
                        st.dataframe(pd.DataFrame(error_summary), use_container_width=True)
                        
                        remediation_choice = st.radio(
                            "How would you like to handle invalid records?",
                            [
                                "Run predictions ONLY on valid rows (ignore/exclude invalid rows)",
                                "Impute missing/invalid values with medians & valid defaults, then process all",
                                "Cancel and abort batch processing"
                            ]
                        )
                        
                        if st.button("Proceed with Selected Action", type="primary"):
                            if remediation_choice.startswith("Run predictions ONLY"):
                                predictions = model.predict(df_valid)
                                probabilities = model.predict_proba(df_valid)[:, 1] * 100
                                results_df = df_valid.copy()
                                results_df['Prediction Status'] = ['Failure Risk' if p == 1 else 'Normal' for p in predictions]
                                results_df['Failure Probability [%]'] = probabilities.round(2)
                                
                                st.success(f"✅ Successfully processed {len(results_df)} valid records. ({len(df_invalid)} invalid records excluded).")
                                st.dataframe(results_df, use_container_width=True)
                                
                                batch_csv = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Download Valid Records Report", data=batch_csv, file_name="valid_machines_report.csv", mime="text/csv")
                                
                            elif remediation_choice.startswith("Impute missing"):
                                repaired_df = st.session_state.raw_batch_df.copy()
                                numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
                                for col in numeric_cols:
                                    repaired_df[col] = pd.to_numeric(repaired_df[col], errors='coerce')
                                    repaired_df[col].fillna(repaired_df[col].median(), inplace=True)
                                
                                repaired_df['Type'] = repaired_df['Type'].astype(str).str.upper().str.strip()
                                repaired_df.loc[~repaired_df['Type'].isin(['L', 'M', 'H']), 'Type'] = 'L'
                                
                                predictions = model.predict(repaired_df[['Type'] + numeric_cols])
                                probabilities = model.predict_proba(repaired_df[['Type'] + numeric_cols])[:, 1] * 100
                                repaired_df['Prediction Status'] = ['Failure Risk' if p == 1 else 'Normal' for p in predictions]
                                repaired_df['Failure Probability [%]'] = probabilities.round(2)
                                
                                st.success("✅ Successfully imputed and processed all records!")
                                st.dataframe(repaired_df, use_container_width=True)
                                
                                batch_csv = repaired_df.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Download Imputed Batch Report", data=batch_csv, file_name="imputed_batch_report.csv", mime="text/csv")
                            else:
                                st.info("Batch processing aborted.")
                    else:
                        predictions = model.predict(df_valid)
                        probabilities = model.predict_proba(df_valid)[:, 1] * 100
                        results_df = df_valid.copy()
                        results_df['Prediction Status'] = ['Failure Risk' if p == 1 else 'Normal' for p in predictions]
                        results_df['Failure Probability [%]'] = probabilities.round(2)
                        
                        st.success(f"✅ All {len(results_df)} records passed validation cleanly!")
                        st.dataframe(results_df, use_container_width=True)
                        
                        batch_csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Batch Report", data=batch_csv, file_name="batch_report.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Failed to read file: {e}")

    # Session History Log
    if not st.session_state.history.empty:
        st.markdown("---")
        st.subheader("📋 Session Prediction History Log")
        st.dataframe(st.session_state.history, use_container_width=True)
        
        csv_data = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Session History CSV",
            data=csv_data,
            file_name="predictive_maintenance_session_history.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer with Contact & Project Info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><b>David Oyeleke • Predictive Maintenance Systems</b></p>
    <p>Get in touch with David: <a href='mailto:thedavidoyeleke@gmail.com'>thedavidoyeleke@gmail.com</a></p>
</div>
""", unsafe_allow_html=True)