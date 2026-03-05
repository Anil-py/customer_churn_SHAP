import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import json
import re
from streamlit_shap import st_shap

# --- BUG FIX: INTERCEPTOR FOR XGBOOST/SHAP ---
# This patches the known compatibility bug between XGBoost 3.0+ and SHAP
original_save_config = xgb.core.Booster.save_config

def patched_save_config(self, *args, **kwargs):
    config_str = original_save_config(self, *args, **kwargs)
    config = json.loads(config_str)
    
    # Secretly remove the brackets before SHAP sees them
    try:
        bs = config["learner"]["learner_model_param"]["base_score"]
        if isinstance(bs, str) and "[" in bs:
            config["learner"]["learner_model_param"]["base_score"] = re.sub(r"[^\d\.E\-]", "", bs)
    except KeyError:
        pass
        
    return json.dumps(config)

xgb.core.Booster.save_config = patched_save_config
# ---------------------------------------------

# --- 1. Page Configuration ---
st.set_page_config(page_title="Bank Customer Churn Predictor V3", layout="wide")

# --- 2. Asset Loading ---
@st.cache_resource
def load_model():
    return joblib.load('xgb_churn_model.pkl')

model = load_model()

# --- 3. Sidebar - Customer Input ---
st.sidebar.header("📋 Customer Financial Profile")

def user_input_features():
    geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 92, 40)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
    num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
    est_salary = st.sidebar.number_input("Estimated Annual Salary ($)", 0.0, 200000.0, 75000.0)

    # Map Geography (France is baseline 0,0)
    if geography == "Germany":
        geo_germany, geo_spain = 1, 0
    elif geography == "Spain":
        geo_germany, geo_spain = 0, 1
    else:  
        geo_germany, geo_spain = 0, 0

    # Map Binary variables
    is_active_val = 1 if is_active == "Yes" else 0
    has_card_val = 1 if has_card == "Yes" else 0

    # Build the Dictionary WITH SYNTHETIC FEATURES
    data = {
        'CreditScore': credit_score,
        'Gender': 1 if gender == "Female" else 0,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_card_val,
        'IsActiveMember': is_active_val,
        'EstimatedSalary': est_salary,
        'Geography_Germany': geo_germany,
        'Geography_Spain': geo_spain,
        'BalanceSalaryRatio': balance / (est_salary + 1),
        'TenureByAge': tenure / age,
        'CreditScoreByAge': credit_score / age,
        'EngagementScore': is_active_val + has_card_val + num_products
    }
    
    df = pd.DataFrame(data, index=[0])
    
    # Force the exact column order XGBoost expects
    expected_columns = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 
        'Geography_Spain', 'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreByAge', 'EngagementScore'
    ]
    df = df[expected_columns]
    
    return df

# Generate the dataframe for prediction
input_df = user_input_features()

# --- 4. Main UI Layout ---
st.title("🏦 Bank Customer Churn Predictor V3")
st.markdown("This tool utilizes an **Optimized XGBoost Classifier** and **SHAP (Explainable AI)** to identify churn risk and the underlying financial drivers.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risk Analysis")
    
    # Predict Probability
    raw_output = model.predict_proba(input_df)[0][1]
    
    # Clean the probability just in case
    clean_prob = re.sub(r"[^\d\.E\-]", "", str(raw_output))
    prob = float(clean_prob)
        
    prediction = 1 if prob > 0.5 else 0
    
    if prediction == 1:
        st.error(f"### **High Risk of Churn**")
    else:
        st.success(f"### **Low Risk (Retention)**")
    
    st.metric("Churn Probability", f"{prob:.2%}")
    st.progress(prob)
    st.write("---")
    st.caption("**Strategic Insight:** Threshold is set at 50% for proactive outreach.")

with col2:
    st.subheader("Why this prediction? (SHAP)")
    
    # Initialize Explainer (The monkey patch above prevents the crash here!)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # Visualize Force Plot
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0]), height=200)
    
    st.info("**Guide:** Red bars increase churn risk. Blue bars decrease churn risk.")
