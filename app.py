import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import joblib
from streamlit_shap import st_shap

# 1. Page Configuration
st.set_page_config(page_title="Bank Churn Strategic Advisor", layout="wide")

# 2. Asset Loading
@st.cache_resource
def load_model():
    # Ensure 'xgb_churn_model.pkl' is in the same folder as this script
    return joblib.load('xgb_churn_model.pkl')

model = load_model()

# 3. Sidebar - Customer Input
st.sidebar.header("📋 Customer Financial Profile")

def user_input_features():
    # Location Selection
    geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    
    # Financial & Demographic Inputs
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 92, 40)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
    num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
    est_salary = st.sidebar.number_input("Estimated Annual Salary ($)", 0.0, 200000.0, 75000.0)

    # --- Geography Mapping (N-1 Encoding) ---
    if geography == "Germany":
        geo_germany, geo_spain = 1, 0
    elif geography == "Spain":
        geo_germany, geo_spain = 0, 1
    else:  # France
        geo_germany, geo_spain = 0, 0

    # Create Dictionary (Matches training order: Raw first, then Synthetic)
    data = {
        'CreditScore': credit_score,
        'Gender': 1 if gender == "Female" else 0,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active == "Yes" else 0,
        'EstimatedSalary': est_salary,
        'Geography_Germany': geo_germany,
        'Geography_Spain': geo_spain,
        # Synthetic Features (Calculated at the end)
        'BalanceSalaryRatio': balance / (est_salary + 1),
        'TenureByAge': tenure / age,
        'CreditScoreByAge': credit_score / age,
        'EngagementScore': (1 if is_active == "Yes" else 0) + (1 if has_card == "Yes" else 0) + num_products
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Main UI Layout
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("""
This tool utilizes an **Optimized XGBoost Classifier** and **SHAP (Explainable AI)** to identify churn risk and the underlying financial drivers for individual customers.
""")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risk Analysis")
    
    # 1. Get raw probability
    raw_prob = model.predict_proba(input_df)[0][1]
    
    # 2. Convert to string and clean (strips brackets if they exist)
    clean_prob = str(raw_prob).replace('[', '').replace(']', '')
    
    # 3. Final conversion to float
    prob = float(clean_prob)
    
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.error(f"### **High Risk of Churn**")
    else:
        st.success(f"### **Low Risk (Retention)**")
    
    st.metric("Churn Probability", f"{prob:.2%}")
    st.progress(prob)
    st.write("---")
    st.caption("**Decision Logic:** A probability above 50% triggers a high-risk alert.")

with col2:
    st.subheader("Why this prediction? (SHAP)")
    # Initialize the SHAP explainer for the tree model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # Visualize the Force Plot
    st_shap(shap.force_plot(explainer.expected_value, shap_values, input_df), height=200)
    
    st.info("""
    **How to read this plot:** - **Red features** push the probability toward Churn (higher).
    - **Blue features** pull the probability toward Retention (lower).
    - The length of the bar represents the **strength** of that feature's impact.
    """)

st.divider()

st.markdown("**Note:** This model was optimized for high Recall (0.71) to ensure the majority of at-risk customers are identified.")

