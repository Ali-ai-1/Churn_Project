import streamlit as st
import pandas as pd
import pickle

# load model
model = pickle.load(open("models/best_model.pkl", "rb"))

st.title("Customer Churn Prediction System")

# basic inputs
tenure = st.slider("Tenure Months", 1, 72, 12)
monthly_spend = st.number_input("Monthly Spend", 10.0, 500.0, 50.0)
days_since_login = st.slider("Days Since Login", 0, 90, 10)
contract_type = st.selectbox("Contract Type", ["monthly", "annual", "two_year"])

if st.button("Predict"):

    # input data (sirf user inputs)
    input_dict = {
        "tenure_months": tenure,
        "monthly_spend": monthly_spend,
        "days_since_login": days_since_login,
        "contract_type": contract_type,
    }

    df = pd.DataFrame([input_dict])

    # 🔥 IMPORTANT FIX: match model expected columns
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_names_in_]

    # prediction
    prob = model.predict_proba(df)[0][1]

    st.success(f"Churn Probability: {prob:.2%}")
