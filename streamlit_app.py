import streamlit as st
import pandas as pd
import pickle

# load model
model = pickle.load(open("models/best_model.pkl", "rb"))

st.title("Customer Churn Prediction System")

tenure = st.slider("Tenure Months", 1, 72, 12)
monthly_spend = st.number_input("Monthly Spend", 10.0, 500.0, 50.0)
days_since_login = st.slider("Days Since Login", 0, 90, 10)

contract_type = st.selectbox("Contract Type", ["monthly", "annual", "two_year"])

if st.button("Predict"):

    input_dict = {
        "tenure_months": tenure,
        "monthly_spend": monthly_spend,
        "days_since_login": days_since_login,
        "contract_type": contract_type,

        "age": 30,
        "region": "North",
        "company_size": "small",
        "feature_adoption_score": 0.5,
        "avg_session_mins": 10,
        "logins_per_month": 12,
        "onboarding_pct": 0.8,
        "spend_mom_change_pct": 0,
        "payment_failures_12m": 0,
        "plan_downgrades": 0,
        "refund_requests": 0,
        "support_tickets_90d": 0,
        "nps_score": 7,
        "competitor_mentions": 0,
        "email_open_rate": 0.4,
        "days_to_first_value": 5
    }

    data = pd.DataFrame([input_dict])

    data = data.reindex(columns=model.feature_names_in_, fill_value=0)

    prob = model.predict_proba(data)[0][1]

    st.success(f"Churn Probability: {prob:.2%}")
