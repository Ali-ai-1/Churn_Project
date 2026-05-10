import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("models/best_model.pkl", "rb"))

st.title("Customer Churn Prediction")

# inputs
tenure = st.slider("Tenure", 1, 72, 12)
spend = st.number_input("Monthly Spend", 10.0, 500.0, 50.0)

if st.button("Predict"):
    data = pd.DataFrame([{
        "tenure_months": tenure,
        "monthly_spend": spend
    }])

    data = data.reindex(columns=model.feature_names_in_, fill_value=0)

    prob = model.predict_proba(data)[0][1]

    st.success(f"Churn Probability: {prob:.2%}")
