import streamlit as st
import pandas as pd
from src.predict import predict_fraud

st.title("💳 AI-Powered Banking Fraud Detection")

st.write("Enter transaction details to check fraud probability")

input_data = {}

for col in [
    'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
    'V10','V11','V12','V13','V14','V15','V16','V17',
    'V18','V19','V20','V21','V22','V23','V24','V25',
    'V26','V27','V28','Amount'
]:
    input_data[col] = st.number_input(col, value=0.0)

if st.button("Check Fraud"):
    df = pd.DataFrame([input_data])
    prob = predict_fraud(df)

    st.write(f"Fraud Probability: **{prob:.2f}**")

    if prob > 0.7:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Transaction is Normal")