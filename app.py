import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="FraudGuard Analytics", layout="wide")

st.title("🛡️ FraudGuard: Transaction Risk Telemetry")
st.markdown("Real-time fraud detection and behavioral risk scoring platform.")

@st.cache_data
def load_data():
    return pd.read_csv("dataset/fraud_transactions.csv")

df = load_data()

# Load Model
try:
    with open("fraud_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
        model = artifacts["model"]
        model_columns = artifacts["columns"]
except Exception:
    model = None

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
total_tx = len(df)
fraud_tx = int(df["fraud"].sum())
fraud_rate = (fraud_tx / total_tx) * 100
fraud_loss = df[df["fraud"] == 1]["amount"].sum()

col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("Flagged Fraud", f"{fraud_tx:,}")
col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
col4.metric("Prevented Loss", f"${fraud_loss:,.2f}")

st.divider()

# Live Risk Evaluator
st.subheader("⚡ Real-Time Transaction Risk Scorer")
st.markdown("Simulate a transaction to evaluate risk score and anomaly flags.")

with st.form("risk_form"):
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Transaction Amount ($)", value=250.0)
    avg_amount = c2.number_input("Customer Avg Amount ($)", value=120.0)
    distance = c3.number_input("Distance from Home (miles)", value=15.0)
    
    c4, c5, c6 = st.columns(3)
    tx_24h = c4.slider("Transactions in Last 24h", 0, 20, 2)
    failed_attempts = c5.slider("Failed Password Attempts", 0, 5, 0)
    tx_hour = c6.slider("Hour of Day (0-23)", 0, 23, 14)

    c7, c8, c9 = st.columns(3)
    category = c7.selectbox("Merchant Category", df["merchant_category"].unique())
    payment = c8.selectbox("Payment Method", df["payment_method"].unique())
    device_changed = c9.checkbox("Device Changed")
    international = c9.checkbox("International Transaction")

    submitted = st.form_submit_button("Score Transaction")

    if submitted and model is not None:
        input_dict = {
            "amount": amount,
            "transaction_hour": tx_hour,
            "distance_from_home": distance,
            "transactions_last_24h": tx_24h,
            "avg_amount": avg_amount,
            "device_changed": int(device_changed),
            "international": int(international),
            "failed_attempts": failed_attempts,
            "account_age_days": 365,
            "amount_to_avg_ratio": amount / (avg_amount + 1e-5),
        }
        
        # Build vector matching model columns
        input_df = pd.DataFrame([input_dict])
        for col in model_columns:
            if col.startswith("merchant_category_"):
                cat_name = col.replace("merchant_category_", "")
                input_df[col] = int(category == cat_name)
            elif col.startswith("payment_method_"):
                pay_name = col.replace("payment_method_", "")
                input_df[col] = int(payment == pay_name)
            elif col not in input_df.columns:
                input_df[col] = 0
                
        input_df = input_df[model_columns]
        risk_score = model.predict_proba(input_df)[0][1]

        if risk_score > 0.65:
            st.error(f"🚨 HIGH RISK DETECTED | Fraud Probability: {risk_score * 100:.1f}% — Transaction Blocked")
        elif risk_score > 0.35:
            st.warning(f"⚠️ SUSPICIOUS TRANSACTION | Fraud Probability: {risk_score * 100:.1f}% — 3D Secure / OTP Required")
        else:
            st.success(f"✅ APPROVED | Fraud Probability: {risk_score * 100:.1f}% — Safe Transaction")

st.divider()

# Charts
c1, c2 = st.columns(2)
with c1:
    st.subheader("Fraud Rate by Merchant Category")
    cat_rates = df.groupby("merchant_category")["fraud"].mean() * 100
    st.bar_chart(cat_rates)

with c2:
    st.subheader("Transaction Velocity vs. Fraud")
    vel_rates = df.groupby("transactions_last_24h")["fraud"].mean() * 100
    st.line_chart(vel_rates)