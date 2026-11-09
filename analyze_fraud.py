import pandas as pd
import numpy as np

df = pd.read_csv("dataset/fraud_transactions.csv")

print("=" * 65)
print("FRAUDGUARD: EXECUTIVE DATA & BUSINESS ANALYSIS")
print("=" * 65)

# 1. Class Imbalance
total_tx = len(df)
fraud_tx = df["fraud"].sum()
legit_tx = total_tx - fraud_tx
fraud_rate = (fraud_tx / total_tx) * 100

print(f"\n1. TRANSACTION VOLUME & CLASS BALANCE:")
print(f"   Total Transactions : {total_tx:,}")
print(f"   Legitimate (0)     : {legit_tx:,} ({100 - fraud_rate:.2f}%)")
print(f"   Fraudulent (1)     : {fraud_tx:,} ({fraud_rate:.2f}%)")

# 2. Financial Exposure (Dollar Loss)
total_volume = df["amount"].sum()
fraud_volume = df[df["fraud"] == 1]["amount"].sum()
fraud_vol_pct = (fraud_volume / total_volume) * 100

print(f"\n2. FINANCIAL EXPOSURE:")
print(f"   Total Processed Volume : ${total_volume:,.2f}")
print(f"   Total Fraud Loss       : ${fraud_volume:,.2f} ({fraud_vol_pct:.2f}% of total volume)")
print(f"   Avg Legit Amount       : ${df[df['fraud'] == 0]['amount'].mean():.2f}")
print(f"   Avg Fraud Amount       : ${df[df['fraud'] == 1]['amount'].mean():.2f}")

# 3. Behavioral Risk Multipliers
print(f"\n3. HIGH-RISK BEHAVIORAL SIGNALS:")

# Device Changed
dev_fraud = df.groupby("device_changed")["fraud"].mean() * 100
print(f"   Fraud Rate (Device NOT Changed) : {dev_fraud.get(0, 0):.2f}%")
print(f"   Fraud Rate (Device Changed)     : {dev_fraud.get(1, 0):.2f}%")

# International
intl_fraud = df.groupby("international")["fraud"].mean() * 100
print(f"   Fraud Rate (Domestic)           : {intl_fraud.get(0, 0):.2f}%")
print(f"   Fraud Rate (International)      : {intl_fraud.get(1, 0):.2f}%")

# Failed Attempts
fail_fraud = df.groupby("failed_attempts")["fraud"].mean() * 100
print("\n   Fraud Rate by Failed Login Attempts:")
for attempts, rate in fail_fraud.items():
    print(f"     - {attempts} failed attempts: {rate:.2f}% fraud rate")

# 4. Merchant Category Breakdown
print(f"\n4. FRAUD CONCENTRATION BY MERCHANT CATEGORY:")
cat_summary = df.groupby("merchant_category").agg(
    total_tx=("fraud", "count"),
    fraud_tx=("fraud", "sum"),
    fraud_rate=("fraud", lambda x: (x.sum() / len(x)) * 100),
    total_loss=("amount", lambda x: x[df.loc[x.index, "fraud"] == 1].sum())
).sort_values("fraud_rate", ascending=False)

print(cat_summary.to_string())

print("\n" + "=" * 65)