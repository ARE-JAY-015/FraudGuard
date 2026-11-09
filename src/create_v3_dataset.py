import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FRAUDGUARD V3 - DATASET PREPARATION
# ============================================================

# Paths
INPUT_PATH = Path("dataset/fraud_transactions.csv")
OUTPUT_PATH = Path("dataset/fraud_transactions_v3.csv")

# Reproducibility
np.random.seed(42)

print("=" * 60)
print("FRAUDGUARD V3 - DATASET PREPARATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load original dataset
# ------------------------------------------------------------

if not INPUT_PATH.exists():
    print("\nERROR: Original dataset not found!")
    print(INPUT_PATH.resolve())
    exit()

df = pd.read_csv(INPUT_PATH)

print(f"\nOriginal rows: {len(df)}")
print(f"Original columns: {len(df.columns)}")

# ------------------------------------------------------------
# 2. Validate transaction IDs
# ------------------------------------------------------------

if df["transaction_id"].duplicated().any():
    print("\nWARNING: Duplicate transaction IDs found.")
    df["transaction_id"] = range(1, len(df) + 1)
else:
    print("\nTransaction IDs are unique.")

# ------------------------------------------------------------
# 3. Create customer IDs
# ------------------------------------------------------------

number_of_customers = 1000

customer_ids = [
    f"CUST{str(i).zfill(4)}"
    for i in range(1, number_of_customers + 1)
]

# Randomly assign transactions to customers
df["customer_id"] = np.random.choice(
    customer_ids,
    size=len(df),
    replace=True
)

# ------------------------------------------------------------
# 4. Reorder columns
# ------------------------------------------------------------

columns = [
    "transaction_id",
    "customer_id",
    "amount",
    "transaction_hour",
    "distance_from_home",
    "transactions_last_24h",
    "avg_amount",
    "device_changed",
    "international",
    "failed_attempts",
    "account_age_days",
    "merchant_category",
    "payment_method",
    "fraud"
]

df = df[columns]

# ------------------------------------------------------------
# 5. Display customer statistics
# ------------------------------------------------------------

customer_counts = df["customer_id"].value_counts()

print("\nCustomer statistics:")
print(f"Total customers: {df['customer_id'].nunique()}")
print(f"Average transactions/customer: {customer_counts.mean():.2f}")
print(f"Minimum transactions/customer: {customer_counts.min()}")
print(f"Maximum transactions/customer: {customer_counts.max()}")

# ------------------------------------------------------------
# 6. Fraud statistics
# ------------------------------------------------------------

fraud_count = df["fraud"].sum()
legitimate_count = len(df) - fraud_count

print("\nFraud distribution:")
print(f"Legitimate transactions: {legitimate_count}")
print(f"Fraudulent transactions: {fraud_count}")

fraud_percentage = (fraud_count / len(df)) * 100

print(f"Fraud percentage: {fraud_percentage:.2f}%")

# ------------------------------------------------------------
# 7. Save V3 dataset
# ------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nV3 dataset created successfully!")
print(f"Saved to: {OUTPUT_PATH.resolve()}")

print("\nNew columns:")
print("customer_id")

print("\n" + "=" * 60)
print("PHASE 1 STEP 2 COMPLETE")
print("=" * 60)