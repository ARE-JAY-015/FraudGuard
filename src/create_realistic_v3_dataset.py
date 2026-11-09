import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FRAUDGUARD V3
# REALISTIC CUSTOMER BEHAVIOR GENERATOR
# ============================================================

np.random.seed(42)

INPUT_PATH = Path("dataset/fraud_transactions.csv")
OUTPUT_PATH = Path("dataset/fraud_transactions_v3.csv")

NUMBER_OF_CUSTOMERS = 1000

print("=" * 65)
print("FRAUDGUARD V3 - REALISTIC CUSTOMER DATASET")
print("=" * 65)

# ------------------------------------------------------------
# 1. Load original dataset
# ------------------------------------------------------------

if not INPUT_PATH.exists():
    print("\nERROR: Original dataset not found!")
    print(INPUT_PATH.resolve())
    exit()

df = pd.read_csv(INPUT_PATH)

print(f"\nOriginal transactions: {len(df)}")

# ------------------------------------------------------------
# 2. Create customer IDs
# ------------------------------------------------------------

customer_ids = [
    f"CUST{str(i).zfill(4)}"
    for i in range(1, NUMBER_OF_CUSTOMERS + 1)
]

# Assign transactions to customers
df["customer_id"] = np.random.choice(
    customer_ids,
    size=len(df),
    replace=True
)

# ------------------------------------------------------------
# 3. Create customer profiles
# ------------------------------------------------------------

merchant_categories = [
    "grocery",
    "online_shopping",
    "restaurant",
    "fuel",
    "travel",
    "electronics"
]

payment_methods = [
    "UPI",
    "card",
    "wallet",
    "bank_transfer"
]

profiles = {}

for customer_id in customer_ids:

    # Spending level
    spending_type = np.random.choice(
        ["low", "medium", "high"],
        p=[0.45, 0.40, 0.15]
    )

    if spending_type == "low":
        normal_amount = np.random.uniform(300, 1200)

    elif spending_type == "medium":
        normal_amount = np.random.uniform(1200, 5000)

    else:
        normal_amount = np.random.uniform(5000, 25000)

    # Normal active hours
    normal_start = np.random.randint(6, 12)
    normal_end = np.random.randint(19, 24)

    # Preferred behavior
    preferred_merchant = np.random.choice(merchant_categories)

    preferred_payment = np.random.choice(payment_methods)

    # Typical distance
    normal_distance = np.random.uniform(2, 30)

    profiles[customer_id] = {
        "normal_amount": normal_amount,
        "normal_start": normal_start,
        "normal_end": normal_end,
        "preferred_merchant": preferred_merchant,
        "preferred_payment": preferred_payment,
        "normal_distance": normal_distance
    }

# ------------------------------------------------------------
# 4. Apply customer behavior
# ------------------------------------------------------------

for index, row in df.iterrows():

    customer_id = row["customer_id"]
    profile = profiles[customer_id]

    # Current fraud label
    is_fraud = int(row["fraud"])

    if is_fraud == 0:

        # ----------------------------------------------------
        # NORMAL TRANSACTION
        # ----------------------------------------------------

        # Amount close to customer's normal spending
        amount = np.random.normal(
            profile["normal_amount"],
            profile["normal_amount"] * 0.25
        )

        amount = max(amount, 10)

        # Normal transaction hour
        hour = np.random.randint(
            profile["normal_start"],
            profile["normal_end"] + 1
        )

        # Normal distance
        distance = np.random.normal(
            profile["normal_distance"],
            5
        )

        distance = max(distance, 0.1)

        # Normal device
        device_changed = 0

        # Mostly domestic
        international = np.random.choice(
            [0, 1],
            p=[0.95, 0.05]
        )

        # Usually no failed attempts
        failed_attempts = np.random.choice(
            [0, 1, 2],
            p=[0.90, 0.08, 0.02]
        )

        merchant = profile["preferred_merchant"]
        payment = profile["preferred_payment"]

    else:

        # ----------------------------------------------------
        # FRAUDULENT TRANSACTION
        # ----------------------------------------------------

        # Much larger amount
        amount = profile["normal_amount"] * np.random.uniform(
            4, 15
        )

        # Unusual time
        unusual_hours = list(
            set(range(24)) -
            set(range(
                profile["normal_start"],
                profile["normal_end"] + 1
            ))
        )

        hour = np.random.choice(unusual_hours)

        # Unusual distance
        distance = profile["normal_distance"] * np.random.uniform(
            3, 10
        )

        distance = max(distance, 10)

        # New device
        device_changed = 1

        # Higher probability of international
        international = np.random.choice(
            [0, 1],
            p=[0.35, 0.65]
        )

        # Failed attempts more likely
        failed_attempts = np.random.choice(
            [1, 2, 3, 4, 5],
            p=[0.10, 0.20, 0.30, 0.25, 0.15]
        )

        # Fraud may use unusual merchant/payment method
        merchant = np.random.choice(merchant_categories)

        payment = np.random.choice(payment_methods)

    # --------------------------------------------------------
    # Write generated behavioral values
    # --------------------------------------------------------

    df.at[index, "amount"] = amount
    df.at[index, "transaction_hour"] = hour
    df.at[index, "distance_from_home"] = distance
    df.at[index, "device_changed"] = device_changed
    df.at[index, "international"] = international
    df.at[index, "failed_attempts"] = failed_attempts
    df.at[index, "merchant_category"] = merchant
    df.at[index, "payment_method"] = payment

# ------------------------------------------------------------
# 5. Calculate customer historical average amount
# ------------------------------------------------------------

df["avg_amount"] = (
    df.groupby("customer_id")["amount"]
    .transform("mean")
)

# ------------------------------------------------------------
# 6. Calculate transaction count per customer
# ------------------------------------------------------------

customer_transaction_count = (
    df.groupby("customer_id")["transaction_id"]
    .transform("count")
)

df["transactions_last_24h"] = np.minimum(
    customer_transaction_count - 1,
    12
)

# ------------------------------------------------------------
# 7. Preserve column order
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
# 8. Save
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------------------------
# 9. Statistics
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("CUSTOMER STATISTICS")
print("-" * 65)

counts = df["customer_id"].value_counts()

print(f"Customers: {df['customer_id'].nunique()}")
print(f"Average transactions/customer: {counts.mean():.2f}")
print(f"Minimum transactions/customer: {counts.min()}")
print(f"Maximum transactions/customer: {counts.max()}")

print("\n" + "-" * 65)
print("FRAUD STATISTICS")
print("-" * 65)

fraud_count = df["fraud"].sum()
legitimate_count = len(df) - fraud_count

print(f"Legitimate: {legitimate_count}")
print(f"Fraud: {fraud_count}")
print(f"Fraud percentage: {fraud_count / len(df) * 100:.2f}%")

print("\n" + "-" * 65)
print("OUTPUT")
print("-" * 65)

print(f"Saved to:")
print(OUTPUT_PATH.resolve())

print("\n" + "=" * 65)
print("REALISTIC V3 DATASET CREATED")
print("=" * 65)