import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of transactions
n = 10000

# Create transaction data
data = {
    "transaction_id": range(1, n + 1),

    "amount": np.random.lognormal(
        mean=6,
        sigma=1,
        size=n
    ),

    "transaction_hour": np.random.randint(
        0, 24, n
    ),

    "distance_from_home": np.random.exponential(
        scale=20,
        size=n
    ),

    "transactions_last_24h": np.random.poisson(
        lam=3,
        size=n
    ),

    "avg_amount": np.random.lognormal(
        mean=5.5,
        sigma=0.7,
        size=n
    ),

    "device_changed": np.random.choice(
        [0, 1],
        size=n,
        p=[0.9, 0.1]
    ),

    "international": np.random.choice(
        [0, 1],
        size=n,
        p=[0.95, 0.05]
    ),

    "failed_attempts": np.random.poisson(
        lam=0.5,
        size=n
    ),

    "account_age_days": np.random.randint(
        30, 3000, n
    ),

    "merchant_category": np.random.choice(
        [
            "grocery",
            "electronics",
            "travel",
            "restaurant",
            "fuel",
            "online_shopping"
        ],
        size=n
    ),

    "payment_method": np.random.choice(
        [
            "card",
            "UPI",
            "bank_transfer",
            "wallet"
        ],
        size=n
    )
}

# Convert to DataFrame
df = pd.DataFrame(data)

# --------------------------------
# Create fraud score
# --------------------------------

fraud_score = (
    (df["amount"] > 3000).astype(int) * 2
    + (df["transaction_hour"].isin([0, 1, 2, 3, 4])).astype(int) * 2
    + (df["distance_from_home"] > 100).astype(int) * 2
    + (df["transactions_last_24h"] > 10).astype(int) * 2
    + (df["device_changed"] == 1).astype(int) * 2
    + (df["international"] == 1).astype(int) * 2
    + (df["failed_attempts"] > 2).astype(int) * 2
)

# Add randomness
random_component = np.random.random(n)

# Create fraud label
df["fraud"] = (
    (fraud_score >= 5)
    & (random_component > 0.25)
).astype(int)

# Save CSV
df.to_csv("fraud_transactions.csv", index=False)

# Show results
print("===================================")
print("FraudGuard Dataset Created!")
print("===================================")

print(f"Total transactions: {len(df)}")
print(f"Fraudulent transactions: {df['fraud'].sum()}")
print(f"Legitimate transactions: {(df['fraud'] == 0).sum()}")

print("\nFirst 5 transactions:")
print(df.head())

print("\nDataset columns:")
print(df.columns.tolist())