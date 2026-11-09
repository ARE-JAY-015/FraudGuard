import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FRAUDGUARD V3
# REALISTIC TRANSACTION TIMESTAMPS
# ============================================================

np.random.seed(42)

INPUT_PATH = Path("dataset/fraud_transactions_v3.csv")
OUTPUT_PATH = Path("dataset/fraud_transactions_v3_timestamped.csv")

print("=" * 65)
print("FRAUDGUARD V3 - REALISTIC TIMESTAMP GENERATION")
print("=" * 65)

# ------------------------------------------------------------
# 1. Load V3 dataset
# ------------------------------------------------------------

if not INPUT_PATH.exists():
    print("\nERROR: V3 dataset not found!")
    print(INPUT_PATH.resolve())
    exit()

df = pd.read_csv(INPUT_PATH)

print(f"\nLoaded transactions: {len(df)}")

# ------------------------------------------------------------
# 2. Create base date
# ------------------------------------------------------------

start_date = pd.Timestamp("2026-01-01")

# ------------------------------------------------------------
# 3. Generate timestamps
# ------------------------------------------------------------

timestamps = []

for index, row in df.iterrows():

    # Spread transactions across approximately 6 months
    random_days = np.random.randint(0, 180)

    base_date = start_date + pd.Timedelta(
        days=int(random_days)
    )

    # Use the existing transaction hour
    hour = int(row["transaction_hour"])

    # Random minutes and seconds
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)

    timestamp = base_date.replace(
        hour=hour,
        minute=int(minute),
        second=int(second)
    )

    timestamps.append(timestamp)

df["timestamp"] = timestamps

# ------------------------------------------------------------
# 4. Convert to datetime
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ------------------------------------------------------------
# 5. Sort chronologically
# ------------------------------------------------------------

df = df.sort_values(
    by=["customer_id", "timestamp"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 6. Calculate time since previous customer transaction
# ------------------------------------------------------------

df["previous_transaction_time"] = (
    df.groupby("customer_id")["timestamp"]
    .shift(1)
)

df["minutes_since_previous"] = (
    (
        df["timestamp"]
        - df["previous_transaction_time"]
    ).dt.total_seconds() / 60
)

# First transaction for each customer
df["minutes_since_previous"] = (
    df["minutes_since_previous"]
    .fillna(-1)
)

# ------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------------------------
# 8. Display results
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("TIMESTAMP INFORMATION")
print("-" * 65)

print(
    f"First transaction: "
    f"{df['timestamp'].min()}"
)

print(
    f"Last transaction: "
    f"{df['timestamp'].max()}"
)

print(
    f"Unique dates: "
    f"{df['timestamp'].dt.date.nunique()}"
)

print("\nSample transactions:")

print(
    df[
        [
            "transaction_id",
            "customer_id",
            "timestamp",
            "amount",
            "fraud"
        ]
    ].head(15).to_string(index=False)
)

print("\n" + "-" * 65)
print("OUTPUT")
print("-" * 65)

print(
    f"Saved to:\n"
    f"{OUTPUT_PATH.resolve()}"
)

print("\n" + "=" * 65)
print("TIMESTAMP GENERATION COMPLETE")
print("=" * 65)