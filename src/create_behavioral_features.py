import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FRAUDGUARD V3
# HISTORICAL BEHAVIORAL FEATURE ENGINE
# ============================================================

INPUT_PATH = Path(
    "dataset/fraud_transactions_v3_timestamped.csv"
)

OUTPUT_PATH = Path(
    "dataset/fraud_transactions_v3_behavioral.csv"
)

print("=" * 70)
print("FRAUDGUARD V3 - HISTORICAL BEHAVIOR ENGINE")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

if not INPUT_PATH.exists():
    print("\nERROR: Dataset not found!")
    print(INPUT_PATH.resolve())
    exit()

df = pd.read_csv(INPUT_PATH)

print(f"\nLoaded transactions: {len(df)}")

# ------------------------------------------------------------
# 2. Convert timestamp
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# ------------------------------------------------------------
# 3. Sort by customer and time
# ------------------------------------------------------------

df = df.sort_values(
    ["customer_id", "timestamp"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 4. Previous transaction amount
# ------------------------------------------------------------

df["previous_amount"] = (
    df.groupby("customer_id")["amount"]
    .shift(1)
)

# ------------------------------------------------------------
# 5. Amount ratio to previous transaction
# ------------------------------------------------------------

df["amount_ratio_to_previous"] = (
    df["amount"] /
    df["previous_amount"]
)

df["amount_ratio_to_previous"] = (
    df["amount_ratio_to_previous"]
    .replace([np.inf, -np.inf], np.nan)
)

# First transaction has no previous amount
df["amount_ratio_to_previous"] = (
    df["amount_ratio_to_previous"]
    .fillna(1.0)
)

# ------------------------------------------------------------
# 6. Historical average BEFORE current transaction
# ------------------------------------------------------------

df["historical_avg_amount"] = (
    df.groupby("customer_id")["amount"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)

# ------------------------------------------------------------
# 7. Historical maximum BEFORE current transaction
# ------------------------------------------------------------

df["historical_max_amount"] = (
    df.groupby("customer_id")["amount"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .max()
    )
)

# ------------------------------------------------------------
# 8. Amount ratio to historical average
# ------------------------------------------------------------

df["amount_ratio_to_history"] = (
    df["amount"] /
    df["historical_avg_amount"]
)

df["amount_ratio_to_history"] = (
    df["amount_ratio_to_history"]
    .replace([np.inf, -np.inf], np.nan)
)

# First transaction
df["amount_ratio_to_history"] = (
    df["amount_ratio_to_history"]
    .fillna(1.0)
)

# ------------------------------------------------------------
# 9. Previous transaction timestamp
# ------------------------------------------------------------

df["previous_transaction_time"] = (
    df.groupby("customer_id")["timestamp"]
    .shift(1)
)

# ------------------------------------------------------------
# 10. Minutes since previous transaction
# ------------------------------------------------------------

df["minutes_since_previous"] = (
    (
        df["timestamp"]
        - df["previous_transaction_time"]
    )
    .dt.total_seconds()
    / 60
)

# First transaction = -1
df["minutes_since_previous"] = (
    df["minutes_since_previous"]
    .fillna(-1)
)

# ------------------------------------------------------------
# 11. Transactions in previous 1 hour
# ------------------------------------------------------------

def transactions_last_1h(group):

    timestamps = group["timestamp"]

    result = []

    for i, current_time in enumerate(timestamps):

        start_time = current_time - pd.Timedelta(
            hours=1
        )

        count = (
            (timestamps.iloc[:i] >= start_time)
            &
            (timestamps.iloc[:i] < current_time)
        ).sum()

        result.append(count)

    return pd.Series(
        result,
        index=group.index
    )


df["transactions_last_1h"] = (
    df.groupby("customer_id", group_keys=False)
    .apply(
        transactions_last_1h,
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)

# ------------------------------------------------------------
# 12. Transactions in previous 24 hours
# ------------------------------------------------------------

def transactions_last_24h(group):

    timestamps = group["timestamp"]

    result = []

    for i, current_time in enumerate(timestamps):

        start_time = current_time - pd.Timedelta(
            hours=24
        )

        count = (
            (timestamps.iloc[:i] >= start_time)
            &
            (timestamps.iloc[:i] < current_time)
        ).sum()

        result.append(count)

    return pd.Series(
        result,
        index=group.index
    )


df["transactions_last_24h_real"] = (
    df.groupby("customer_id", group_keys=False)
    .apply(
        transactions_last_24h,
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)

# ------------------------------------------------------------
# 13. Amount spent in previous 24 hours
# ------------------------------------------------------------

def amount_last_24h(group):

    timestamps = group["timestamp"]
    amounts = group["amount"]

    result = []

    for i, current_time in enumerate(timestamps):

        start_time = current_time - pd.Timedelta(
            hours=24
        )

        mask = (
            (timestamps.iloc[:i] >= start_time)
            &
            (timestamps.iloc[:i] < current_time)
        )

        total = amounts.iloc[:i][mask].sum()

        result.append(total)

    return pd.Series(
        result,
        index=group.index
    )


df["amount_last_24h"] = (
    df.groupby("customer_id", group_keys=False)
    .apply(
        amount_last_24h,
        include_groups=False
    )
    .reset_index(level=0, drop=True)
)

# ------------------------------------------------------------
# 14. Customer's normal transaction hour
# ------------------------------------------------------------

df["usual_transaction_hour"] = (
    df.groupby("customer_id")["transaction_hour"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)

# First transaction fallback
df["usual_transaction_hour"] = (
    df["usual_transaction_hour"]
    .fillna(df["transaction_hour"])
)

# ------------------------------------------------------------
# 15. Difference from usual transaction hour
# ------------------------------------------------------------

df["hour_deviation"] = abs(
    df["transaction_hour"]
    -
    df["usual_transaction_hour"]
)

# ------------------------------------------------------------
# 16. Historical transaction count
# ------------------------------------------------------------

df["previous_transaction_count"] = (
    df.groupby("customer_id")
    .cumcount()
)

# ------------------------------------------------------------
# 17. First transaction indicator
# ------------------------------------------------------------

df["is_first_transaction"] = (
    df["previous_transaction_count"] == 0
).astype(int)

# ------------------------------------------------------------
# 18. Save
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------------------------
# 19. Display results
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("NEW BEHAVIORAL FEATURES")
print("-" * 70)

features = [
    "previous_amount",
    "amount_ratio_to_previous",
    "historical_avg_amount",
    "historical_max_amount",
    "amount_ratio_to_history",
    "minutes_since_previous",
    "transactions_last_1h",
    "transactions_last_24h_real",
    "amount_last_24h",
    "usual_transaction_hour",
    "hour_deviation",
    "previous_transaction_count",
    "is_first_transaction"
]

for feature in features:
    print(f"✓ {feature}")

# ------------------------------------------------------------
# 20. Sample
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("SAMPLE BEHAVIORAL ANALYSIS")
print("-" * 70)

sample_columns = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "previous_amount",
    "historical_avg_amount",
    "amount_ratio_to_history",
    "transactions_last_1h",
    "transactions_last_24h_real",
    "amount_last_24h",
    "fraud"
]

print(
    df[sample_columns]
    .head(15)
    .to_string(index=False)
)

print("\n" + "-" * 70)
print("OUTPUT")
print("-" * 70)

print(
    OUTPUT_PATH.resolve()
)

print("\n" + "=" * 70)
print("BEHAVIORAL FEATURE ENGINE COMPLETE")
print("=" * 70)