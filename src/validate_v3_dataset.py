import pandas as pd
from pathlib import Path


# ============================================================
# FRAUDGUARD V3 - DATASET VALIDATION
# ============================================================

DATASET_PATH = Path(
    "dataset/fraud_transactions_v3_training.csv"
)

print("=" * 70)
print("FRAUDGUARD V3 - DATASET VALIDATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

if not DATASET_PATH.exists():
    print("\nERROR: V3 training dataset not found!")
    print(DATASET_PATH.resolve())
    raise SystemExit(1)

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",

    "distance_from_home",
    "device_changed",
    "international",
    "failed_attempts",
    "account_age_days",

    "merchant_category",
    "payment_method",

    "fraud",

    "previous_transaction_time",
    "previous_amount",

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
    "is_first_transaction",

    "fraud_scenario"
]

print("\nChecking required columns...")

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("MISSING COLUMNS:")

    for column in missing_columns:
        print(f"  - {column}")
else:
    print("All required columns are present.")


# ============================================================
# 3. TRANSACTION ID CHECK
# ============================================================

print("\nTransaction ID check:")

duplicate_ids = (
    df["transaction_id"]
    .duplicated()
    .sum()
)

print(f"Duplicate transaction IDs: {duplicate_ids}")


# ============================================================
# 4. CUSTOMER INFORMATION
# ============================================================

print("\nCustomer information:")

customer_count = df["customer_id"].nunique()

customer_counts = (
    df["customer_id"]
    .value_counts()
)

print(f"Unique customers: {customer_count}")

print(
    f"Average transactions/customer: "
    f"{customer_counts.mean():.2f}"
)

print(
    f"Minimum transactions/customer: "
    f"{customer_counts.min()}"
)

print(
    f"Maximum transactions/customer: "
    f"{customer_counts.max()}"
)


# ============================================================
# 5. TIMESTAMP CHECK
# ============================================================

print("\nTimestamp check:")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamps = (
    df["timestamp"]
    .isna()
    .sum()
)

print(f"Invalid timestamps: {invalid_timestamps}")

print(
    f"First timestamp: "
    f"{df['timestamp'].min()}"
)

print(
    f"Last timestamp: "
    f"{df['timestamp'].max()}"
)


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\nMissing values:")

missing = df.isnull().sum()

total_missing = missing.sum()

print(f"Total missing values: {total_missing}")

if total_missing > 0:

    print("\nColumns containing missing values:")

    for column, count in missing.items():

        if count > 0:
            print(f"{column}: {count}")


# ============================================================
# 7. PREVIOUS TRANSACTION VALIDATION
# ============================================================

print("\nPrevious transaction validation:")

expected_missing_previous_time = (
    df["is_first_transaction"] == 1
).sum()

actual_missing_previous_time = (
    df["previous_transaction_time"]
    .isna()
    .sum()
)

print(
    f"Expected missing previous times: "
    f"{expected_missing_previous_time}"
)

print(
    f"Actual missing previous times: "
    f"{actual_missing_previous_time}"
)

previous_time_check = (
    actual_missing_previous_time
    == expected_missing_previous_time
)

if previous_time_check:

    print(
        "PASSED - Missing previous times "
        "occur only for first transactions."
    )

else:

    print(
        "FAILED - Unexpected missing "
        "previous transaction times."
    )


# ============================================================
# 8. FRAUD DISTRIBUTION
# ============================================================

print("\nFraud distribution:")

fraud_count = int(
    df["fraud"].sum()
)

legitimate_count = (
    len(df) - fraud_count
)

fraud_percentage = (
    fraud_count / len(df) * 100
)

print(
    f"Legitimate transactions: "
    f"{legitimate_count}"
)

print(
    f"Fraudulent transactions: "
    f"{fraud_count}"
)

print(
    f"Fraud percentage: "
    f"{fraud_percentage:.2f}%"
)


# ============================================================
# 9. FRAUD SCENARIOS
# ============================================================

print("\nFraud scenario distribution:")

print(
    df[
        df["fraud"] == 1
    ]["fraud_scenario"]
    .value_counts()
)


# ============================================================
# 10. AMOUNT VALIDATION
# ============================================================

print("\nAmount validation:")

print(
    f"Minimum amount: "
    f"{df['amount'].min():.2f}"
)

print(
    f"Maximum amount: "
    f"{df['amount'].max():.2f}"
)

print(
    f"Average amount: "
    f"{df['amount'].mean():.2f}"
)

negative_amounts = (
    df["amount"] < 0
).sum()

print(
    f"Negative amounts: "
    f"{negative_amounts}"
)


# ============================================================
# 11. VELOCITY VALIDATION
# ============================================================

print("\nVelocity validation:")

print(
    "Maximum transactions in 1 hour:",
    int(
        df["transactions_last_1h"].max()
    )
)

print(
    "Maximum transactions in 24 hours:",
    int(
        df["transactions_last_24h_real"].max()
    )
)

print(
    "Maximum amount in 24 hours:",
    round(
        df["amount_last_24h"].max(),
        2
    )
)


# ============================================================
# 12. RAPID FRAUD VALIDATION
# ============================================================

print("\nRapid fraud validation:")

rapid_fraud = df[
    (
        df["fraud"] == 1
    )
    &
    (
        df["fraud_scenario"]
        == "rapid_transactions"
    )
]

print(
    f"Rapid fraud transactions: "
    f"{len(rapid_fraud)}"
)

if len(rapid_fraud) > 0:

    print(
        "Maximum transactions in 1 hour "
        "during rapid fraud:",
        int(
            rapid_fraud[
                "transactions_last_1h"
            ].max()
        )
    )

    valid_minutes = (
        rapid_fraud[
            "minutes_since_previous"
        ]
        .replace(-1, pd.NA)
        .dropna()
    )

    if len(valid_minutes) > 0:

        print(
            "Minimum time between "
            "rapid-fraud transactions:",
            round(
                valid_minutes.min(),
                2
            ),
            "minutes"
        )


# ============================================================
# 13. FIRST TRANSACTION CHECK
# ============================================================

print("\nFirst transaction validation:")

first_transactions = df[
    df["is_first_transaction"] == 1
]

print(
    f"Transactions marked as first: "
    f"{len(first_transactions)}"
)

print(
    f"Unique customers: "
    f"{customer_count}"
)


# ============================================================
# 14. CUSTOMER TIMESTAMP ORDER
# ============================================================

print("\nCustomer timestamp validation:")

ordered = (
    df.sort_values(
        [
            "customer_id",
            "timestamp"
        ]
    )
)

timestamp_errors = 0

for customer_id, group in ordered.groupby(
    "customer_id"
):

    if not group[
        "timestamp"
    ].is_monotonic_increasing:

        timestamp_errors += 1

print(
    "Customers with timestamp "
    f"ordering errors: {timestamp_errors}"
)


# ============================================================
# 15. RAPID FRAUD EXAMPLE
# ============================================================

print("\n" + "-" * 70)
print("RAPID FRAUD EXAMPLE")
print("-" * 70)

if len(rapid_fraud) > 0:

    example_customer = (
        rapid_fraud[
            "customer_id"
        ].iloc[0]
    )

    example = (
        df[
            df["customer_id"]
            == example_customer
        ]
        .sort_values("timestamp")
    )

    print(
        f"\nCustomer: {example_customer}\n"
    )

    print(
        example[
            [
                "transaction_id",
                "timestamp",
                "amount",
                "minutes_since_previous",
                "transactions_last_1h",
                "transactions_last_24h_real",
                "amount_ratio_to_history",
                "fraud",
                "fraud_scenario"
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)

validation_passed = (
    len(missing_columns) == 0
    and duplicate_ids == 0
    and invalid_timestamps == 0
    and negative_amounts == 0
    and previous_time_check
    and timestamp_errors == 0
)

if validation_passed:

    print("V3 DATASET VALIDATION PASSED")

else:

    print("V3 DATASET VALIDATION FOUND ISSUES")

print("=" * 70)