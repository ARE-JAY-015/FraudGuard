import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# FRAUDGUARD V3 - TRANSACTION ANALYZER
# ============================================================

print("=" * 70)
print("FRAUDGUARD V3 - TRANSACTION INTELLIGENCE ENGINE")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "fraud_transactions_v3_training.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraudguard_v3.pkl"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading transaction database...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print(f"Dataset loaded successfully.")
print(f"Transactions: {len(df):,}")
print(f"Customers: {df['customer_id'].nunique():,}")


# ============================================================
# PREPARE TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


# ============================================================
# SORT TRANSACTIONS
# ============================================================

df = df.sort_values(
    ["customer_id", "timestamp"]
).reset_index(drop=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading FraudGuard V3 model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# FEATURES USED BY V3 MODEL
# ============================================================

MODEL_FEATURES = [
    "amount",
    "distance_from_home",
    "device_changed",
    "international",
    "failed_attempts",
    "merchant_category",
    "payment_method",
    "account_age_days",
    "previous_amount",
    "historical_avg_amount",
    "historical_max_amount",
    "minutes_since_previous",
    "transactions_last_1h",
    "transactions_last_24h_real",
    "amount_last_24h",
    "amount_ratio_to_history",
    "previous_transaction_count",
    "is_first_transaction",
    "usual_transaction_hour",
    "hour_deviation",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _safe_float(value, default=0.0):
    """
    Convert a value safely to float.
    """

    try:

        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


def _safe_int(value, default=0):
    """
    Convert a value safely to integer.
    """

    try:

        if pd.isna(value):
            return default

        return int(value)

    except (TypeError, ValueError):

        return default


# ============================================================
# BEHAVIOR ANALYSIS
# ============================================================

def analyze_behavior(transaction):
    """
    Analyze the behavioral characteristics of a transaction.
    """

    reasons = []

    amount = _safe_float(transaction.get("amount"))

    historical_avg = _safe_float(
        transaction.get("historical_avg_amount"),
        amount
    )

    amount_ratio = _safe_float(
        transaction.get("amount_ratio_to_history"),
        1.0
    )

    transactions_1h = _safe_int(
        transaction.get("transactions_last_1h")
    )

    transactions_24h = _safe_int(
        transaction.get("transactions_last_24h_real")
    )

    failed_attempts = _safe_int(
        transaction.get("failed_attempts")
    )

    device_changed = _safe_int(
        transaction.get("device_changed")
    )

    international = _safe_int(
        transaction.get("international")
    )

    hour_deviation = _safe_float(
        transaction.get("hour_deviation")
    )

    # --------------------------------------------------------
    # Amount anomaly
    # --------------------------------------------------------

    if amount_ratio >= 3:
        reasons.append(
            "Transaction amount is more than 3x the customer's historical average."
        )

    elif amount_ratio >= 2:
        reasons.append(
            "Transaction amount is significantly higher than the customer's normal spending."
        )

    # --------------------------------------------------------
    # Rapid transactions
    # --------------------------------------------------------

    if transactions_1h >= 4:

        reasons.append(
            "Multiple transactions detected within a short time period."
        )

    elif transactions_1h >= 2:

        reasons.append(
            "Unusually high transaction velocity detected."
        )

    # --------------------------------------------------------
    # 24 hour activity
    # --------------------------------------------------------

    if transactions_24h >= 5:

        reasons.append(
            "High number of transactions detected within 24 hours."
        )

    # --------------------------------------------------------
    # Failed attempts
    # --------------------------------------------------------

    if failed_attempts >= 3:

        reasons.append(
            "Multiple failed payment attempts detected."
        )

    elif failed_attempts >= 2:

        reasons.append(
            "Several failed payment attempts detected."
        )

    # --------------------------------------------------------
    # Device change
    # --------------------------------------------------------

    if device_changed == 1:

        reasons.append(
            "Transaction was performed after a device change."
        )

    # --------------------------------------------------------
    # International transaction
    # --------------------------------------------------------

    if international == 1:

        reasons.append(
            "International transaction detected."
        )

    # --------------------------------------------------------
    # Unusual transaction time
    # --------------------------------------------------------

    if hour_deviation >= 6:

        reasons.append(
            "Transaction occurred at an unusual time for this customer."
        )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    if not reasons:

        reasons.append(
            "No major behavioral anomaly detected."
        )

    return {
        "reasons": reasons,

        "amount": round(amount, 2),

        "historical_avg_amount": round(
            historical_avg,
            2
        ),

        "amount_ratio_to_history": round(
            amount_ratio,
            2
        ),

        "transactions_last_1h": transactions_1h,

        "transactions_last_24h": transactions_24h,

        "failed_attempts": failed_attempts,

        "device_changed": device_changed,

        "international": international,

        "hour_deviation": round(
            hour_deviation,
            2
        ),
    }


# ============================================================
# CUSTOMER HISTORY
# ============================================================

def get_customer_history(customer_id, transaction_id=None):
    """
    Return previous transactions belonging to a customer.
    """

    history = df[
        df["customer_id"] == customer_id
    ].copy()

    history = history.sort_values(
        "timestamp"
    )

    # Remove current transaction when requested
    if transaction_id is not None:

        history = history[
            history["transaction_id"] != transaction_id
        ]

    history = history.sort_values(
        "timestamp",
        ascending=False
    )

    result = []

    for _, row in history.head(10).iterrows():

        result.append({
            "transaction_id": _safe_int(
                row["transaction_id"]
            ),

            "timestamp": str(
                row["timestamp"]
            ),

            "amount": round(
                _safe_float(row["amount"]),
                2
            ),

            "fraud": _safe_int(
                row["fraud"]
            )
        })

    return result


# ============================================================
# FIND TRANSACTION
# ============================================================

def find_transaction(transaction_id):
    """
    Find a transaction by transaction ID.
    """

    matches = df[
        df["transaction_id"] == transaction_id
    ]

    if matches.empty:

        return None

    return matches.iloc[0]


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_transaction(transaction):
    """
    Run FraudGuard V3 ML model prediction.
    """

    # Convert transaction to dataframe
    input_data = pd.DataFrame(
        [transaction]
    )

    # Make sure every model feature exists
    for feature in MODEL_FEATURES:

        if feature not in input_data.columns:

            input_data[feature] = 0

    # Keep only model features
    input_data = input_data[
        MODEL_FEATURES
    ]

    # Prediction
    prediction = model.predict(
        input_data
    )[0]

    # Probability
    probabilities = model.predict_proba(
        input_data
    )[0]

    # Fraud probability
    fraud_probability = float(
        probabilities[1]
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if fraud_probability >= 0.80:

        risk_level = "HIGH"

    elif fraud_probability >= 0.50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    return {
        "fraud": bool(prediction),

        "fraud_probability": round(
            fraud_probability,
            4
        ),

        "risk_level": risk_level
    }


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_transaction(transaction_id):
    """
    Complete transaction analysis.

    This function:
    1. Finds the transaction
    2. Runs ML prediction
    3. Analyzes customer behavior
    4. Retrieves customer history
    5. Returns complete result
    """

    # --------------------------------------------------------
    # Find transaction
    # --------------------------------------------------------

    transaction = find_transaction(
        transaction_id
    )

    if transaction is None:

        return {
            "success": False,

            "error": (
                f"Transaction ID "
                f"{transaction_id} was not found."
            )
        }

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = predict_transaction(
        transaction
    )

    # --------------------------------------------------------
    # Behavioral analysis
    # --------------------------------------------------------

    behavior = analyze_behavior(
        transaction
    )

    # --------------------------------------------------------
    # Customer history
    # --------------------------------------------------------

    customer_history = get_customer_history(
        transaction["customer_id"],
        transaction_id
    )

    # --------------------------------------------------------
    # Transaction information
    # --------------------------------------------------------

    transaction_info = {

        "transaction_id": _safe_int(
            transaction["transaction_id"]
        ),

        "customer_id": str(
            transaction["customer_id"]
        ),

        "timestamp": str(
            transaction["timestamp"]
        ),

        "amount": round(
            _safe_float(
                transaction["amount"]
            ),
            2
        )
    }

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        "success": True,

        "transaction": transaction_info,

        "prediction": prediction,

        "behavior_analysis": behavior,

        "customer_history": customer_history
    }


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    print("\nAnalyzer loaded successfully.")

    print(
        "\nAvailable function:"
    )

    print(
        "analyze_transaction(transaction_id)"
    )

    print(
        "\nExample:"
    )

    print(
        "analyze_transaction(100436)"
    )