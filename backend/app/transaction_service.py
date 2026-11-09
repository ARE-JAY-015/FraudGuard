import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# FRAUDGUARD V3 - TRANSACTION SERVICE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "dataset" / "fraud_transactions_v3_timestamped.csv"
MODEL_PATH = BASE_DIR / "models" / "fraudguard_v3.pkl"


print("=" * 70)
print("FRAUDGUARD V3 - TRANSACTION SERVICE")
print("=" * 70)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("\nLoading transaction database...")

df = pd.read_csv(DATASET_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print(f"Transactions loaded: {len(df):,}")
print(f"Customers loaded: {df['customer_id'].nunique():,}")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading FraudGuard V3 model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# EXISTING TRANSACTION VERIFICATION
# ============================================================

def verify_transaction(transaction_id):
    """
    Verify a transaction that already exists in the dataset.
    """

    matches = df[df["transaction_id"] == transaction_id]

    if matches.empty:
        return {
            "success": False,
            "error": f"Transaction {transaction_id} not found."
        }

    transaction = matches.iloc[0]

    return {
        "success": True,

        "transaction": {
            "transaction_id": int(transaction["transaction_id"]),
            "customer_id": str(transaction["customer_id"]),
            "timestamp": str(transaction["timestamp"]),
            "amount": float(transaction["amount"])
        },

        "prediction": {
            "fraud": bool(transaction["fraud"]),
            "fraud_probability": None,
            "risk_level": (
                "HIGH"
                if int(transaction["fraud"]) == 1
                else "LOW"
            )
        }
    }


# ============================================================
# NEW TRANSACTION PREDICTION
# ============================================================

def predict_new_transaction(transaction_data):
    """
    Predict whether a completely new transaction is fraudulent.

    The transaction does NOT need to exist in the dataset.
    """

    customer_id = transaction_data["customer_id"]

    amount = float(transaction_data["amount"])
    distance_from_home = float(
        transaction_data["distance_from_home"]
    )

    device_changed = int(
        transaction_data["device_changed"]
    )

    international = int(
        transaction_data["international"]
    )

    failed_attempts = int(
        transaction_data["failed_attempts"]
    )

    merchant_category = str(
        transaction_data["merchant_category"]
    )

    payment_method = str(
        transaction_data["payment_method"]
    )

    account_age_days = float(
        transaction_data["account_age_days"]
    )


    # --------------------------------------------------------
    # Find customer history
    # --------------------------------------------------------

    customer_history = df[
        df["customer_id"] == customer_id
    ].copy()

    customer_history = customer_history.sort_values(
        "timestamp"
    )


    # --------------------------------------------------------
    # Historical features
    # --------------------------------------------------------

    if len(customer_history) > 0:

        previous_amount = float(
            customer_history["amount"].iloc[-1]
        )

        historical_avg_amount = float(
            customer_history["amount"].mean()
        )

        historical_max_amount = float(
            customer_history["amount"].max()
        )

        previous_transaction_count = int(
            len(customer_history)
        )

        last_timestamp = customer_history[
            "timestamp"
        ].iloc[-1]

    else:

        previous_amount = amount
        historical_avg_amount = amount
        historical_max_amount = amount
        previous_transaction_count = 0
        last_timestamp = None


    # --------------------------------------------------------
    # Historical ratios
    # --------------------------------------------------------

    if historical_avg_amount > 0:

        amount_ratio_to_history = (
            amount / historical_avg_amount
        )

    else:

        amount_ratio_to_history = 1.0


    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    now = pd.Timestamp.now()

    if last_timestamp is not None:

        minutes_since_previous = (
            now - last_timestamp
        ).total_seconds() / 60

    else:

        minutes_since_previous = -1


    # --------------------------------------------------------
    # Recent transaction counts
    # --------------------------------------------------------

    if last_timestamp is not None:

        one_hour_ago = now - pd.Timedelta(hours=1)

        twenty_four_hours_ago = (
            now - pd.Timedelta(hours=24)
        )

        transactions_last_1h = int(
            (
                customer_history["timestamp"]
                >= one_hour_ago
            ).sum()
        )

        transactions_last_24h_real = int(
            (
                customer_history["timestamp"]
                >= twenty_four_hours_ago
            ).sum()
        )

        amount_last_24h = float(
            customer_history.loc[
                customer_history["timestamp"]
                >= twenty_four_hours_ago,
                "amount"
            ].sum()
        )

    else:

        transactions_last_1h = 0
        transactions_last_24h_real = 0
        amount_last_24h = 0.0


    # --------------------------------------------------------
    # Usual transaction hour
    # --------------------------------------------------------

    if len(customer_history) > 0:

        usual_transaction_hour = float(
            customer_history["timestamp"]
            .dt.hour
            .mean()
        )

    else:

        usual_transaction_hour = float(
            now.hour
        )


    current_hour = now.hour

    hour_deviation = abs(
        current_hour - usual_transaction_hour
    )


    # --------------------------------------------------------
    # First transaction
    # --------------------------------------------------------

    is_first_transaction = (
        1 if previous_transaction_count == 0 else 0
    )


    # --------------------------------------------------------
    # Build model input
    # --------------------------------------------------------

    new_transaction = pd.DataFrame([{

        "amount": amount,

        "distance_from_home":
            distance_from_home,

        "device_changed":
            device_changed,

        "international":
            international,

        "failed_attempts":
            failed_attempts,

        "merchant_category":
            merchant_category,

        "payment_method":
            payment_method,

        "account_age_days":
            account_age_days,

        "previous_amount":
            previous_amount,

        "historical_avg_amount":
            historical_avg_amount,

        "historical_max_amount":
            historical_max_amount,

        "minutes_since_previous":
            minutes_since_previous,

        "transactions_last_1h":
            transactions_last_1h,

        "transactions_last_24h_real":
            transactions_last_24h_real,

        "amount_last_24h":
            amount_last_24h,

        "amount_ratio_to_history":
            amount_ratio_to_history,

        "previous_transaction_count":
            previous_transaction_count,

        "is_first_transaction":
            is_first_transaction,

        "usual_transaction_hour":
            usual_transaction_hour,

        "hour_deviation":
            hour_deviation
    }])


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(new_transaction)[0]
    )

    probabilities = model.predict_proba(
        new_transaction
    )[0]

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


    # --------------------------------------------------------
    # Behavioral analysis
    # --------------------------------------------------------

    reasons = []

    if failed_attempts >= 2:

        reasons.append(
            "Several failed payment attempts detected."
        )

    if device_changed == 1:

        reasons.append(
            "Transaction made from a changed device."
        )

    if international == 1:

        reasons.append(
            "International transaction detected."
        )

    if amount_ratio_to_history >= 2:

        reasons.append(
            "Transaction amount is significantly "
            "higher than historical average."
        )

    if transactions_last_1h >= 3:

        reasons.append(
            "High transaction frequency detected."
        )

    if distance_from_home >= 200:

        reasons.append(
            "Transaction is unusually far from home."
        )

    if not reasons:

        reasons.append(
            "No major behavioral anomaly detected."
        )


    # --------------------------------------------------------
    # Customer history
    # --------------------------------------------------------

    history = []

    if len(customer_history) > 0:

        recent_history = customer_history.tail(10)

        for _, row in recent_history.iterrows():

            history.append({

                "transaction_id":
                    int(row["transaction_id"]),

                "timestamp":
                    str(row["timestamp"]),

                "amount":
                    float(row["amount"]),

                "fraud":
                    int(row["fraud"])
            })


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "success": True,

        "transaction": {

            "customer_id":
                customer_id,

            "amount":
                amount,

            "distance_from_home":
                distance_from_home,

            "device_changed":
                device_changed,

            "international":
                international,

            "failed_attempts":
                failed_attempts,

            "merchant_category":
                merchant_category,

            "payment_method":
                payment_method,

            "account_age_days":
                account_age_days
        },

        "prediction": {

            "fraud":
                bool(prediction),

            "fraud_probability":
                round(
                    fraud_probability,
                    4
                ),

            "risk_level":
                risk_level
        },

        "behavior_analysis": {

            "reasons":
                reasons,

            "amount":
                amount,

            "historical_avg_amount":
                round(
                    historical_avg_amount,
                    2
                ),

            "amount_ratio_to_history":
                round(
                    amount_ratio_to_history,
                    2
                ),

            "transactions_last_1h":
                transactions_last_1h,

            "transactions_last_24h":
                transactions_last_24h_real,

            "failed_attempts":
                failed_attempts,

            "device_changed":
                device_changed,

            "international":
                international,

            "hour_deviation":
                round(
                    hour_deviation,
                    2
                )
        },

        "customer_history":
            history
    }