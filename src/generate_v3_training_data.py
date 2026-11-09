import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FRAUDGUARD V3
# REALISTIC TRANSACTION + FRAUD BURST SIMULATOR
# ============================================================

np.random.seed(42)

OUTPUT_PATH = Path(
    "dataset/fraud_transactions_v3_training.csv"
)

NUMBER_OF_CUSTOMERS = 1000
NUMBER_OF_TRANSACTIONS = 10000

FRAUD_TRANSACTIONS = 500
NORMAL_TRANSACTIONS = (
    NUMBER_OF_TRANSACTIONS - FRAUD_TRANSACTIONS
)

START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-06-30")

MERCHANT_CATEGORIES = [
    "grocery",
    "online_shopping",
    "restaurant",
    "fuel",
    "travel",
    "electronics"
]

PAYMENT_METHODS = [
    "UPI",
    "card",
    "wallet",
    "bank_transfer"
]

# ============================================================
# 1. CUSTOMER PROFILES
# ============================================================

print("=" * 70)
print("FRAUDGUARD V3 - REALISTIC TRANSACTION SIMULATOR")
print("=" * 70)

print("\nCreating customer profiles...")

customers = []

for i in range(1, NUMBER_OF_CUSTOMERS + 1):

    customer_id = f"CUST{str(i).zfill(4)}"

    spending_type = np.random.choice(
        ["low", "medium", "high"],
        p=[0.50, 0.35, 0.15]
    )

    if spending_type == "low":
        normal_amount = np.random.uniform(300, 1200)

    elif spending_type == "medium":
        normal_amount = np.random.uniform(1200, 5000)

    else:
        normal_amount = np.random.uniform(5000, 20000)

    usual_start = np.random.randint(7, 12)
    usual_end = np.random.randint(19, 23)

    preferred_merchant = np.random.choice(
        MERCHANT_CATEGORIES
    )

    preferred_payment = np.random.choice(
        PAYMENT_METHODS
    )

    normal_distance = np.random.uniform(2, 25)

    account_age_days = np.random.randint(
        30,
        3000
    )

    customers.append({
        "customer_id": customer_id,
        "normal_amount": normal_amount,
        "usual_start": usual_start,
        "usual_end": usual_end,
        "preferred_merchant": preferred_merchant,
        "preferred_payment": preferred_payment,
        "normal_distance": normal_distance,
        "account_age_days": account_age_days
    })

customers_df = pd.DataFrame(customers)

print(
    f"Created {len(customers_df)} customer profiles."
)

# ============================================================
# 2. NORMAL TRANSACTIONS
# ============================================================

print("\nGenerating normal transactions...")

normal_transactions = []

transaction_id = 1

for customer in customers:

    customer_id = customer["customer_id"]

    count = np.random.randint(
        5,
        16
    )

    for _ in range(count):

        days = np.random.randint(
            0,
            (END_DATE - START_DATE).days
        )

        date = (
            START_DATE
            + pd.Timedelta(days=int(days))
        )

        hour = np.random.randint(
            customer["usual_start"],
            customer["usual_end"] + 1
        )

        minute = np.random.randint(
            0,
            60
        )

        second = np.random.randint(
            0,
            60
        )

        timestamp = date.replace(
            hour=int(hour),
            minute=int(minute),
            second=int(second)
        )

        amount = np.random.normal(
            customer["normal_amount"],
            customer["normal_amount"] * 0.20
        )

        amount = max(
            round(amount, 2),
            10
        )

        distance = np.random.normal(
            customer["normal_distance"],
            4
        )

        distance = max(
            round(distance, 2),
            0.1
        )

        normal_transactions.append({
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "timestamp": timestamp,
            "amount": amount,
            "distance_from_home": distance,
            "device_changed": 0,
            "international": np.random.choice(
                [0, 1],
                p=[0.96, 0.04]
            ),
            "failed_attempts": np.random.choice(
                [0, 1, 2],
                p=[0.92, 0.06, 0.02]
            ),
            "merchant_category":
                customer["preferred_merchant"],
            "payment_method":
                customer["preferred_payment"],
            "account_age_days":
                customer["account_age_days"],
            "fraud": 0,
            "fraud_scenario": "normal"
        })

        transaction_id += 1

normal_df = pd.DataFrame(
    normal_transactions
)

# Keep exactly 9500 normal transactions
if len(normal_df) > NORMAL_TRANSACTIONS:

    normal_df = (
        normal_df
        .sample(
            NORMAL_TRANSACTIONS,
            random_state=42
        )
        .reset_index(drop=True)
    )

print(
    f"Normal transactions generated: "
    f"{len(normal_df)}"
)

# ============================================================
# 3. FRAUD BURST GENERATOR
# ============================================================

print("\nGenerating fraud transactions...")

fraud_transactions = []

# We create 100 fraud events.
# Each event produces 5 transactions.
# Therefore:
#
# 100 events × 5 transactions = 500 fraud transactions

FRAUD_EVENTS = 100
TRANSACTIONS_PER_EVENT = 5

fraud_customers = np.random.choice(
    customers,
    size=FRAUD_EVENTS,
    replace=False
)

fraud_scenarios = [
    "spending_spike",
    "account_takeover",
    "rapid_transactions",
    "international_anomaly",
    "failed_attempt_attack"
]

fraud_transaction_id = 100000

for event_number, customer in enumerate(
    fraud_customers,
    start=1
):

    scenario = fraud_scenarios[
        (event_number - 1)
        % len(fraud_scenarios)
    ]

    customer_id = customer["customer_id"]

    # --------------------------------------------------------
    # Select a starting point
    # --------------------------------------------------------

    base_day = np.random.randint(
        0,
        (END_DATE - START_DATE).days
    )

    base_date = (
        START_DATE
        + pd.Timedelta(
            days=int(base_day)
        )
    )

    # --------------------------------------------------------
    # Different fraud scenarios
    # --------------------------------------------------------

    if scenario == "spending_spike":

        base_hour = np.random.randint(
            customer["usual_start"],
            customer["usual_end"]
        )

        base_minute = np.random.randint(
            0,
            30
        )

        intervals = [
            0,
            120,
            300,
            600,
            900
        ]

        amount_multiplier = [
            3,
            5,
            8,
            12,
            18
        ]

    elif scenario == "account_takeover":

        base_hour = np.random.choice(
            [1, 2, 3, 4, 5]
        )

        base_minute = np.random.randint(
            0,
            30
        )

        intervals = [
            0,
            90,
            180,
            300,
            450
        ]

        amount_multiplier = [
            2,
            3,
            5,
            7,
            10
        ]

    elif scenario == "rapid_transactions":

        base_hour = np.random.randint(
            customer["usual_start"],
            customer["usual_end"]
        )

        base_minute = np.random.randint(
            0,
            55
        )

        # VERY short intervals
        intervals = [
            0,
            45,
            90,
            150,
            240
        ]

        amount_multiplier = [
            1.5,
            2,
            2.5,
            4,
            6
        ]

    elif scenario == "international_anomaly":

        base_hour = np.random.randint(
            customer["usual_start"],
            customer["usual_end"]
        )

        base_minute = np.random.randint(
            0,
            30
        )

        intervals = [
            0,
            180,
            420,
            900,
            1500
        ]

        amount_multiplier = [
            2,
            2.5,
            3,
            4,
            5
        ]

    else:

        # failed_attempt_attack

        base_hour = np.random.choice(
            [0, 1, 2, 3, 4, 5]
        )

        base_minute = np.random.randint(
            0,
            30
        )

        intervals = [
            0,
            60,
            120,
            180,
            240
        ]

        amount_multiplier = [
            2,
            3,
            4,
            6,
            8
        ]

    # --------------------------------------------------------
    # Generate the five transactions
    # --------------------------------------------------------

    for j in range(
        TRANSACTIONS_PER_EVENT
    ):

        timestamp = (
            base_date
            .replace(
                hour=int(base_hour),
                minute=int(base_minute),
                second=int(
                    np.random.randint(
                        0,
                        60
                    )
                )
            )
            +
            pd.Timedelta(
                seconds=int(
                    intervals[j]
                )
            )
        )

        amount = (
            customer["normal_amount"]
            *
            amount_multiplier[j]
            *
            np.random.uniform(
                0.90,
                1.10
            )
        )

        # ----------------------------------------------------
        # Default fraud behavior
        # ----------------------------------------------------

        distance = (
            customer["normal_distance"]
            *
            np.random.uniform(
                4,
                12
            )
        )

        device_changed = 1
        international = 1

        failed_attempts = np.random.randint(
            1,
            6
        )

        merchant = np.random.choice(
            MERCHANT_CATEGORIES
        )

        payment = np.random.choice(
            PAYMENT_METHODS
        )

        # ----------------------------------------------------
        # Scenario-specific behavior
        # ----------------------------------------------------

        if scenario == "spending_spike":

            device_changed = 0
            international = 0
            failed_attempts = 0

            distance = customer[
                "normal_distance"
            ]

        elif scenario == "account_takeover":

            device_changed = 1
            international = 1

            failed_attempts = np.random.randint(
                2,
                6
            )

        elif scenario == "rapid_transactions":

            device_changed = 0
            international = 0

            failed_attempts = np.random.randint(
                0,
                3
            )

            distance = customer[
                "normal_distance"
            ]

        elif scenario == "international_anomaly":

            device_changed = 0
            international = 1

            failed_attempts = np.random.randint(
                0,
                3
            )

        elif scenario == "failed_attempt_attack":

            device_changed = 1

            failed_attempts = np.random.randint(
                3,
                8
            )

        fraud_transactions.append({
            "transaction_id":
                fraud_transaction_id,

            "customer_id":
                customer_id,

            "timestamp":
                timestamp,

            "amount":
                round(
                    amount,
                    2
                ),

            "distance_from_home":
                round(
                    distance,
                    2
                ),

            "device_changed":
                device_changed,

            "international":
                international,

            "failed_attempts":
                failed_attempts,

            "merchant_category":
                merchant,

            "payment_method":
                payment,

            "account_age_days":
                customer[
                    "account_age_days"
                ],

            "fraud": 1,

            "fraud_scenario":
                scenario
        })

        fraud_transaction_id += 1

print(
    f"Fraud transactions generated: "
    f"{len(fraud_transactions)}"
)

fraud_df = pd.DataFrame(
    fraud_transactions
)

# ============================================================
# 4. COMBINE
# ============================================================

df = pd.concat(
    [
        normal_df,
        fraud_df
    ],
    ignore_index=True
)

# ============================================================
# 5. SORT CHRONOLOGICALLY
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    [
        "customer_id",
        "timestamp"
    ]
).reset_index(drop=True)

# ============================================================
# 6. HISTORICAL FEATURES
# ============================================================

print("\nCalculating historical features...")

df["previous_amount"] = (
    df.groupby(
        "customer_id"
    )["amount"]
    .shift(1)
)

df["historical_avg_amount"] = (
    df.groupby(
        "customer_id"
    )["amount"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)

df["historical_max_amount"] = (
    df.groupby(
        "customer_id"
    )["amount"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .max()
    )
)

df["previous_transaction_time"] = (
    df.groupby(
        "customer_id"
    )["timestamp"]
    .shift(1)
)

df["minutes_since_previous"] = (
    (
        df["timestamp"]
        -
        df["previous_transaction_time"]
    )
    .dt.total_seconds()
    / 60
)

# ============================================================
# 7. VELOCITY FEATURES
# ============================================================

def calculate_velocity(group):

    times = group[
        "timestamp"
    ].tolist()

    results = []

    for i, current_time in enumerate(
        times
    ):

        count = 0

        for previous_time in times[:i]:

            difference = (
                current_time
                -
                previous_time
            ).total_seconds()

            if difference <= 3600:

                count += 1

        results.append(count)

    return pd.Series(
        results,
        index=group.index
    )


df["transactions_last_1h"] = (
    df.groupby(
        "customer_id",
        group_keys=False
    )
    .apply(
        calculate_velocity,
        include_groups=False
    )
    .reset_index(
        level=0,
        drop=True
    )
)

# ============================================================
# 8. 24-HOUR VELOCITY
# ============================================================

def calculate_24h(group):

    times = group[
        "timestamp"
    ].tolist()

    results = []

    for i, current_time in enumerate(
        times
    ):

        count = 0

        for previous_time in times[:i]:

            difference = (
                current_time
                -
                previous_time
            ).total_seconds()

            if difference <= 86400:

                count += 1

        results.append(count)

    return pd.Series(
        results,
        index=group.index
    )


df["transactions_last_24h_real"] = (
    df.groupby(
        "customer_id",
        group_keys=False
    )
    .apply(
        calculate_24h,
        include_groups=False
    )
    .reset_index(
        level=0,
        drop=True
    )
)

# ============================================================
# 9. AMOUNT SPENT IN PREVIOUS 24 HOURS
# ============================================================

def calculate_amount_24h(group):

    times = group[
        "timestamp"
    ].tolist()

    amounts = group[
        "amount"
    ].tolist()

    results = []

    for i, current_time in enumerate(
        times
    ):

        total = 0

        for k in range(i):

            difference = (
                current_time
                -
                times[k]
            ).total_seconds()

            if difference <= 86400:

                total += amounts[k]

        results.append(total)

    return pd.Series(
        results,
        index=group.index
    )


df["amount_last_24h"] = (
    df.groupby(
        "customer_id",
        group_keys=False
    )
    .apply(
        calculate_amount_24h,
        include_groups=False
    )
    .reset_index(
        level=0,
        drop=True
    )
)

# ============================================================
# 10. HISTORICAL RATIOS
# ============================================================

df["amount_ratio_to_history"] = (
    df["amount"]
    /
    df["historical_avg_amount"]
)

df["amount_ratio_to_history"] = (
    df["amount_ratio_to_history"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

# ============================================================
# 11. TRANSACTION COUNT
# ============================================================

df["previous_transaction_count"] = (
    df.groupby(
        "customer_id"
    ).cumcount()
)

df["is_first_transaction"] = (
    df[
        "previous_transaction_count"
    ] == 0
).astype(int)

# ============================================================
# 12. USUAL TRANSACTION HOUR
# ============================================================

df["usual_transaction_hour"] = (
    df.groupby(
        "customer_id"
    )["timestamp"]
    .transform(
        lambda x:
        x.shift(1)
        .dt.hour
        .expanding()
        .mean()
    )
)

df["usual_transaction_hour"] = (
    df[
        "usual_transaction_hour"
    ].fillna(
        df["timestamp"].dt.hour
    )
)

df["hour_deviation"] = abs(
    df["timestamp"].dt.hour
    -
    df["usual_transaction_hour"]
)

# ============================================================
# 13. CLEAN FIRST TRANSACTIONS
# ============================================================

df["previous_amount"] = (
    df["previous_amount"]
    .fillna(df["amount"])
)

df["historical_avg_amount"] = (
    df["historical_avg_amount"]
    .fillna(df["amount"])
)

df["historical_max_amount"] = (
    df["historical_max_amount"]
    .fillna(df["amount"])
)

df["amount_ratio_to_history"] = (
    df["amount_ratio_to_history"]
    .fillna(1.0)
)

df["minutes_since_previous"] = (
    df["minutes_since_previous"]
    .fillna(-1)
)

# ============================================================
# 14. FINAL DATASET
# ============================================================

# Sort globally by timestamp
df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# ============================================================
# 15. SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ============================================================
# 16. REPORT
# ============================================================

print("\n" + "=" * 70)
print("V3 DATASET CREATED")
print("=" * 70)

print(
    f"\nTotal transactions: "
    f"{len(df)}"
)

print(
    f"Customers: "
    f"{df['customer_id'].nunique()}"
)

print(
    f"Legitimate transactions: "
    f"{(df['fraud'] == 0).sum()}"
)

print(
    f"Fraudulent transactions: "
    f"{(df['fraud'] == 1).sum()}"
)

print(
    f"Fraud rate: "
    f"{df['fraud'].mean() * 100:.2f}%"
)

print("\nFraud scenarios:")

print(
    df[df["fraud"] == 1]
    ["fraud_scenario"]
    .value_counts()
)

print("\nVelocity statistics:")

print(
    "Maximum transactions in 1 hour:",
    int(
        df[
            "transactions_last_1h"
        ].max()
    )
)

print(
    "Maximum transactions in 24 hours:",
    int(
        df[
            "transactions_last_24h_real"
        ].max()
    )
)

print(
    "Maximum amount in 24 hours:",
    round(
        df[
            "amount_last_24h"
        ].max(),
        2
    )
)

# ============================================================
# 17. FRAUD BURST EXAMPLE
# ============================================================

print("\n" + "-" * 70)
print("EXAMPLE FRAUD BURST")
print("-" * 70)

# Find a customer who actually has
# a rapid fraud burst.

fraud_example_rows = df[
    (df["fraud"] == 1) &
    (df["fraud_scenario"] == "rapid_transactions")
]

if len(fraud_example_rows) > 0:

    fraud_customer = (
        fraud_example_rows
        ["customer_id"]
        .iloc[0]
    )

    example = (
        df[
            df["customer_id"]
            == fraud_customer
        ]
        .sort_values("timestamp")
    )

    print(
        f"\nShowing fraud history for: "
        f"{fraud_customer}"
    )

    print(
        example[
            [
                "transaction_id",
                "customer_id",
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
        .tail(15)
        .to_string(index=False)
    )

else:

    print(
        "Could not find a rapid fraud example."
    )

print(
    example[
        [
            "transaction_id",
            "customer_id",
            "timestamp",
            "amount",
            "minutes_since_previous",
            "transactions_last_1h",
            "amount_ratio_to_history",
            "fraud",
            "fraud_scenario"
        ]
    ]
    .tail(10)
    .to_string(index=False)
)

print("\nSaved to:")

print(
    OUTPUT_PATH.resolve()
)

print("\n" + "=" * 70)
print("FRAUDGUARD V3 SIMULATOR COMPLETE")
print("=" * 70)