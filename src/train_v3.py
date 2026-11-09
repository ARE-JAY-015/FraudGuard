import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

import joblib


# ============================================================
# FRAUDGUARD V3 - MODEL TRAINING
# ============================================================

DATASET_PATH = Path(
    "dataset/fraud_transactions_v3_training.csv"
)

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("FRAUDGUARD V3 - MODEL TRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# 2. CONVERT TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


# ============================================================
# 3. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


print(
    f"First transaction: "
    f"{df['timestamp'].min()}"
)

print(
    f"Last transaction: "
    f"{df['timestamp'].max()}"
)


# ============================================================
# 4. TARGET
# ============================================================

TARGET = "fraud"

y = df[TARGET].astype(int)


# ============================================================
# 5. REMOVE INFORMATION THAT SHOULD NOT BE USED
# ============================================================

# These columns are identifiers, labels, or information
# that would not be available as an input feature.

columns_to_remove = [
    "fraud",
    "fraud_scenario",

    "transaction_id",

    "timestamp",

    "customer_id",

    "previous_transaction_time"
]


X = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# ============================================================
# 6. REMOVE OLD / DUPLICATE FEATURES
# ============================================================

# transaction_hour and transactions_last_24h are not
# present in the current V3 simulator, so we do not use them.

print("\nFeatures selected:")

for column in X.columns:
    print(f"  - {column}")


# ============================================================
# 7. IDENTIFY FEATURE TYPES
# ============================================================

# ============================================================
# 7. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = [
    "merchant_category",
    "payment_method"
]

numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


print(
    f"\nNumerical features: "
    f"{len(numerical_features)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_features)}"
)

print("\nCategorical columns:")

for column in categorical_features:
    print(f"  - {column}")
    


print(
    f"\nNumerical features: "
    f"{len(numerical_features)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_features)}"
)


# ============================================================
# 8. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 9. TEMPORAL TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# We do NOT randomly shuffle transactions.
#
# Earlier transactions -> training
# Later transactions  -> testing
#
# This better represents real-world fraud detection.

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print("\nTemporal split:")

print(
    f"Training rows: "
    f"{len(X_train)}"
)

print(
    f"Testing rows : "
    f"{len(X_test)}"
)

print(
    f"\nTraining period:"
)

print(
    df["timestamp"].iloc[0]
)

print(
    "to"
)

print(
    df["timestamp"].iloc[
        split_index - 1
    ]
)

print(
    f"\nTesting period:"
)

print(
    df["timestamp"].iloc[
        split_index
    ]
)

print(
    "to"
)

print(
    df["timestamp"].iloc[-1]
)


# ============================================================
# 10. FRAUD DISTRIBUTION
# ============================================================

print("\nFraud distribution:")

print(
    "Training legitimate:",
    int(
        (y_train == 0).sum()
    )
)

print(
    "Training fraud:",
    int(
        (y_train == 1).sum()
    )
)

print(
    "Testing legitimate:",
    int(
        (y_test == 0).sum()
    )
)

print(
    "Testing fraud:",
    int(
        (y_test == 1).sum()
    )
)


# ============================================================
# 11. RANDOM FOREST
# ============================================================

print("\nCreating Random Forest...")

model = RandomForestClassifier(
    n_estimators=400,

    max_depth=15,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 12. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 13. TRAIN
# ============================================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# ============================================================
# 14. PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 17. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print(
    f"\nROC-AUC: "
    f"{roc_auc:.4f}"
)


# ============================================================
# 18. PRECISION-RECALL AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print(
    f"PR-AUC: "
    f"{pr_auc:.4f}"
)


# ============================================================
# 19. SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "fraudguard_v3.pkl"
)

joblib.dump(
    pipeline,
    model_path
)


print(
    f"\nModel saved to:"
)

print(
    model_path.resolve()
)


# ============================================================
# 20. SAVE TEST RESULTS
# ============================================================

results = df.iloc[
    split_index:
].copy()

results["predicted_fraud"] = y_pred

results["fraud_probability"] = (
    y_probability
)

results_path = Path(
    "outputs"
)

results_path.mkdir(
    exist_ok=True
)

results_file = (
    results_path
    / "v3_test_predictions.csv"
)

results.to_csv(
    results_file,
    index=False
)


print(
    f"\nTest predictions saved to:"
)

print(
    results_file.resolve()
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FRAUDGUARD V3 TRAINING COMPLETED")
print("=" * 70)