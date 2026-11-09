import pandas as pd

df = pd.read_csv("../dataset/fraud_transactions.csv")

print("Dataset loaded successfully!")
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------
# 1. Load dataset
# ---------------------------------------

df = pd.read_csv("../dataset/fraud_transactions.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# ---------------------------------------
# 2. Separate features and target
# ---------------------------------------

X = df.drop("fraud", axis=1)
y = df["fraud"]

# transaction_id is just an identifier.
# It shouldn't be used by the model.
X = X.drop("transaction_id", axis=1)

# ---------------------------------------
# 3. Identify categorical columns
# ---------------------------------------

categorical_features = [
    "merchant_category",
    "payment_method"
]

numerical_features = [
    "amount",
    "transaction_hour",
    "distance_from_home",
    "transactions_last_24h",
    "avg_amount",
    "device_changed",
    "international",
    "failed_attempts",
    "account_age_days"
]

# ---------------------------------------
# 4. Preprocessing
# ---------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# ---------------------------------------
# 5. Create ML model
# ---------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

# ---------------------------------------
# 6. Split data
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------------------------------
# 7. Train model
# ---------------------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")

# ---------------------------------------
# 8. Make predictions
# ---------------------------------------

y_pred = pipeline.predict(X_test)

# ---------------------------------------
# 9. Evaluate
# ---------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(pipeline, "fraud_model.pkl")

print("\nModel saved successfully!")