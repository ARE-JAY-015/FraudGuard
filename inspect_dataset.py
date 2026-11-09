import pandas as pd
from pathlib import Path

# Dataset location
dataset_path = Path("dataset/fraud_transactions.csv")

# Check that file exists
if not dataset_path.exists():
    print("ERROR: Dataset not found!")
    print("Expected location:")
    print(dataset_path.resolve())
    exit()

# Load dataset
df = pd.read_csv(dataset_path)

print("\n" + "=" * 60)
print("FRAUDGUARD DATASET INSPECTION")
print("=" * 60)

print("\nDataset location:")
print(dataset_path.resolve())

print("\nDataset shape:")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n" + "-" * 60)
print("COLUMN NAMES")
print("-" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n" + "-" * 60)
print("FIRST 10 ROWS")
print("-" * 60)

print(df.head(10).to_string())

print("\n" + "-" * 60)
print("DATA TYPES")
print("-" * 60)

print(df.dtypes)

print("\n" + "-" * 60)
print("MISSING VALUES")
print("-" * 60)

missing = df.isnull().sum()

for column, count in missing.items():
    print(f"{column}: {count}")

print("\n" + "-" * 60)
print("UNIQUE VALUES")
print("-" * 60)

for column in df.columns:
    print(f"{column}: {df[column].nunique()}")

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)