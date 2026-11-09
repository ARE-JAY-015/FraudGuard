import csv
import math
from collections import defaultdict

print("=" * 65)
print("FRAUDGUARD: PURE PYTHON RISK & ANOMALY DETECTION ENGINE")
print("=" * 65)

# 1. Load Dataset using standard library (No external DLLs)
dataset_path = "dataset/fraud_transactions.csv"
rows = []

with open(dataset_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        # Cast numeric fields
        r["amount"] = float(r["amount"])
        r["avg_amount"] = float(r["avg_amount"])
        r["distance_from_home"] = float(r["distance_from_home"])
        r["transactions_last_24h"] = int(r["transactions_last_24h"])
        r["device_changed"] = int(r["device_changed"])
        r["international"] = int(r["international"])
        r["failed_attempts"] = int(r["failed_attempts"])
        r["account_age_days"] = int(r["account_age_days"])
        r["fraud"] = int(r["fraud"])
        rows.append(r)

total_tx = len(rows)
fraud_tx = sum(1 for r in rows if r["fraud"] == 1)
legit_tx = total_tx - fraud_tx
total_volume = sum(r["amount"] for r in rows)
fraud_volume = sum(r["amount"] for r in rows if r["fraud"] == 1)

print(f"\n1. DATASET OVERVIEW:")
print(f"   Total Transactions : {total_tx:,}")
print(f"   Legitimate (0)     : {legit_tx:,} ({legit_tx/total_tx*100:.2f}%)")
print(f"   Fraudulent (1)     : {fraud_tx:,} ({fraud_tx/total_tx*100:.2f}%)")
print(f"   Gross Volume       : ${total_volume:,.2f}")
print(f"   Total Fraud Loss   : ${fraud_volume:,.2f}")

# 2. Risk Scoring Algorithm
# Derived from banking rule engines (Stripe Radar / Visa Risk Rules)
tp = fp = fn = tn = 0
prevented_dollars = 0.0

for r in rows:
    score = 0
    ratio = r["amount"] / (r["avg_amount"] + 1e-5)
    
    # Rule 1: Surge spending relative to customer baseline
    if ratio >= 4.0:
        score += 35
    elif ratio >= 2.5:
        score += 20
        
    # Rule 2: Authentication failures
    if r["failed_attempts"] >= 3:
        score += 30
    elif r["failed_attempts"] >= 1:
        score += 15
        
    # Rule 3: Hardware fingerprint mismatch
    if r["device_changed"] == 1:
        score += 20
        
    # Rule 4: Geovelocity / Cross-border
    if r["international"] == 1:
        score += 15
    if r["distance_from_home"] > 50:
        score += 15
        
    # Rule 5: High velocity burst
    if r["transactions_last_24h"] >= 8:
        score += 20

    r["risk_score"] = min(score, 100)
    
    # Decision Threshold: >= 65 is flagged as FRAUD
    pred = 1 if r["risk_score"] >= 65 else 0
    r["pred"] = pred
    
    # Confusion Matrix tallies
    if pred == 1 and r["fraud"] == 1:
        tp += 1
        prevented_dollars += r["amount"]
    elif pred == 1 and r["fraud"] == 0:
        fp += 1
    elif pred == 0 and r["fraud"] == 1:
        fn += 1
    else:
        tn += 1

# 3. Model & Business Metrics
precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
fpr = (fp / legit_tx * 100) if legit_tx > 0 else 0

print("\n" + "=" * 65)
print("2. RISK ENGINE EVALUATION (CONFUSION MATRIX & METRICS)")
print("=" * 65)
print(f"   True Positives  (Fraud Prevented)    : {tp:,}")
print(f"   False Positives (Legit Blocked)      : {fp:,}")
print(f"   False Negatives (Fraud Missed)       : {fn:,}")
print(f"   True Negatives  (Legit Approved)     : {tn:,}")
print("-" * 65)
print(f"   Detection Rate (Recall)              : {recall:.2f}%")
print(f"   Alert Precision                      : {precision:.2f}%")
print(f"   False Positive Rate (FPR)            : {fpr:.2f}%")
print(f"   Capital Saved                        : ${prevented_dollars:,.2f} ({prevented_dollars/fraud_volume*100:.1f}%)")

# 4. Merchant Category Breakdown
cat_stats = defaultdict(lambda: {"total": 0, "fraud": 0, "loss": 0.0})
for r in rows:
    c = r["merchant_category"]
    cat_stats[c]["total"] += 1
    if r["fraud"] == 1:
        cat_stats[c]["fraud"] += 1
        cat_stats[c]["loss"] += r["amount"]

print("\n" + "=" * 65)
print("3. MERCHANT CATEGORY RISK BREAKDOWN")
print("=" * 65)
print(f"{'Category':<18} | {'Total Tx':<9} | {'Fraud Tx':<9} | {'Fraud Rate':<10} | {'Loss ($)':<12}")
print("-" * 65)
for c, d in sorted(cat_stats.items(), key=lambda x: x[1]["fraud"]/x[1]["total"], reverse=True):
    rate = (d["fraud"] / d["total"]) * 100
    print(f"{c:<18} | {d['total']:<9} | {d['fraud']:<9} | {rate:>9.2f}% | ${d['loss']:>10,.2f}")

print("\n" + "=" * 65)
print("EXECUTION COMPLETE — ZERO EXTERNAL DEPENDENCIES")
print("=" * 65)



# Add to bottom of train_model.py:
output_path = "dataset/scored_transactions.csv"
fieldnames = list(rows[0].keys())

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nAudit log saved to: {output_path}")