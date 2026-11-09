# FraudGuard | Financial Transaction Anomaly & Risk Engine

An automated fraud detection and behavioral telemetry pipeline that scores incoming transactions against historical baseline deviations, geovelocity flags, and authentication failure patterns.

## Business Problem & Context
Payment fraud costs digital merchants an estimated 1.8% to 3.2% in gross chargeback losses annually. The core analytical challenge is balancing **Recall** (catching fraudulent attempts) against **Precision / False Positive Rate** (avoiding blocking high-value legitimate customers).

## Key Analytical Findings
- **Spend Deviation Ratio:** Transactions exceeding 3.5× a customer's rolling 30-day baseline average represent the single highest correlation with unauthorized account takeover.
- **Authentication Surge:** Accounts with 2+ failed authentication attempts combined with an unrecognized device fingerprint exhibit a 4.2× higher fraud rate than the portfolio average.
- **Category Risk Concentration:** Digital goods and travel categories accounted for over 60% of total fraudulent dollar losses.

## Engine Architecture & Decision Logic
The pipeline scores each record from 0 to 100 based on five behavioral risk multipliers:
1. **Surge Factor:** Current transaction amount vs. customer historical baseline.
2. **Device Anomaly:** Hardware fingerprint mismatch detection.
3. **Geovelocity Check:** Distance from primary billing origin and cross-border indicators.
4. **Velocity Burst:** Surge in transaction frequency over rolling 24-hour windows.
5. **Authentication Security:** Failed password/OTP attempts prior to transaction submission.

### Decision Thresholds:
- **Score ≥ 65:** Immediate Block (Auto-Decline)
- **Score 40–64:** Challenge Required (3D Secure / OTP Step-up)
- **Score < 40:** Clean Approval (Zero checkout friction)

## Performance & Evaluation Metrics
- **Dataset Size:** 10,000 real-world simulated transaction logs
- **Recall (Fraud Interception Rate):** ~85%+ of fraudulent transactions caught
- **False Positive Rate:** < 2.5% legitimate customer impact
- **Capital Protected:** Measured in direct gross dollar volume saved vs. total exposure

## Running the Engine
