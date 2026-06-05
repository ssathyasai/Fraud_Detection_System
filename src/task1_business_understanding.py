"""Task 1: Business Understanding"""

def run():
    print("=" * 60)
    print("TASK 1 — Business Understanding: Fraud Detection")
    print("=" * 60)
    print("""
WHY IS FRAUD DETECTION DIFFICULT?
───────────────────────────────────
1. Class Imbalance — fraud is ~5-12% of transactions.
   A naïve model predicting "legitimate" for everything achieves >88% accuracy
   yet catches ZERO fraudulent transactions.

2. Adversarial — fraudsters constantly adapt to evade detection rules.

3. Sequential Context — a $5000 electronics purchase after a $20 grocery
   transaction is far more suspicious than the same purchase in isolation.

4. Cost Asymmetry:
   • False Negative (missed fraud) → direct financial loss, liability
   • False Positive (wrong alarm)  → customer friction, lost revenue

5. Real-time Constraint — decisions must happen in milliseconds.

WHY IS ACCURACY ALONE MISLEADING?
───────────────────────────────────
Dataset: 437 legitimate (87.4%) + 63 fraud (12.6%)

  Dummy classifier (always predicts 0):
    Accuracy  = 87.4%    ← looks good!
    Recall    =  0.0%    ← catches NO fraud
    Precision =  0.0%
    F1 Score  =  0.0%

  Better metrics for imbalanced problems:
    • Recall   — "of all actual frauds, how many did we catch?"
    • Precision — "of our fraud alerts, how many are real?"
    • F1 Score  — harmonic mean of precision + recall
    • PR-AUC    — precision-recall area under curve
""")

if __name__ == "__main__":
    run()
