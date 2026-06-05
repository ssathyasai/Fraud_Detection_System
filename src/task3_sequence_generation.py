"""Task 3: Sequence Generation — sliding window over transaction_sequence"""
import pandas as pd
import numpy as np
import pickle
import os

SEQ_LEN = 5   # Txn1…Txn5 → predict Txn6

def run(data_path="data/fraud_detection_500.csv", artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Normalise features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[["amt_norm", "seq_norm"]] = scaler.fit_transform(
        df[["transaction_amount", "transaction_sequence"]])

    features = df[["amt_norm", "seq_norm"]].values
    labels   = df["fraud"].values

    X_seqs, y_labels = [], []
    for i in range(len(features) - SEQ_LEN):
        X_seqs.append(features[i: i + SEQ_LEN])
        y_labels.append(labels[i + SEQ_LEN])

    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    print("=" * 55)
    print("TASK 3 — Sequence Generation")
    print("=" * 55)
    print(f"Sequence shape   : {X.shape}  → ({len(X)} windows × {SEQ_LEN} steps × 2 features)")
    print(f"Label shape      : {y.shape}")
    print(f"Fraud in seqs    : {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"\nExample — Txn1…Txn{SEQ_LEN} features:\n{X[0]}")
    print(f"Predict Txn{SEQ_LEN+1} fraud? → {y[0]}")

    np.save(f"{artifacts_dir}/X_seq.npy", X)
    np.save(f"{artifacts_dir}/y_seq.npy", y)
    with open(f"{artifacts_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nArtifacts saved → {artifacts_dir}/")
    return X, y, scaler

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
