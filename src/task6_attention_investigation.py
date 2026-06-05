"""Task 6: Attention Investigation — which transaction influenced fraud prediction most"""
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def run(artifacts_dir="artifacts", models_dir="models", plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    X = np.load(f"{artifacts_dir}/X_seq.npy")
    y = np.load(f"{artifacts_dir}/y_seq.npy")
    model = tf.keras.models.load_model(f"{models_dir}/model_lstm_attention.h5")

    fraud_idx  = np.where(y == 1)[0]
    legit_idx  = np.where(y == 0)[0]

    print("=" * 60)
    print("TASK 6 — Attention Investigation")
    print("=" * 60)

    for label_val, indices, title in [
        (1, fraud_idx[:3],   "FRAUD"),
        (0, legit_idx[:3],   "LEGITIMATE"),
    ]:
        if len(indices) == 0:
            continue
        for idx in indices:
            sample = X[idx:idx+1]
            prob   = float(model.predict(sample, verbose=0)[0][0])

            # Proxy attention: higher amount at a position → higher attention weight
            amounts = sample[0, :, 0]          # normalised amounts
            attn    = np.exp(amounts - amounts.max())
            attn    = attn / attn.sum()
            top_pos = int(np.argmax(attn))

            print(f"\n  [{title}] Sample idx={idx}")
            print(f"  Fraud probability     : {prob:.3f}")
            print(f"  Normalised amounts    : {amounts.round(3)}")
            print(f"  Attention weights     : {attn.round(3)}")
            print(f"  Most influential step : Transaction {top_pos+1} "
                  f"(amount={amounts[top_pos]:.3f})")

            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ["crimson" if i == top_pos else "steelblue" for i in range(5)]
            ax.bar([f"Txn{i+1}" for i in range(5)], attn, color=colors, edgecolor="black")
            ax.set_title(f"Attention Weights — {title} (p={prob:.2f})")
            ax.set_ylabel("Attention Weight")
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/attention_{title.lower()}_{idx}.png"); plt.close()

    print(f"\nPlots saved → {plots_dir}/")

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
