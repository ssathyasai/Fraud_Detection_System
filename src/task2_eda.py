"""Task 2: EDA — fraud %, imbalance, visualisations"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def run(data_path="data/fraud_detection_500.csv", plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    total  = len(df)
    n_fr   = df["fraud"].sum()
    n_leg  = total - n_fr

    print("=" * 55)
    print("TASK 2 — EDA: Fraud Detection")
    print("=" * 55)
    print(f"Total transactions  : {total}")
    print(f"Fraud               : {n_fr}  ({n_fr/total*100:.1f}%)")
    print(f"Legitimate          : {n_leg} ({n_leg/total*100:.1f}%)")
    print(f"Class imbalance     : 1 : {n_leg // max(n_fr,1)}")
    print(f"\nAmount stats:\n{df['transaction_amount'].describe().round(2).to_string()}")
    print(f"\nSequence stats:\n{df['transaction_sequence'].describe().round(2).to_string()}")

    # ── Plot 1: Amount distribution split by fraud flag ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df[df["fraud"]==0]["transaction_amount"].hist(
        bins=30, ax=axes[0], color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_title("Legitimate — Transaction Amount")
    axes[0].set_xlabel("Amount ($)"); axes[0].set_ylabel("Count")

    df[df["fraud"]==1]["transaction_amount"].hist(
        bins=20, ax=axes[1], color="crimson", edgecolor="black", alpha=0.8)
    axes[1].set_title("Fraud — Transaction Amount")
    axes[1].set_xlabel("Amount ($)")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/amount_distribution.png"); plt.close()

    # ── Plot 2: Fraud vs Legitimate counts ───────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Legitimate", "Fraud"], [n_leg, n_fr],
           color=["steelblue", "crimson"], edgecolor="black")
    ax.set_title("Fraud vs Legitimate Count")
    ax.set_ylabel("Count")
    for i, v in enumerate([n_leg, n_fr]):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/fraud_vs_legit.png"); plt.close()

    # ── Plot 3: Correlation heatmap ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[["transaction_amount", "transaction_sequence", "fraud"]].corr()
    im   = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    cols = corr.columns.tolist()
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/correlation_heatmap.png"); plt.close()

    print(f"\nPlots saved → {plots_dir}/")
    return df

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
