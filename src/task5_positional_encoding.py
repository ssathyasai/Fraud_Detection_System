"""Task 5: Positional Encoding for transaction sequences + explanation"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i]     = np.sin(pos / (10000 ** (2*i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (2*i/d_model)))
    return PE

def run(plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    SEQ_LEN = 5
    D_MODEL = 16
    PE = positional_encoding(SEQ_LEN, D_MODEL)

    print("=" * 60)
    print("TASK 5 — Positional Encoding for Transaction Sequences")
    print("=" * 60)
    print(f"PE shape: {PE.shape}\n")
    for i in range(SEQ_LEN):
        print(f"  Txn {i+1} PE (first 6): {PE[i, :6].round(4)}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(PE, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    ax.set_title("Transaction Order — Positional Encoding Heatmap")
    ax.set_xlabel("Encoding Dimension"); ax.set_ylabel("Transaction Position")
    ax.set_yticks(range(SEQ_LEN))
    ax.set_yticklabels([f"Txn {i+1}" for i in range(SEQ_LEN)])
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/transaction_pe_heatmap.png"); plt.close()
    print(f"\nSaved: {plots_dir}/transaction_pe_heatmap.png")

    print("""
WHY ORDER OF TRANSACTIONS MATTERS:
  Sequence A: grocery($50) → restaurant($30) → electronics($4500)
              ← escalation pattern, high fraud risk

  Sequence B: electronics($4500) → grocery($50) → restaurant($30)
              ← high first then normal, lower risk

  Without PE: model can't distinguish Seq A from Seq B (same values, different order).
  With PE:    each step gets a unique position vector → model detects escalation.
  
  Transaction 1 (Txn1) and Transaction 5 (Txn5) have DIFFERENT PE vectors
  even if the transaction amounts are identical.
""")
    return PE

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
