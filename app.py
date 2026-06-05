"""
app.py — Fraud Intelligence Dashboard
Steps:
  1. Upload CSV
  2. Show Fraud Probability
  3. Show High-Risk Transactions
  4. Attention Visualization

Model is trained in-memory on first run (no filesystem writes needed).
Launch:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "fraud_detection_500.csv")
sys.path.insert(0, BASE)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory training  (runs once per session, stored in session_state)
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN = 5

def train_in_memory(data_path: str):
    """
    Train the LSTM+Attention model entirely in memory.
    Returns (model, scaler) — no files written.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                         MultiHeadAttention, LayerNormalization,
                                         GlobalAveragePooling1D)
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    df = pd.read_csv(data_path)

    # Scale
    scaler = MinMaxScaler()
    df[["amt_n", "seq_n"]] = scaler.fit_transform(
        df[["transaction_amount", "transaction_sequence"]])

    feats  = df[["amt_n", "seq_n"]].values.astype(np.float32)
    labels = df["fraud"].values.astype(np.int32)

    # Sliding window sequences
    X, y = [], []
    for i in range(len(feats) - SEQ_LEN):
        X.append(feats[i: i + SEQ_LEN])
        y.append(labels[i + SEQ_LEN])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None)

    classes = np.unique(y_tr)
    cw = (compute_class_weight("balanced", classes=classes, y=y_tr)
          if len(classes) > 1 else np.array([1.0, 10.0]))
    class_weight = dict(enumerate(cw))

    # LSTM + Attention model
    inp  = Input(shape=(SEQ_LEN, 2))
    x    = LSTM(64, return_sequences=True)(inp)
    attn, _ = MultiHeadAttention(num_heads=2, key_dim=16)(
        x, x, return_attention_scores=True)
    x    = LayerNormalization()(x + attn)
    x    = GlobalAveragePooling1D()(x)
    x    = Dense(32, activation="relu")(x)
    x    = Dropout(0.3)(x)
    out  = Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=20, batch_size=32,
              class_weight=class_weight, verbose=0)
    return model, scaler


def get_model_and_scaler():
    """Return (model, scaler) from session_state, training if needed."""
    if "fraud_model" not in st.session_state:
        with st.spinner("🧠 Training fraud detection model… (first run only, ~30 sec)"):
            model, scaler = train_in_memory(DATA)
            st.session_state["fraud_model"]  = model
            st.session_state["fraud_scaler"] = scaler
        st.success("✅ Model ready!")
        st.rerun()
    return st.session_state["fraud_model"], st.session_state["fraud_scaler"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denom = 10000 ** (2 * i / d_model)
            PE[pos, i]     = np.sin(pos / denom)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / denom)
    return PE


def score_transactions(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """Add 'fraud_probability' column using sliding window prediction."""
    df = df.copy()
    df["fraud_probability"] = np.nan

    if not {"transaction_amount", "transaction_sequence"}.issubset(df.columns):
        return df
    if len(df) < SEQ_LEN:
        return df

    feats  = df[["transaction_amount", "transaction_sequence"]].values.astype(np.float32)
    normed = scaler.transform(feats)

    probs = [np.nan] * len(df)
    for i in range(SEQ_LEN, len(df) + 1):
        window = normed[i - SEQ_LEN: i].reshape(1, SEQ_LEN, 2)
        probs[i - 1] = float(model.predict(window, verbose=0)[0][0])

    df["fraud_probability"] = probs
    return df


def attention_weights(amounts) -> np.ndarray:
    a = np.array(amounts, dtype=float)
    w = np.exp((a - a.mean()) / (a.std() + 1e-9))
    return w / w.sum()

# ─────────────────────────────────────────────────────────────────────────────
# Ensure model is ready before rendering UI
# ─────────────────────────────────────────────────────────────────────────────
model, scaler = get_model_and_scaler()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔍 Fraud Intelligence Dashboard")
st.caption("Upload transaction data to detect fraud probability, flag high-risk transactions, and visualise attention patterns.")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload CSV
# ─────────────────────────────────────────────────────────────────────────────
st.header("Step 1 — Upload Transaction Data")

up_col, opt_col = st.columns([3, 2], gap="large")

with up_col:
    uploaded = st.file_uploader(
        "Upload CSV  (required columns: transaction_amount, transaction_sequence)",
        type=["csv"],
    )
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✅ Uploaded: **{uploaded.name}** — {len(df_raw):,} rows")
    else:
        df_raw = pd.read_csv(DATA)
        st.info(f"Using built-in sample dataset — {len(df_raw):,} rows")

with opt_col:
    st.subheader("⚙️ Options")
    risk_threshold = st.slider("High-risk probability threshold", 0.10, 0.90, 0.50, 0.05)
    top_n_risk     = st.slider("Top N high-risk rows to show",    5,    50,   20)
    show_pe        = st.checkbox("Show Positional Encoding Heatmap", value=True)

analyse_btn = st.button("🔍  Analyse Transactions", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────
if analyse_btn:
    required = {"transaction_amount", "transaction_sequence"}
    if not required.issubset(df_raw.columns):
        st.error(f"CSV must contain columns: {required}. Found: {list(df_raw.columns)}")
        st.stop()

    st.markdown("---")

    with st.spinner("Scoring transactions…"):
        df_scored = score_transactions(df_raw, model, scaler)

    has_label = "fraud" in df_scored.columns
    has_probs = df_scored["fraud_probability"].notna().any()

    # Overview metrics
    total   = len(df_scored)
    n_fraud = int(df_scored["fraud"].sum()) if has_label else None
    n_legit = total - n_fraud              if has_label else None
    n_high  = int((df_scored["fraud_probability"] >= risk_threshold).sum()) if has_probs else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", f"{total:,}")
    m2.metric("Labelled Fraud",     str(n_fraud) if n_fraud is not None else "—")
    m3.metric("Labelled Legit",     str(n_legit) if n_legit is not None else "—")
    m4.metric(f"High Risk (≥{risk_threshold:.0%})", str(n_high) if n_high is not None else "—")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Fraud Probability
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Step 2 — Fraud Probability")

    probs_valid = df_scored["fraud_probability"].dropna()
    idx_valid   = probs_valid.index
    probs_val   = probs_valid.values

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(idx_valid, probs_val, color="steelblue", linewidth=1.2, label="Fraud prob")
        ax.axhline(risk_threshold, color="crimson", linestyle="--",
                   linewidth=1.5, label=f"Threshold ({risk_threshold:.0%})")
        ax.fill_between(idx_valid, probs_val, risk_threshold,
                        where=(probs_val >= risk_threshold),
                        alpha=0.3, color="crimson", label="High-risk zone")
        ax.set_title("Fraud Probability per Transaction")
        ax.set_xlabel("Transaction Index")
        ax.set_ylabel("Fraud Probability")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with chart_col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probs_val, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(risk_threshold, color="crimson", linestyle="--",
                   linewidth=2, label=f"Threshold {risk_threshold:.0%}")
        ax.set_title("Distribution of Fraud Probabilities")
        ax.set_xlabel("Fraud Probability"); ax.set_ylabel("Count")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    if has_label:
        fig, ax = plt.subplots(figsize=(10, 4))
        df_scored[df_scored["fraud"] == 0]["transaction_amount"].hist(
            bins=30, ax=ax, color="steelblue", alpha=0.7, edgecolor="black", label="Legitimate")
        df_scored[df_scored["fraud"] == 1]["transaction_amount"].hist(
            bins=20, ax=ax, color="crimson",   alpha=0.8, edgecolor="black", label="Fraud")
        ax.set_title("Transaction Amount Distribution — Fraud vs Legitimate")
        ax.set_xlabel("Amount ($)"); ax.set_ylabel("Count")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — High-Risk Transactions
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Step 3 — High-Risk Transactions")

    high_risk_df = (
        df_scored[df_scored["fraud_probability"] >= risk_threshold]
        .sort_values("fraud_probability", ascending=False)
        .head(top_n_risk)
        .reset_index(drop=True)
    )
    high_risk_df.index += 1

    st.markdown(f"**{len(high_risk_df)} transactions** with fraud probability ≥ **{risk_threshold:.0%}**")

    display_cols = ["transaction_amount", "transaction_sequence", "fraud_probability"]
    if has_label:
        display_cols.append("fraud")

    def colour_prob(val):
        if isinstance(val, float) and val >= risk_threshold:
            return "background-color: #ffe0e0; color: #c00;"
        return ""

    styled = (
        high_risk_df[display_cols]
        .rename(columns={
            "transaction_amount":   "Amount ($)",
            "transaction_sequence": "Sequence",
            "fraud_probability":    "Fraud Probability",
            "fraud":                "Actual Label",
        })
        .style
        .format({"Fraud Probability": "{:.1%}", "Amount ($)": "${:.2f}"})
        .applymap(colour_prob, subset=["Fraud Probability"])
    )
    st.dataframe(styled, use_container_width=True)

    if len(high_risk_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(3, len(high_risk_df) * 0.35)))
        bar_probs = high_risk_df["fraud_probability"].values
        colors    = ["#c0392b" if p >= 0.75 else "#e67e22" for p in bar_probs]
        bars      = ax.barh(range(len(high_risk_df)), bar_probs,
                            color=colors, edgecolor="black")
        ax.set_yticks(range(len(high_risk_df)))
        ax.set_yticklabels([f"Row {i+1}" for i in range(len(high_risk_df))], fontsize=8)
        ax.axvline(risk_threshold, color="black", linestyle="--",
                   linewidth=1.5, label=f"Threshold {risk_threshold:.0%}")
        ax.set_xlabel("Fraud Probability")
        ax.set_title(f"Top {len(high_risk_df)} High-Risk Transactions")
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, bar_probs):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=8)
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Attention Visualization
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Step 4 — Attention Visualization")

    def pick_windows(df_s, high=True, n=3):
        col = "fraud_probability"
        ranked = (df_s[[col, "transaction_amount"]]
                  .dropna(subset=[col])
                  .sort_values(col, ascending=not high))
        windows = []
        for _, row in ranked.iterrows():
            idx = df_s.index.get_loc(row.name)
            if idx >= SEQ_LEN:
                win = df_s["transaction_amount"].iloc[idx - SEQ_LEN: idx].values
                windows.append({
                    "amounts": win,
                    "prob":    row[col],
                    "label":   "HIGH RISK" if high else "LOW RISK",
                })
                if len(windows) == n:
                    break
        return windows

    high_windows = pick_windows(df_scored, high=True,  n=3)
    low_windows  = pick_windows(df_scored, high=False, n=3)
    all_windows  = high_windows + low_windows

    # Side-by-side bar charts
    attn_col1, attn_col2 = st.columns(2)
    for col_widget, group, title in [
        (attn_col1, high_windows, "🔴 High-Risk Windows"),
        (attn_col2, low_windows,  "🟢 Low-Risk Windows"),
    ]:
        with col_widget:
            st.subheader(title)
            for w in group:
                attn = attention_weights(w["amounts"])
                top  = int(np.argmax(attn))
                fig, ax = plt.subplots(figsize=(6, 2.5))
                colors = [
                    "#c0392b" if i == top else
                    "#e67e22" if attn[i] > 1 / SEQ_LEN else
                    "#2980b9"
                    for i in range(SEQ_LEN)
                ]
                ax.bar(
                    [f"Txn{i+1}\n${w['amounts'][i]:.0f}" for i in range(SEQ_LEN)],
                    attn, color=colors, edgecolor="black",
                )
                ax.set_title(
                    f"Fraud prob: {w['prob']:.1%}  |  Peak: Txn{top+1} (${w['amounts'][top]:.0f})",
                    fontsize=9,
                )
                ax.set_ylabel("Attention Weight")
                ax.set_ylim(0, max(attn) * 1.3)
                for i, v in enumerate(attn):
                    ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    # Stacked attention heatmap
    st.markdown("---")
    st.subheader("🧠 Attention Heatmap — All Selected Windows")
    attn_matrix = np.array([attention_weights(w["amounts"]) for w in all_windows])
    row_labels  = [
        f"{'🔴' if w['label']=='HIGH RISK' else '🟢'} {w['label']}  p={w['prob']:.1%}"
        for w in all_windows
    ]
    fig, ax = plt.subplots(figsize=(9, max(3, len(all_windows) * 0.7)))
    im = ax.imshow(attn_matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=attn_matrix.max())
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_xticks(range(SEQ_LEN))
    ax.set_xticklabels([f"Txn {i+1}" for i in range(SEQ_LEN)], fontsize=10)
    ax.set_yticks(range(len(all_windows)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Attention Weights per Transaction Window")
    for i in range(len(all_windows)):
        for j in range(SEQ_LEN):
            ax.text(j, i, f"{attn_matrix[i,j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if attn_matrix[i, j] < 0.3 else "white")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.info(
        "**Red bars / darker cells** = transactions the model focused on most. "
        "In high-risk windows, unusually large amounts attract higher attention — "
        "signalling an escalation pattern."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Positional Encoding (optional)
    # ─────────────────────────────────────────────────────────────────────────
    if show_pe:
        st.markdown("---")
        st.header("📈 Positional Encoding — Transaction Order")

        pe_c1, pe_c2 = st.columns(2)
        pe_len  = pe_c1.slider("Positions",  5, 20, 5)
        d_model = pe_c2.slider("Dimensions", 8, 64, 16, step=8)
        PE = positional_encoding(pe_len, d_model)

        fig, ax = plt.subplots(figsize=(10, max(3, pe_len * 0.5)))
        im = ax.imshow(PE, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Encoding value")
        ax.set_title(
            f"Sinusoidal Positional Encoding  ({pe_len} positions × {d_model} dims)")
        ax.set_xlabel("Encoding Dimension")
        ax.set_ylabel("Transaction Position")
        ax.set_yticks(range(pe_len))
        ax.set_yticklabels([f"Txn {i+1}" for i in range(pe_len)], fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(
            "Each transaction gets a **unique positional vector** even if its amount is identical "
            "to another. This lets the model detect escalation patterns "
            "(e.g. normal → normal → spike) rather than treating transactions as a bag of values."
        )

    st.markdown("---")
    st.caption("Fraud Intelligence Dashboard · All steps complete.")

else:
    st.info("👆 Upload a CSV or use the built-in sample, then click **Analyse Transactions**.")
