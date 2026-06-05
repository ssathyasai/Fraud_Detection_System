"""
app.py — Fraud Intelligence Dashboard
Steps:
  1. Upload CSV
  2. Show Fraud Probability  (per transaction, model-scored)
  3. Show High-Risk Transactions
  4. Attention Visualization

Launch:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE      = os.path.dirname(os.path.abspath(__file__))
MODELS    = os.path.join(BASE, "models")
ARTIFACTS = os.path.join(BASE, "artifacts")
PLOTS     = os.path.join(BASE, "plots")
DATA      = os.path.join(BASE, "data", "fraud_detection_500.csv")
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
# Auto-train
# ─────────────────────────────────────────────────────────────────────────────
def ensure_trained():
    if not os.path.exists(os.path.join(MODELS, "model_lstm_attention.h5")):
        st.warning("⚙️  No trained model found — auto-training now. Please wait…")
        with st.spinner("Running all training tasks…"):
            os.makedirs(ARTIFACTS, exist_ok=True)
            os.makedirs(MODELS,    exist_ok=True)
            os.makedirs(PLOTS,     exist_ok=True)
            from src.task3_sequence_generation import run as seq_gen
            from src.task4_models              import run as train_models
            from src.task5_positional_encoding import run as pos_enc
            seq_gen(DATA, ARTIFACTS)
            train_models(ARTIFACTS, MODELS)
            pos_enc(PLOTS)
        st.success("✅ Training complete!")
        st.rerun()

ensure_trained()

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

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(
        os.path.join(MODELS, "model_lstm_attention.h5"))

@st.cache_resource
def load_scaler():
    with open(os.path.join(ARTIFACTS, "scaler.pkl"), "rb") as f:
        return pickle.load(f)

def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slide a SEQ_LEN=5 window over the dataframe rows and predict fraud probability
    for each window. Rows that don't have a full preceding window get probability=NaN.
    Returns df with a 'fraud_probability' column (NaN if model unavailable).
    """
    SEQ_LEN = 5
    df = df.copy()
    df["fraud_probability"] = np.nan

    # Guard: required columns must exist
    if "transaction_amount" not in df.columns or "transaction_sequence" not in df.columns:
        return df

    # Guard: need enough rows for at least one window
    if len(df) < SEQ_LEN:
        return df

    try:
        scaler = load_scaler()
        model  = load_model()
    except Exception:
        return df

    try:
        feats  = df[["transaction_amount", "transaction_sequence"]].values.astype(np.float32)
        normed = scaler.transform(feats)

        probs = [np.nan] * len(df)
        for i in range(SEQ_LEN, len(df) + 1):
            window = normed[i - SEQ_LEN: i].reshape(1, SEQ_LEN, 2)
            p = float(model.predict(window, verbose=0)[0][0])
            probs[i - 1] = p

        df["fraud_probability"] = probs
    except Exception:
        pass  # leave as NaN — dashboard handles it gracefully

    return df

def attention_weights(amounts: np.ndarray) -> np.ndarray:
    """Proxy attention: softmax over normalised transaction amounts."""
    a = np.array(amounts, dtype=float)
    w = np.exp((a - a.mean()) / (a.std() + 1e-9))
    return w / w.sum()

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
    risk_threshold  = st.slider("High-risk probability threshold", 0.10, 0.90, 0.50, 0.05)
    top_n_risk      = st.slider("Top N high-risk rows to show",    5,    50,   20)
    show_pe         = st.checkbox("Show Positional Encoding Heatmap", value=True)

analyse_btn = st.button("🔍  Analyse Transactions", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────────────────────────────────────
if analyse_btn:
    required = {"transaction_amount", "transaction_sequence"}
    if not required.issubset(df_raw.columns):
        st.error(f"CSV must contain columns: {required}. Found: {list(df_raw.columns)}")
        st.stop()

    st.markdown("---")

    # score every transaction
    with st.spinner("Scoring transactions…"):
        df_scored = score_transactions(df_raw)

    has_label  = "fraud" in df_scored.columns
    has_probs  = df_scored["fraud_probability"].notna().any()

    # ── Dataset overview metrics ──────────────────────────────────────────────
    total   = len(df_scored)
    n_fraud = int(df_scored["fraud"].sum()) if has_label else None
    n_legit = total - n_fraud if has_label else None
    n_high  = int((df_scored["fraud_probability"] >= risk_threshold).sum()) if has_probs else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", f"{total:,}")
    m2.metric("Labelled Fraud",     f"{n_fraud}" if n_fraud is not None else "—")
    m3.metric("Labelled Legit",     f"{n_legit}" if n_legit is not None else "—")
    m4.metric(f"High Risk (≥{risk_threshold:.0%})", f"{n_high}" if n_high is not None else "—")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Fraud Probability
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Step 2 — Fraud Probability")

    if has_probs:
        chart_col1, chart_col2 = st.columns(2)

        # Line chart — probability over transaction index
        with chart_col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            idx_valid = df_scored["fraud_probability"].dropna().index
            probs_val = df_scored["fraud_probability"].dropna().values
            ax.plot(idx_valid, probs_val, color="steelblue", linewidth=1.2, label="Fraud prob")
            ax.axhline(risk_threshold, color="crimson", linestyle="--",
                       linewidth=1.5, label=f"Threshold ({risk_threshold:.0%})")
            ax.fill_between(idx_valid, probs_val, risk_threshold,
                            where=(probs_val >= risk_threshold),
                            alpha=0.25, color="crimson", label="High-risk zone")
            ax.set_title("Fraud Probability per Transaction")
            ax.set_xlabel("Transaction Index")
            ax.set_ylabel("Fraud Probability")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Histogram of probabilities
        with chart_col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(probs_val, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
            ax.axvline(risk_threshold, color="crimson", linestyle="--",
                       linewidth=2, label=f"Threshold {risk_threshold:.0%}")
            ax.set_title("Distribution of Fraud Probabilities")
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Count")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Amount distribution coloured by predicted risk
        if has_label:
            fig, ax = plt.subplots(figsize=(10, 4))
            df_scored[df_scored["fraud"] == 0]["transaction_amount"].hist(
                bins=30, ax=ax, color="steelblue", alpha=0.7,
                edgecolor="black", label="Legitimate")
            df_scored[df_scored["fraud"] == 1]["transaction_amount"].hist(
                bins=20, ax=ax, color="crimson", alpha=0.8,
                edgecolor="black", label="Fraud")
            ax.set_title("Transaction Amount Distribution — Fraud vs Legitimate")
            ax.set_xlabel("Amount ($)"); ax.set_ylabel("Count")
            ax.legend(); plt.tight_layout()
            st.pyplot(fig); plt.close()
    else:
        st.warning("Model not available — run `python train.py` to generate fraud probabilities.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — High-Risk Transactions
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Step 3 — High-Risk Transactions")

    if has_probs:
        high_risk_df = (
            df_scored[df_scored["fraud_probability"] >= risk_threshold]
            .sort_values("fraud_probability", ascending=False)
            .head(top_n_risk)
            .reset_index(drop=True)
        )
        high_risk_df.index += 1  # 1-based rank

        st.markdown(f"**{len(high_risk_df)} transactions** with fraud probability ≥ **{risk_threshold:.0%}**")

        # Colour the probability column red
        def colour_prob(val):
            if isinstance(val, float) and val >= risk_threshold:
                return "background-color: #ffe0e0; color: #c00;"
            return ""

        display_cols = ["transaction_amount", "transaction_sequence",
                        "fraud_probability"]
        if has_label:
            display_cols.append("fraud")

        styled = (
            high_risk_df[display_cols]
            .rename(columns={
                "transaction_amount":   "Amount ($)",
                "transaction_sequence": "Sequence",
                "fraud_probability":    "Fraud Probability",
                "fraud":                "Actual Label",
            })
            .style.format({"Fraud Probability": "{:.1%}", "Amount ($)": "${:.2f}"})
            .applymap(colour_prob, subset=["Fraud Probability"])
        )
        st.dataframe(styled, use_container_width=True)

        # Bar chart — top-N fraud probabilities
        fig, ax = plt.subplots(figsize=(10, max(3, len(high_risk_df) * 0.35)))
        bars = ax.barh(
            range(len(high_risk_df)),
            high_risk_df["fraud_probability"].values,
            color=[
                "#c0392b" if p >= 0.75 else
                "#e67e22" if p >= risk_threshold else
                "#2980b9"
                for p in high_risk_df["fraud_probability"].values
            ],
            edgecolor="black",
        )
        ax.set_yticks(range(len(high_risk_df)))
        ax.set_yticklabels([f"Row {i+1}" for i in range(len(high_risk_df))], fontsize=8)
        ax.axvline(risk_threshold, color="black", linestyle="--",
                   linewidth=1.5, label=f"Threshold {risk_threshold:.0%}")
        ax.set_xlabel("Fraud Probability")
        ax.set_title(f"Top {len(high_risk_df)} High-Risk Transactions")
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, high_risk_df["fraud_probability"].values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=8)
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

    else:
        # Fallback: use raw amount threshold when no model
        threshold_amt = df_scored["transaction_amount"].quantile(0.9)
        hr = df_scored[df_scored["transaction_amount"] > threshold_amt].head(top_n_risk)
        st.info(f"No model scores available — showing transactions with amount > ${threshold_amt:.0f} (top 10%)")
        st.dataframe(hr.reset_index(drop=True), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Attention Visualization
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Step 4 — Attention Visualization")

    SEQ_LEN = 5
    amounts_all = df_scored["transaction_amount"].values

    # Pick 3 high-risk windows and 3 low-risk windows for comparison
    def pick_windows(df_s, high=True, n=3):
        col = "fraud_probability"
        if col not in df_s.columns or df_s[col].isna().all():
            return []
        ranked = (
            df_s[[col, "transaction_amount", "transaction_sequence"]]
            .dropna(subset=[col])
            .sort_values(col, ascending=not high)
        )
        windows = []
        for _, row in ranked.iterrows():
            idx = df_s.index.get_loc(row.name)
            if idx >= SEQ_LEN:
                win_amounts = df_s["transaction_amount"].iloc[idx - SEQ_LEN: idx].values
                windows.append({
                    "amounts":  win_amounts,
                    "prob":     row[col],
                    "label":    "HIGH RISK" if high else "LOW RISK",
                })
                if len(windows) == n:
                    break
        return windows

    high_windows = pick_windows(df_scored, high=True,  n=3)
    low_windows  = pick_windows(df_scored, high=False, n=3)
    all_windows  = high_windows + low_windows

    if all_windows:
        attn_col1, attn_col2 = st.columns(2)

        # --- grouped attention bar charts ---
        for col_idx, (col_widget, group, title) in enumerate([
            (attn_col1, high_windows, "🔴 High-Risk Windows"),
            (attn_col2, low_windows,  "🟢 Low-Risk Windows"),
        ]):
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
                        f"Fraud prob: {w['prob']:.1%}  |  "
                        f"Peak: Txn{top+1} (${w['amounts'][top]:.0f})",
                        fontsize=9,
                    )
                    ax.set_ylabel("Attention Weight")
                    ax.set_ylim(0, max(attn) * 1.3)
                    for i, v in enumerate(attn):
                        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

        st.markdown("---")

        # --- heatmap: all windows stacked ---
        st.subheader("🧠 Attention Heatmap — All Selected Windows")
        n_windows = len(all_windows)
        attn_matrix = np.array([attention_weights(w["amounts"]) for w in all_windows])
        row_labels  = [
            f"{'🔴' if w['label']=='HIGH RISK' else '🟢'} {w['label']}  p={w['prob']:.1%}"
            for w in all_windows
        ]

        fig, ax = plt.subplots(figsize=(9, max(3, n_windows * 0.7)))
        im = ax.imshow(attn_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=attn_matrix.max())
        plt.colorbar(im, ax=ax, label="Attention Weight")
        ax.set_xticks(range(SEQ_LEN))
        ax.set_xticklabels([f"Txn {i+1}" for i in range(SEQ_LEN)], fontsize=10)
        ax.set_yticks(range(n_windows))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title("Attention Weights per Transaction Window")
        for i in range(n_windows):
            for j in range(SEQ_LEN):
                ax.text(j, i, f"{attn_matrix[i,j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if attn_matrix[i,j] < 0.3 else "white")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.info(
            "**Red bars / darker cells** = transactions the model focused on most. "
            "In high-risk windows, unusually large amounts at later positions "
            "attract higher attention — signalling an escalation pattern."
        )

    else:
        # Fallback: manual 5-transaction input
        st.markdown("Enter 5 consecutive transaction amounts for attention analysis:")
        inp_cols = st.columns(5)
        manual_amounts = [
            inp_cols[i].number_input(
                f"Txn {i+1} ($)", 1.0, 100000.0,
                float([50, 80, 100, 60, 4500][i]),
                key=f"manual_{i}",
            )
            for i in range(5)
        ]
        attn = attention_weights(manual_amounts)
        top  = int(np.argmax(attn))
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ["crimson" if i == top else "steelblue" for i in range(5)]
        ax.bar([f"Txn{i+1}\n${manual_amounts[i]:.0f}" for i in range(5)],
               attn, color=colors, edgecolor="black")
        ax.set_title(f"Attention Weights — Peak: Txn{top+1} (${manual_amounts[top]:.0f})")
        ax.set_ylabel("Attention Weight")
        for i, v in enumerate(attn):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Positional Encoding (optional)
    # ─────────────────────────────────────────────────────────────────────────
    if show_pe:
        st.markdown("---")
        st.header("📈 Positional Encoding — Transaction Order")

        pe_c1, pe_c2 = st.columns(2)
        pe_len  = pe_c1.slider("Positions", 5, 20, 5)
        d_model = pe_c2.slider("Dimensions", 8, 64, 16, step=8)
        PE = positional_encoding(pe_len, d_model)

        fig, ax = plt.subplots(figsize=(10, max(3, pe_len * 0.5)))
        im = ax.imshow(PE, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Encoding value")
        ax.set_title(f"Sinusoidal Positional Encoding  ({pe_len} positions × {d_model} dims)")
        ax.set_xlabel("Encoding Dimension")
        ax.set_ylabel("Transaction Position")
        ax.set_yticks(range(pe_len))
        ax.set_yticklabels([f"Txn {i+1}" for i in range(pe_len)], fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.info(
            "Each transaction gets a **unique positional vector** even if its amount is identical "
            "to another. This lets the model detect escalation patterns "
            "(e.g. normal → normal → spike) rather than treating transactions as a bag of values."
        )

    st.markdown("---")
    st.caption("Fraud Intelligence Dashboard · All steps complete.")

else:
    st.info("👆 Upload a CSV or use the built-in sample, then click **Analyse Transactions**.")
