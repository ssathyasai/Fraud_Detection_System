"""
app.py — Streamlit Dashboard: Fraud Detection
Auto-trains if no model found.
Launch:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, sys
import matplotlib.pyplot as plt
import tensorflow as tf

BASE      = os.path.dirname(os.path.abspath(__file__))
MODELS    = os.path.join(BASE, "models")
ARTIFACTS = os.path.join(BASE, "artifacts")
PLOTS     = os.path.join(BASE, "plots")
DATA      = os.path.join(BASE, "data", "fraud_detection_500.csv")

# ── Auto-train ────────────────────────────────────────────────────────────────
def ensure_trained():
    if not os.path.exists(os.path.join(MODELS, "model_lstm_attention.h5")):
        st.warning("⚙️  No trained model found. Auto-training now — please wait…")
        with st.spinner("Training all tasks…"):
            sys.path.insert(0, BASE)
            os.makedirs(ARTIFACTS, exist_ok=True)
            os.makedirs(MODELS, exist_ok=True)
            os.makedirs(PLOTS, exist_ok=True)
            from src.task3_sequence_generation    import run as seq_gen
            from src.task4_models                 import run as train_models
            from src.task5_positional_encoding    import run as pos_enc
            seq_gen(DATA, ARTIFACTS)
            train_models(ARTIFACTS, MODELS)
            pos_enc(PLOTS)
        st.success("✅ Training complete!"); st.rerun()

def positional_encoding(seq_len=5, d_model=16):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i]     = np.sin(pos / (10000 ** (2*i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (2*i/d_model)))
    return PE

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        os.path.join(MODELS, "model_lstm_attention.h5"))

@st.cache_resource
def load_scaler():
    with open(os.path.join(ARTIFACTS, "scaler.pkl"), "rb") as f:
        return pickle.load(f)

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection AI", page_icon="🔍", layout="wide")
ensure_trained()

st.title("🔍 Deep Learning Fraud Detection System")
st.caption("Sequential transaction analysis using LSTM + Self-Attention")

tab1, tab2, tab3 = st.tabs(["📊 Upload & Analyse", "⚡ Real-time Simulation", "📈 Positional Encoding"])

# ── Tab 1: Upload CSV ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Transaction Data")
    uploaded = st.file_uploader("Upload CSV (columns: transaction_amount, fraud)", type=["csv"])
    use_sample = st.checkbox("Use sample dataset", value=True if not uploaded else False)

    if uploaded:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = pd.read_csv(DATA)
    else:
        df = None

    if df is not None:
        n_total = len(df)
        n_fraud = int(df["fraud"].sum()) if "fraud" in df.columns else 0
        n_legit = n_total - n_fraud

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", n_total)
        c2.metric("Fraud",    f"{n_fraud} ({n_fraud/n_total*100:.1f}%)")
        c3.metric("Legitimate", f"{n_legit} ({n_legit/n_total*100:.1f}%)")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            if "fraud" in df.columns:
                df[df["fraud"]==0]["transaction_amount"].hist(
                    bins=25, ax=ax, color="steelblue", alpha=0.7, label="Legit", edgecolor="black")
                df[df["fraud"]==1]["transaction_amount"].hist(
                    bins=15, ax=ax, color="crimson", alpha=0.7, label="Fraud", edgecolor="black")
                ax.legend()
            else:
                df["transaction_amount"].hist(bins=25, ax=ax, color="steelblue", edgecolor="black")
            ax.set_title("Transaction Amount Distribution")
            ax.set_xlabel("Amount ($)"); ax.set_ylabel("Count")
            st.pyplot(fig); plt.close()

        with col2:
            if "fraud" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Legitimate", "Fraud"], [n_legit, n_fraud],
                       color=["steelblue", "crimson"], edgecolor="black")
                ax.set_title("Fraud vs Legitimate")
                ax.set_ylabel("Count")
                for i, v in enumerate([n_legit, n_fraud]):
                    ax.text(i, v+2, str(v), ha="center", fontweight="bold")
                st.pyplot(fig); plt.close()

        if "transaction_amount" in df.columns:
            threshold = df["transaction_amount"].quantile(0.9)
            st.subheader(f"⚠️ High Risk Transactions (amount > ${threshold:.0f})")
            high_risk = df[df["transaction_amount"] > threshold]
            st.dataframe(high_risk.head(20))

# ── Tab 2: Real-time Simulation ───────────────────────────────────────────────
with tab2:
    st.subheader("Real-time Fraud Prediction")
    st.markdown("Enter 5 consecutive transaction amounts to predict if the next is fraud.")
    cols = st.columns(5)
    amounts = [cols[i].number_input(f"Txn {i+1} ($)", 1.0, 10000.0,
               float([50, 80, 100, 60, 4500][i])) for i in range(5)]
    seq_num = st.slider("Transaction sequence number (position in session)", 1, 20, 5)

    if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
        try:
            scaler = load_scaler()
            model  = load_model()

            raw   = np.array([[a, seq_num] for a in amounts], dtype=np.float32)
            normed= scaler.transform(raw)
            seq   = normed.reshape(1, 5, 2)

            prob = float(model.predict(seq, verbose=0)[0][0])

            st.subheader("Result")
            if prob > 0.5:
                st.error(f"🚨 HIGH FRAUD RISK — {prob*100:.1f}% probability")
            elif prob > 0.25:
                st.warning(f"⚠️ MEDIUM RISK — {prob*100:.1f}% probability")
            else:
                st.success(f"✅ LOW RISK — {prob*100:.1f}% probability")
            st.progress(prob, text=f"Fraud probability: {prob*100:.1f}%")

            # Attention proxy
            arr  = np.array(amounts, dtype=float)
            attn = np.exp((arr - arr.mean()) / (arr.std() + 1e-9))
            attn = attn / attn.sum()
            top  = int(np.argmax(attn))
            st.subheader("Attention Map")
            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ["crimson" if i == top else "steelblue" for i in range(5)]
            ax.bar([f"Txn{i+1}" for i in range(5)], attn, color=colors, edgecolor="black")
            ax.set_title("Which transaction influenced prediction most (red = highest attention)")
            ax.set_ylabel("Attention Weight")
            st.pyplot(fig); plt.close()
            st.info(f"Transaction {top+1} (${amounts[top]:.0f}) had the highest attention weight.")

        except Exception as e:
            st.error(f"Error: {e}. Run  python train.py  first.")

# ── Tab 3: Positional Encoding ────────────────────────────────────────────────
with tab3:
    st.subheader("Positional Encoding — Transaction Order")
    PE = positional_encoding(seq_len=5, d_model=16)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(PE, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    ax.set_yticks(range(5)); ax.set_yticklabels([f"Txn {i+1}" for i in range(5)])
    ax.set_title("Sinusoidal Positional Encoding — 5 Transactions × 16 Dims")
    ax.set_xlabel("Encoding Dimension")
    st.pyplot(fig); plt.close()
    st.info(
        "Each transaction gets a unique position vector. "
        "Txn1 (first) and Txn5 (last) have different vectors even if amounts are identical. "
        "This lets the model detect escalation patterns (normal → normal → FRAUD)."
    )
