"""Task 4: Build & Compare — Dense  vs  LSTM  vs  LSTM+Attention"""
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, LSTM, Dropout, Input, Flatten,
                                     LayerNormalization, MultiHeadAttention,
                                     GlobalAveragePooling1D)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

def run(artifacts_dir="artifacts", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    X = np.load(f"{artifacts_dir}/X_seq.npy")
    y = np.load(f"{artifacts_dir}/y_seq.npy")
    SEQ_LEN, N_FEAT = X.shape[1], X.shape[2]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None)

    classes = np.unique(y_tr)
    if len(classes) > 1:
        cw = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight = dict(enumerate(cw))
    else:
        class_weight = {0: 1.0, 1: 10.0}
    print(f"Class weights: {class_weight}")

    def eval_model(m, name):
        p = (m.predict(X_te, verbose=0) > 0.5).astype(int).flatten()
        print(f"\n{'='*55}")
        print(f"  {name}")
        print('='*55)
        print(classification_report(y_te, p,
              target_names=["Legitimate", "Fraud"], zero_division=0))

    # ── Model A: Dense ───────────────────────────────────────────────
    mA = Sequential([
        Flatten(input_shape=(SEQ_LEN, N_FEAT)),
        Dense(64, activation="relu"), Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    mA.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    mA.fit(X_tr, y_tr, epochs=20, batch_size=32,
           class_weight=class_weight, verbose=0)
    mA.save(f"{models_dir}/model_dense.h5")
    eval_model(mA, "Model A — Dense Network")

    # ── Model B: LSTM ────────────────────────────────────────────────
    mB = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, N_FEAT)),
        Dropout(0.3), Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    mB.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    mB.fit(X_tr, y_tr, epochs=20, batch_size=32,
           class_weight=class_weight, verbose=0)
    mB.save(f"{models_dir}/model_lstm.h5")
    eval_model(mB, "Model B — LSTM")

    # ── Model C: LSTM + Attention ────────────────────────────────────
    inp  = Input(shape=(SEQ_LEN, N_FEAT))
    x    = LSTM(64, return_sequences=True)(inp)
    attn, _ = MultiHeadAttention(num_heads=2, key_dim=16)(x, x,
                                                           return_attention_scores=True)
    x    = LayerNormalization()(x + attn)
    x    = GlobalAveragePooling1D()(x)
    x    = Dense(32, activation="relu")(x)
    x    = Dropout(0.3)(x)
    out  = Dense(1, activation="sigmoid")(x)
    mC   = Model(inp, out)
    mC.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    mC.fit(X_tr, y_tr, epochs=20, batch_size=32,
           class_weight=class_weight, verbose=0)
    mC.save(f"{models_dir}/model_lstm_attention.h5")
    eval_model(mC, "Model C — LSTM + Attention")

    print(f"\nAll models saved → {models_dir}/")
    return mA, mB, mC

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
