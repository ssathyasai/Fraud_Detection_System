"""
train.py — run ALL tasks for Fraud Detection.
Usage:  python train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

BASE      = os.path.dirname(__file__)
DATA      = os.path.join(BASE, "data",      "fraud_detection_500.csv")
ARTIFACTS = os.path.join(BASE, "artifacts")
MODELS    = os.path.join(BASE, "models")
PLOTS     = os.path.join(BASE, "plots")

for d in [ARTIFACTS, MODELS, PLOTS]:
    os.makedirs(d, exist_ok=True)

from src.task1_business_understanding import run as biz
from src.task2_eda                    import run as eda
from src.task3_sequence_generation    import run as seq_gen
from src.task4_models                 import run as models
from src.task5_positional_encoding    import run as pos_enc
from src.task6_attention_investigation import run as attn_inv

if __name__ == "__main__":
    print("\n[1/6] Business Understanding")
    biz()

    print("\n[2/6] EDA")
    eda(DATA, PLOTS)

    print("\n[3/6] Sequence Generation")
    seq_gen(DATA, ARTIFACTS)

    print("\n[4/6] Build Models (Dense / LSTM / LSTM+Attention)")
    models(ARTIFACTS, MODELS)

    print("\n[5/6] Positional Encoding")
    pos_enc(PLOTS)

    print("\n[6/6] Attention Investigation")
    attn_inv(ARTIFACTS, MODELS, PLOTS)

    print("\n✅  All tasks complete.")
    print("   Launch dashboard:  streamlit run app.py")
