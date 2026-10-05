"""Produce the submission JSON: top-3 labels per test sample for both datasets.

  python src/predict.py --model_A results/A_both_L7_logreg.joblib --model_B results/B_both_L7_logreg.joblib \
      --out <studentID>.json
Assumes features were extracted first (see README).
"""
import argparse, json, numpy as np, joblib
from types import SimpleNamespace
from config import LABELS
from train import build_X


def predict_ds(ds, model_path, split="test"):
    saved = joblib.load(model_path)
    a = SimpleNamespace(**saved["args"]); a.dataset = ds
    X, meta = build_X(a, saved["layers"])
    sel = meta["split"] == split
    proba = saved["clf"].predict_proba(X[sel])
    labels = LABELS[ds]
    return {sid: [labels[j] for j in np.argsort(-p)[:3]] for sid, p in zip(meta["sample_id"][sel], proba)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_A", required=True); ap.add_argument("--model_B", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"dataset_A": predict_ds("A", a.model_A), "dataset_B": predict_ds("B", a.model_B)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}: {len(out['dataset_A'])} A + {len(out['dataset_B'])} B test samples")
