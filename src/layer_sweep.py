"""Fast per-layer sweep: load the MERT file once, fit one logreg per layer.
  python src/layer_sweep.py --dataset A --mert_model MERT-v1-330M --C 0.003 0.01
Prints 'layer k: top1=... top3=... (C=...)' lines (same format train.py --layers all uses).
"""
import argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from config import RESULTS, LABELS
from features import load_mert
from utils import topk, dump

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--mert_model", default="MERT-v1-95M")
    ap.add_argument("--chunk_sec", type=int, default=5); ap.add_argument("--C", type=float, nargs="+", default=[0.003, 0.01])
    a = ap.parse_args()
    z = load_mert(a.dataset, model=a.mert_model, chunk_sec=a.chunk_sec)
    M = z["mean"].astype(np.float32).mean(1); S = z["std"].astype(np.float32).mean(1)      # [N, L, D]
    labels = LABELS[a.dataset]; lab2i = {l: i for i, l in enumerate(labels)}
    y = np.array([lab2i.get(l, -1) for l in z["label"]]); split = z["split"]
    tr, va = split == "train", split == "validation"
    sweep = []
    for l in range(M.shape[1]):
        X = np.concatenate([M[:, l], S[:, l]], 1); best = None
        for C in a.C:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C)).fit(X[tr], y[tr])
            p = clf.predict_proba(X[va]); r = (topk(p, y[va], 1), topk(p, y[va], 3), C)
            if best is None or r[0] > best[0]:
                best = r
        sweep.append(dict(layer=l, top1=best[0], top3=best[1], C=best[2]))
        print(f"layer {l:2d}: top1={best[0]:.3f} top3={best[1]:.3f} (C={best[2]})", flush=True)
    dump(sweep, RESULTS / f"{a.dataset}_mert_{a.mert_model}_layer_sweep.json")
