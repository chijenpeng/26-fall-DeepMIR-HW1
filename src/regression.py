"""Task A as an ordinal problem: (a) year/decade regression, (b) hierarchical coarse->fine.

  python src/regression.py --features mert --layers 6,8,10
Both are evaluated with the same top-1 / top-3 / confusion protocol as the classifiers.
"""
import argparse, json, numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from config import RESULTS, LABELS
from train import build_X
from utils import evaluate, neighbour_error_rate, dump


def regression(X, y, tr, va, alphas):
    best = None
    for al in alphas:
        reg = make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(X[tr], y[tr])
        yhat = reg.predict(X[va])
        proba = np.exp(-np.abs(yhat[:, None] - np.arange(6)[None]) / 0.5)   # closeness -> ranking
        proba /= proba.sum(1, keepdims=True)
        mae = float(np.abs(yhat - y[va]).mean())
        top1 = float((np.round(np.clip(yhat, 0, 5)) == y[va]).mean())
        if best is None or top1 > best[0]:
            best = (top1, al, proba, mae)
    return best


def hierarchical(X, y, tr, va, Cs):
    groups = np.array([0, 0, 1, 1, 2, 2])            # {60s,70s} {80s,90s} {00s,10s}
    g = groups[y]
    best = None
    for C in Cs:
        coarse = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C)).fit(X[tr], g[tr])
        pg = coarse.predict_proba(X[va])              # [n, 3]
        proba = np.zeros((va.sum(), 6))
        for gi in range(3):
            sel = tr & (g == gi)
            fine = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C)).fit(X[sel], y[sel])
            pf = fine.predict_proba(X[va])            # [n, 2] over the two decades of this group
            for j, cls in enumerate(fine.classes_):
                proba[:, cls] = pg[:, gi] * pf[:, j]
        top1 = float((proba.argmax(1) == y[va]).mean())
        if best is None or top1 > best[0]:
            best = (top1, C, proba)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="mert"); ap.add_argument("--layers", default="6,8,10")
    ap.add_argument("--mert_model", default="MERT-v1-95M"); ap.add_argument("--chunk_sec", type=int, default=5)
    ap.add_argument("--n_chunks", type=int, default=None); ap.add_argument("--no_std", action="store_true")
    ap.add_argument("--hc_subset", default=None); ap.add_argument("--lang_model", default="whisper-small")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(); a.dataset = "A"
    layers = [int(x) for x in a.layers.split(",")]
    X, meta = build_X(a, layers)
    labels = LABELS["A"]; lab2i = {l: i for i, l in enumerate(labels)}
    split = meta["split"]; tr, va = split == "train", split == "validation"
    y = np.array([lab2i.get(l, -1) for l in meta["label"]])
    name = a.name or f"A_{a.features}_L{'-'.join(map(str, layers))}"

    top1, al, proba, mae = regression(X, y, tr, va, [1, 10, 100, 1000, 10000])
    r = evaluate(proba, y[va], labels, f"{name} ridge regression", RESULTS / f"{name}_regression_cm.png")
    r.update(method="ridge_regression", alpha=al, mae_decades=mae, neighbour_error_rate=neighbour_error_rate(r["confusion_counts"]))
    dump(r, RESULTS / f"{name}_regression_val.json")
    print(f"regression  : top1={r['top1']:.3f} top3={r['top3']:.3f} MAE={mae:.2f} decades  alpha={al} nbr_err={r['neighbour_error_rate']:.2f}")

    top1, C, proba = hierarchical(X, y, tr, va, [0.001, 0.003, 0.01, 0.03])
    h = evaluate(proba, y[va], labels, f"{name} hierarchical", RESULTS / f"{name}_hier_cm.png")
    h.update(method="hierarchical_3x2", C=C, neighbour_error_rate=neighbour_error_rate(h["confusion_counts"]))
    dump(h, RESULTS / f"{name}_hier_val.json")
    print(f"hierarchical: top1={h['top1']:.3f} top3={h['top3']:.3f} C={C} nbr_err={h['neighbour_error_rate']:.2f}")
