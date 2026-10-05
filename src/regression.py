"""Task 1 (dataset A) as an ordinal problem: (a) ridge regression on the decade index,
(b) hierarchical coarse->fine classification.

  python src/regression.py --features mert --layers 6,8,10

Both are evaluated with the same top-1 / top-3 / confusion protocol as the classifiers.
Reads   the feature files named by --features (see train.build_X).
Writes  results/<name>_regression_val.json, results/<name>_regression_cm.png,
        results/<name>_hier_val.json, results/<name>_hier_cm.png.
"""
import argparse, numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from config import RESULTS, LABELS
from train import add_feature_args, build_X
from utils import evaluate, neighbour_error_rate, dump, targets_and_split_masks


def regression(X, y, tr, va, alphas):
    """Ridge regression on the decade index, alpha selected by validation top-1 of the rounded prediction.
    Returns (top1, alpha, proba, mae); proba ranks the decades by closeness to the predicted value."""
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
    """Coarse classifier over three 20-year groups times one fine classifier per group,
    C (shared by all four) selected by validation top-1. Returns (top1, C, proba)."""
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
    add_feature_args(ap, features="mert", layers="6,8,10")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    a.dataset = "A"
    layers = [int(x) for x in a.layers.split(",")]
    X, meta = build_X(a, layers)
    labels = LABELS["A"]
    y, tr, va = targets_and_split_masks(meta, labels)
    name = a.name or f"A_{a.features}_L{'-'.join(map(str, layers))}"

    top1, al, proba, mae = regression(X, y, tr, va, [1, 10, 100, 1000, 10000])
    r = evaluate(proba, y[va], labels, f"{name} ridge regression", RESULTS / f"{name}_regression_cm.png")
    r.update(method="ridge_regression", alpha=al, mae_decades=mae, neighbour_error_rate=neighbour_error_rate(r["confusion_counts"]))
    dump(r, RESULTS / f"{name}_regression_val.json")
    print(f"regression  : top1={r['top1']:.3f} top3={r['top3']:.3f} MAE={mae:.2f} decades  alpha={al} nbr_err={r['neighbour_error_rate']:.2f}")

    top1, C, proba = hierarchical(X, y, tr, va, [0.0003, 0.001, 0.003, 0.01, 0.03])
    h = evaluate(proba, y[va], labels, f"{name} hierarchical", RESULTS / f"{name}_hier_cm.png")
    h.update(method="hierarchical_3x2", C=C, neighbour_error_rate=neighbour_error_rate(h["confusion_counts"]))
    dump(h, RESULTS / f"{name}_hier_val.json")
    print(f"hierarchical: top1={h['top1']:.3f} top3={h['top3']:.3f} C={C} nbr_err={h['neighbour_error_rate']:.2f}")
