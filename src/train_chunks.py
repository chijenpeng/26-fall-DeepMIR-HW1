"""Chunk-level training: every 5 s chunk of a training recording is its own sample
(6x more training data); at evaluation the chunk probabilities of a recording are averaged.

  python src/train_chunks.py --dataset A --layers 6,8,10

Reads   features/<ds>_mert_<mert_model>_5s.npz (or --mert_file).
Writes  results/<name>.joblib (dict: clf, layers, chunk_level, args), results/<name>_val.json,
        results/<name>_cm.png.
"""
import argparse, numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from config import RESULTS, LABELS
from features import load_mert
from utils import evaluate, neighbour_error_rate, dump, targets_and_split_masks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--layers", default="6,8,10")
    ap.add_argument("--mert_model", default="MERT-v1-95M")
    ap.add_argument("--mert_file", default=None)
    ap.add_argument("--no_std", action="store_true")
    ap.add_argument("--C", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03])
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    z = load_mert(a.dataset, path=a.mert_file, model=a.mert_model)
    mean, std = z["mean"].astype(np.float32)[:, :, layers], z["std"].astype(np.float32)[:, :, layers]   # [N, 6, L, D]
    N, K = mean.shape[:2]
    Xc = mean.reshape(N, K, -1) if a.no_std else np.concatenate([mean.reshape(N, K, -1), std.reshape(N, K, -1)], 2)
    labels = LABELS[a.dataset]
    y, tr, va = targets_and_split_masks(z, labels)
    Xtr = Xc[tr].reshape(-1, Xc.shape[2])
    ytr = np.repeat(y[tr], K)
    best = None
    for C in a.C:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C)).fit(Xtr, ytr)
        p = clf.predict_proba(Xc[va].reshape(-1, Xc.shape[2])).reshape(va.sum(), K, -1).mean(1)
        res = evaluate(p, y[va], labels, "")
        if best is None or res["top1"] > best[1]["top1"]:
            best = (clf, res, C)
    clf, res, C = best
    name = a.name or f"{a.dataset}_chunks_L{'-'.join(map(str, layers))}"
    p = clf.predict_proba(Xc[va].reshape(-1, Xc.shape[2])).reshape(va.sum(), K, -1).mean(1)
    res = evaluate(p, y[va], labels, name, RESULTS / f"{name}_cm.png")
    res.update(dict(dataset=a.dataset, features=f"mert chunks L{layers}", C=C, dim=int(Xc.shape[2]),
                    n_train_chunks=int(len(Xtr)), neighbour_error_rate=neighbour_error_rate(res["confusion_counts"])))
    joblib.dump(dict(clf=clf, layers=layers, chunk_level=True, args=vars(a)), RESULTS / f"{name}.joblib")
    dump(res, RESULTS / f"{name}_val.json")
    print(f"{name}: top1={res['top1']:.3f} top3={res['top3']:.3f} C={C} train_chunks={len(Xtr)}")
