"""Train a classifier on the train split, select on validation, save model + metrics.

Examples
  python src/train.py --dataset A --features handcrafted
  python src/train.py --dataset A --features mert --layers 7
  python src/train.py --dataset A --features mert --layers all        # per-layer sweep on val
  python src/train.py --dataset A --features both --layers 7 --clf svm
"""
import argparse, json, numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from config import RESULTS, LABELS
from features import load_handcrafted, load_mert, load_lang, mert_vector
from utils import evaluate, neighbour_error_rate, dump


def make_clf(kind, C):
    if kind == "logreg":
        return LogisticRegression(max_iter=5000, C=C)
    if kind == "svm":
        return SVC(C=C, kernel="rbf", probability=True)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=2000, early_stopping=True, random_state=0)
    raise ValueError(kind)


def build_X(a, layers):
    parts, meta = [], None
    blocks = {"both": {"handcrafted", "mert"}}.get(a.features, set(a.features.split("+")))
    if "handcrafted" in blocks:
        X, names, z = load_handcrafted(a.dataset)
        if a.hc_subset:
            keep = [names.index(n) for n in a.hc_subset.split(",")]
            X = X[:, keep]
        parts.append(X); meta = z
    if "mert" in blocks:
        z = load_mert(a.dataset, model=a.mert_model, chunk_sec=a.chunk_sec)
        parts.append(mert_vector(z, layers, n_chunks=a.n_chunks, use_std=not a.no_std))
        if meta is not None:
            assert (meta["sample_id"] == z["sample_id"]).all()
        meta = z
    if "lang" in blocks:
        z = load_lang(a.dataset, model=a.lang_model)
        if meta is not None:
            assert (meta["sample_id"] == z["sample_id"]).all()
        parts.append(np.log(z["probs"] + 1e-6)); meta = z
    return np.concatenate(parts, 1), meta


def run(a, layers, tag):
    X, meta = build_X(a, layers)
    labels = LABELS[a.dataset]; lab2i = {l: i for i, l in enumerate(labels)}
    split, lab = meta["split"], meta["label"]
    tr, va = split == "train", split == "validation"
    ytr = np.array([lab2i[l] for l in lab[tr]]); yva = np.array([lab2i[l] for l in lab[va]])
    best = None
    for C in a.C:
        clf = make_pipeline(StandardScaler(), make_clf(a.clf, C)).fit(X[tr], ytr)
        p = clf.predict_proba(X[va])
        res = evaluate(p, yva, labels, f"{a.dataset} {tag} C={C}")
        if best is None or res["top1"] > best[1]["top1"]:
            best = (clf, res, C)
    clf, res, C = best
    res.update(dict(dataset=a.dataset, features=a.features, layers=layers, C=C, clf=a.clf, dim=int(X.shape[1]),
                    n_chunks=a.n_chunks, neighbour_error_rate=neighbour_error_rate(res["confusion_counts"])))
    return clf, res, X, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--features", default="both", help="'both' or '+'-joined blocks from {handcrafted, mert, lang}")
    ap.add_argument("--lang_model", default="whisper-small")
    ap.add_argument("--layers", default="7", help="comma list, or 'all' to sweep every layer one at a time")
    ap.add_argument("--mert_model", default="MERT-v1-95M")
    ap.add_argument("--chunk_sec", type=int, default=5)
    ap.add_argument("--n_chunks", type=int, default=None, help="use only first k chunks (5 s each)")
    ap.add_argument("--no_std", action="store_true")
    ap.add_argument("--hc_subset", default=None, help="comma list of handcrafted feature names")
    ap.add_argument("--clf", default="logreg", choices=["logreg", "svm", "mlp"])
    ap.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if a.layers == "all":
        z = load_mert(a.dataset, model=a.mert_model, chunk_sec=a.chunk_sec); L = z["mean"].shape[2]
        sweep = []
        for l in range(L):
            _, res, _, _ = run(a, [l], f"layer{l}")
            sweep.append(dict(layer=l, top1=res["top1"], top3=res["top3"], C=res["C"]))
            print(f"layer {l:2d}: top1={res['top1']:.3f} top3={res['top3']:.3f} (C={res['C']})")
        dump(sweep, RESULTS / f"{a.dataset}_{a.features}_layer_sweep.json")
    else:
        layers = [int(x) for x in a.layers.split(",")]
        name = a.name or f"{a.dataset}_{a.features}_L{'-'.join(map(str, layers))}_{a.clf}"
        clf, res, X, meta = run(a, layers, name)
        evaluate(clf.predict_proba(X[meta["split"] == "validation"]),
                 np.array([LABELS[a.dataset].index(l) for l in meta["label"][meta["split"] == "validation"]]),
                 LABELS[a.dataset], name, RESULTS / f"{name}_cm.png")
        joblib.dump(dict(clf=clf, args=vars(a), layers=layers), RESULTS / f"{name}.joblib")
        dump(res, RESULTS / f"{name}_val.json")
        print(json.dumps({k: v for k, v in res.items() if k != "confusion_counts"}, indent=1))
        print("confusion (rows=true):"); print(np.array(res["confusion_counts"]))
