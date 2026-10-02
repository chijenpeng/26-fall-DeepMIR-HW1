"""t-SNE of a feature set (train + validation), coloured by label.

  python src/tsne.py --dataset A --features mert --layers 6,8,10
"""
import argparse, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from config import RESULTS, LABELS
from train import build_X

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--features", default="mert"); ap.add_argument("--layers", default="6,8,10")
    ap.add_argument("--mert_model", default="MERT-v1-95M"); ap.add_argument("--chunk_sec", type=int, default=5)
    ap.add_argument("--n_chunks", type=int, default=None); ap.add_argument("--no_std", action="store_true")
    ap.add_argument("--hc_subset", default=None); ap.add_argument("--lang_model", default="whisper-small")
    ap.add_argument("--perplexity", type=float, default=30); ap.add_argument("--name", default=None)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    X, meta = build_X(a, layers)
    labels = LABELS[a.dataset]
    sel = meta["split"] != "test"
    Xs = StandardScaler().fit_transform(X[sel])
    Xp = PCA(n_components=min(50, Xs.shape[1]), random_state=0).fit_transform(Xs)
    Z = TSNE(n_components=2, perplexity=a.perplexity, init="pca", random_state=0).fit_transform(Xp)
    lab, spl = meta["label"][sel], meta["split"][sel]
    name = a.name or f"{a.dataset}_{a.features}_L{'-'.join(map(str, layers))}"
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = plt.get_cmap("viridis" if a.dataset == "A" else "tab10")
    for i, l in enumerate(labels):
        c = cmap(i / max(1, len(labels) - 1)) if a.dataset == "A" else cmap(i)
        m = (lab == l) & (spl == "train"); ax.scatter(Z[m, 0], Z[m, 1], s=12, color=c, alpha=.6, label=l)
        m = (lab == l) & (spl == "validation"); ax.scatter(Z[m, 0], Z[m, 1], s=40, color=c, marker="*", edgecolor="k", linewidth=.4)
    ax.legend(markerscale=2); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t-SNE  {name}  (dots = train, stars = validation)")
    plt.tight_layout(); out = RESULTS / f"{name}_tsne.png"; plt.savefig(out, dpi=130)
    np.savez(RESULTS / f"{name}_tsne.npz", Z=Z, label=lab, split=spl, sample_id=meta["sample_id"][sel])
    print("saved", out)
