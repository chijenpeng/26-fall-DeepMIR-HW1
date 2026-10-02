import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix


def topk(proba, y, k):
    return float(np.mean([y[i] in np.argsort(-proba[i])[:k] for i in range(len(y))]))


def evaluate(proba, y, labels, title, out_png=None):
    pred = proba.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(labels)))
    res = {"top1": topk(proba, y, 1), "top3": topk(proba, y, 3), "confusion_counts": cm.tolist(), "labels": labels}
    if out_png:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax[0], cbar=False)
        ax[0].set_title(f"{title}\ncounts  (top1={res['top1']:.3f}, top3={res['top3']:.3f})")
        sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", xticklabels=labels,
                    yticklabels=labels, ax=ax[1], cbar=False, vmin=0, vmax=1)
        ax[1].set_title("row-normalized (recall)")
        for a in ax:
            a.set_xlabel("predicted"); a.set_ylabel("true")
        plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
    return res


def neighbour_error_rate(cm):
    """Share of errors that fall in an adjacent class (ordinal labels only)."""
    cm = np.asarray(cm); off = cm.sum() - np.trace(cm)
    adj = sum(cm[i, j] for i in range(len(cm)) for j in range(len(cm)) if abs(i - j) == 1)
    return float(adj / off) if off else float("nan")


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
