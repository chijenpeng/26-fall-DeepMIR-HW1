"""Evaluation helpers shared by the training scripts: top-k accuracy, confusion matrix (+ plot),
label / split bookkeeping and JSON output. Reads nothing; writes only the paths it is given."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix


def topk(proba, y, k):
    """Share of rows whose true class index y[i] is among the k most probable classes."""
    return float(np.mean([y[i] in np.argsort(-proba[i])[:k] for i in range(len(y))]))


def evaluate(proba, y, labels, title, out_png=None):
    """Top-1 / top-3 accuracy and confusion counts (rows = true class) of the probabilities `proba`.

    With `out_png`, also saves a two-panel confusion-matrix figure (counts, row-normalized)."""
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
            a.set_xlabel("predicted")
            a.set_ylabel("true")
        plt.tight_layout()
        plt.savefig(out_png, dpi=130)
        plt.close()
    return res


def neighbour_error_rate(cm):
    """Share of errors that fall in an adjacent class (ordinal labels only)."""
    cm = np.asarray(cm)
    off = cm.sum() - np.trace(cm)
    adj = sum(cm[i, j] for i in range(len(cm)) for j in range(len(cm)) if abs(i - j) == 1)
    return float(adj / off) if off else float("nan")


def label_indices(label_strings, labels):
    """Class index of every label string; -1 for anything not in `labels` (the unlabeled test clips)."""
    lab2i = {l: i for i, l in enumerate(labels)}
    return np.array([lab2i.get(l, -1) for l in label_strings])


def targets_and_split_masks(meta, labels):
    """(y, train_mask, validation_mask) for a feature file or the `meta` returned by train.build_X.

    y holds the class index of every row (-1 where unlabeled); the masks are boolean arrays over the rows."""
    y = label_indices(meta["label"], labels)
    split = meta["split"]
    return y, split == "train", split == "validation"


def dump(obj, path):
    """Write `obj` to `path` as indented JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
