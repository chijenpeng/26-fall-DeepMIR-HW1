"""Clean single-panel figures for the slides -> slides/figs/.   python src/make_slide_figs.py

Reads   results/*_val.json, layer-sweep, fine-tuning-history and ALM JSON files, two t-SNE PNGs from
        results/, and eda/trainA_spectral_feats.csv.
Writes  slides/figs/*.pdf and slides/figs/tsne_*.png.
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT, RESULTS

OUT = ROOT / "slides" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def confusion(name, out, cm=None, L=None):
    """Single-panel confusion matrix -> slides/figs/<out>. Counts and labels come from
    results/<name>_val.json (which is returned), or from `cm` / `L` when given (returns None)."""
    r = None
    if cm is None:
        r = json.load(open(RESULTS / f"{name}_val.json"))
        cm = np.array(r["confusion_counts"])
        L = r["labels"]
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(L)):
        for j in range(len(L)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if norm[i, j] > 0.5 else "#222", fontweight="bold" if i == j else "normal")
    fs = 8.5 if max(map(len, L)) > 5 else 9.5
    ax.set_xticks(range(len(L)))
    ax.set_xticklabels(L, fontsize=fs)
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels(L, fontsize=fs)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(length=0)
    [s.set_visible(False) for s in ax.spines.values()]
    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / out, bbox_inches="tight", pad_inches=0.04)
    plt.close()
    return r


if __name__ == "__main__":
    a = confusion("A_wt_L1_m330", "cm_task1.pdf")
    b = confusion("B_wl_L23-30_mert7_lang", "cm_task2.pdf")
    confusion("A_best_regression", "cm_task1_regression.pdf")       # ridge regression on the decade index, final features
    print("task1", round(a["top1"], 3), round(a["top3"], 3), "| task2", round(b["top1"], 3), round(b["top3"], 3))
    # low-end trend: median and inter-quartile range of two bands per decade (training split)
    import pandas as pd
    df = pd.read_csv(ROOT / "eda" / "trainA_spectral_feats.csv")
    decs = sorted(df.label.unique())
    x = np.arange(len(decs))
    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    for col, name, c in [("bass_60_120", "60–120 Hz (bass)", "#4c72b0"), ("sub_20_60", "20–60 Hz (sub-bass)", "#dd8452")]:
        g = df.groupby("label")[col]
        med = g.median()[decs].values
        ax.fill_between(x, g.quantile(.25)[decs].values, g.quantile(.75)[decs].values, color=c, alpha=.18, lw=0)
        ax.plot(x, med, "-o", color=c, label=name, lw=2, ms=5)
    ax.set_xticks(x)
    ax.set_xticklabels(decs, fontsize=9.5)
    ax.set_ylabel("Band energy (dB re. total)")
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    [ax.spines[k].set_visible(False) for k in ("top", "right")]
    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / "lowend_trend.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close()

    # input-length ablation with the best models
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    secs = [5, 10, 15, 30]
    for ds, name, c in [("A", "Task 1 (decade)", "#4c72b0"), ("B", "Task 2 (market)", "#dd8452")]:
        r = [json.load(open(RESULTS / f"{ds}_best_{t}s_val.json")) for t in secs]
        ax.plot(secs, [100 * x["top1"] for x in r], "-o", color=c, lw=2, ms=5, label=f"{name}, top-1")
        ax.plot(secs, [100 * x["top3"] for x in r], "--s", color=c, lw=1.5, ms=4, alpha=.75, label=f"{name}, top-3")
    ax.set_xticks(secs)
    ax.set_xlabel("Input length (s)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(35, 95)
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=1.2, handlelength=2.4)
    [ax.spines[k].set_visible(False) for k in ("top", "right")]
    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / "duration_trend.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close()

    # t-SNE panels: crop the matplotlib title off the saved PNGs
    from PIL import Image
    for src, dst in [("A_best_logits_tsne.png", "tsne_task1_logits.png"), ("B_best_tsne.png", "tsne_task2_features.png")]:
        im = Image.open(RESULTS / src)
        w, h = im.size
        im.crop((0, int(h * 0.055), w, h)).save(OUT / dst)

    # ---------------- ablation: per-layer sweeps (relative depth) ----------------
    from config import LABELS
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)
    enc = [("Whisper-tiny", "{ds}_mert_whisper-tiny_layer_sweep", "#dd8452", "-"),
           ("Whisper-large-v3", "{ds}_mert_whisper-large-v3_layer_sweep", "#c44e52", "-"),
           ("MERT-v1-95M", "{ds}_mert_layer_sweep", "#4c72b0", "--"),
           ("MERT-v1-330M", "{ds}_mert_MERT-v1-330M_layer_sweep", "#55a868", "--")]
    used = {"A": {"Whisper-tiny": [1], "MERT-v1-330M": [5, 6, 14]}, "B": {"Whisper-large-v3": [23, 30], "MERT-v1-95M": [7]}}
    for ax, ds, title in zip(axes, "AB", ["Task 1: decade", "Task 2: market"]):
        for name, pat, c, ls in enc:
            sw = json.load(open(RESULTS / (pat.format(ds=ds) + ".json")))
            n = len(sw) - 1
            x = [d["layer"] / n for d in sw]
            yv = [100 * d["top1"] for d in sw]
            ax.plot(x, yv, ls, color=c, lw=1.8, label=name)
            for l in used[ds].get(name, []):
                ax.plot(l / n, yv[l], "*", color=c, ms=13, mec="k", mew=0.6, zorder=5)
        ax.axhline(100 / 6, color="gray", lw=0.8, ls=":")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Relative depth (0 = input, 1 = last layer)")
        ax.grid(alpha=.3)
        [ax.spines[k].set_visible(False) for k in ("top", "right")]
    axes[0].set_ylabel("Validation top-1 (%)")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), columnspacing=1.5)
    plt.tight_layout(pad=0.4)
    plt.savefig(OUT / "layer_sweep.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close()

    # ---------------- ablation: fine-tuning curves (MERT-v1-95M, Task 1, 40 epochs) ----------------
    h = json.load(open(RESULTS / "ft40_A_history.json"))["history"]
    ep = [d["epoch"] for d in h]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    axes[0].plot(ep, [d["loss"] for d in h], color="#4c72b0", lw=2, label="train loss")
    axes[0].plot(ep, [d["val_loss"] for d in h], color="#dd8452", lw=2, label="validation loss")
    axes[0].axvline(11, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(ep, [100 * d["val_top1"] for d in h], color="#dd8452", lw=2, label="fine-tuned, validation top-1")
    axes[1].axhline(48.5, color="#4c72b0", ls="--", lw=1.5, label="frozen MERT-v1-95M (48.5)")
    axes[1].axhline(52.3, color="#55a868", ls="--", lw=1.5, label="final frozen model (52.3)")
    axes[1].axvline(11, color="gray", ls=":", lw=1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.grid(alpha=.3)
        [ax.spines[k].set_visible(False) for k in ("top", "right")]
    plt.tight_layout(pad=0.4)
    plt.savefig(OUT / "finetune_curves.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close()

    # ---------------- ALM confusion matrices (naive prompt, log-prob ranking) ----------------
    for ds, out in [("A", "cm_alm_task1.pdf"), ("B", "cm_alm_task2.pdf")]:
        m = json.load(open(RESULTS / f"alm_Qwen2-Audio-7B-Instruct_{ds}_naive.json"))["metrics"]
        confusion(None, out, cm=m["confusion_counts"], L=LABELS[ds])
    # one confusion matrix per prompt design
    for ds, prompts in [("A", ["naive", "cues", "production"]), ("B", ["naive", "cues"])]:
        for pr in prompts:
            m = json.load(open(RESULTS / f"alm_Qwen2-Audio-7B-Instruct_{ds}_{pr}.json"))["metrics"]
            confusion(None, f"cm_alm_{ds}_{pr}.pdf", cm=m["confusion_counts"], L=LABELS[ds])
