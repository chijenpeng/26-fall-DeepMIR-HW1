"""Clean single-panel figures for the slides -> slides/figs/.   python src/make_slide_figs.py"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import ROOT, RESULTS

OUT = ROOT / "slides" / "figs"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def confusion(name, out):
    r = json.load(open(RESULTS / f"{name}_val.json")); cm = np.array(r["confusion_counts"]); L = r["labels"]
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(L)):
        for j in range(len(L)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if norm[i, j] > 0.5 else "#222", fontweight="bold" if i == j else "normal")
    fs = 8.5 if max(map(len, L)) > 5 else 9.5
    ax.set_xticks(range(len(L))); ax.set_xticklabels(L, fontsize=fs); ax.set_yticks(range(len(L))); ax.set_yticklabels(L, fontsize=fs)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    plt.tight_layout(pad=0.3); plt.savefig(OUT / out, bbox_inches="tight", pad_inches=0.04); plt.close()
    return r


if __name__ == "__main__":
    a = confusion("A_wt_L1_m330", "cm_task1.pdf"); b = confusion("B_wl_L23-30_mert7_lang", "cm_task2.pdf")
    print("task1", round(a["top1"], 3), round(a["top3"], 3), "| task2", round(b["top1"], 3), round(b["top3"], 3))
    # low-end trend: median and inter-quartile range of two bands per decade (training split)
    import pandas as pd
    df = pd.read_csv(ROOT / "eda" / "trainA_spectral_feats.csv"); decs = sorted(df.label.unique()); x = np.arange(len(decs))
    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    for col, name, c in [("bass_60_120", "60–120 Hz (bass)", "#4c72b0"), ("sub_20_60", "20–60 Hz (sub-bass)", "#dd8452")]:
        g = df.groupby("label")[col]; med = g.median()[decs].values
        ax.fill_between(x, g.quantile(.25)[decs].values, g.quantile(.75)[decs].values, color=c, alpha=.18, lw=0)
        ax.plot(x, med, "-o", color=c, label=name, lw=2, ms=5)
    ax.set_xticks(x); ax.set_xticklabels(decs, fontsize=9.5); ax.set_ylabel("Band energy (dB re. total)")
    ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=9.5, loc="lower right"); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    plt.tight_layout(pad=0.3); plt.savefig(OUT / "lowend_trend.pdf", bbox_inches="tight", pad_inches=0.04); plt.close()

    # input-length ablation with the best models
    fig, ax = plt.subplots(figsize=(3.9, 3.5)); secs = [5, 10, 15, 30]
    for ds, name, c in [("A", "Task 1 (decade)", "#4c72b0"), ("B", "Task 2 (market)", "#dd8452")]:
        r = [json.load(open(RESULTS / f"{ds}_best_{t}s_val.json")) for t in secs]
        ax.plot(secs, [100 * x["top1"] for x in r], "-o", color=c, lw=2, ms=5, label=f"{name}, top-1")
        ax.plot(secs, [100 * x["top3"] for x in r], "--s", color=c, lw=1.5, ms=4, alpha=.75, label=f"{name}, top-3")
    ax.set_xticks(secs); ax.set_xlabel("Input length (s)"); ax.set_ylabel("Validation accuracy (%)"); ax.set_ylim(35, 95)
    ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=1.2, handlelength=2.4)
    [ax.spines[k].set_visible(False) for k in ("top", "right")]
    plt.tight_layout(pad=0.3); plt.savefig(OUT / "duration_trend.pdf", bbox_inches="tight", pad_inches=0.04); plt.close()

    # t-SNE panels: crop the matplotlib title off the saved PNGs
    from PIL import Image
    for src, dst in [("A_best_logits_tsne.png", "tsne_task1_logits.png"), ("B_best_tsne.png", "tsne_task2_features.png")]:
        im = Image.open(RESULTS / src); w, h = im.size
        im.crop((0, int(h * 0.055), w, h)).save(OUT / dst)

