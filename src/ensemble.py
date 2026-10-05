"""Average the probabilities of a frozen-feature classifier with those of a fine-tuned encoder.

  python src/ensemble.py --dataset A --frozen results/A_wt_L1_m330.joblib --layers 7 \
      --features "mert@features/A_mert_whisper-tiny_30s.npz:1+mert@features/A_mert_MERT-v1-330M_5s.npz:5,6,14" \
      --finetuned results/ft40_A_proba.npz --weights 0.3 0.5

Reads the classifier saved by train.py, the feature files named by --features and the
<name>_proba.npz written by finetune_mert.py / finetune_whisper.py.
Writes results/ensemble_<dataset>_<finetuned name>.json (validation top-1 / top-3 per weight).
"""
import argparse
from pathlib import Path
import numpy as np, joblib
from config import RESULTS, LABELS
from train import add_feature_args, build_X
from utils import topk, dump

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--frozen", required=True, help="joblib written by train.py")
    add_feature_args(ap, features="mert", layers="7")
    ap.add_argument("--finetuned", required=True, help="<name>_proba.npz written by a fine-tuning script")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.3, 0.5], help="weight of the fine-tuned model")
    a = ap.parse_args()
    labels = LABELS[a.dataset]
    X, meta = build_X(a, [int(x) for x in a.layers.split(",")])
    va = meta["split"] == "validation"
    ids = meta["sample_id"][va]
    y = np.array([labels.index(l) for l in meta["label"][va]])
    p_frozen = joblib.load(a.frozen)["clf"].predict_proba(X[va])
    z = np.load(a.finetuned, allow_pickle=True)
    by_id = dict(zip(z["sample_id"], z["proba"]))
    p_ft = np.stack([by_id[i] for i in ids])
    name = Path(a.finetuned).name.replace("_proba.npz", "")
    rows = [dict(model="frozen", top1=topk(p_frozen, y, 1), top3=topk(p_frozen, y, 3)),
            dict(model="finetuned", top1=topk(p_ft, y, 1), top3=topk(p_ft, y, 3))]
    for w in a.weights:
        p = (1 - w) * p_frozen + w * p_ft
        rows.append(dict(model=f"average, finetuned weight {w}", top1=topk(p, y, 1), top3=topk(p, y, 3)))
    for r in rows:
        print(f"{r['model']:32s} top1={r['top1']:.3f} top3={r['top3']:.3f}")
    dump(dict(dataset=a.dataset, frozen=a.frozen, finetuned=name, rows=rows,
              agreement=float(np.mean(p_frozen.argmax(1) == p_ft.argmax(1)))), RESULTS / f"ensemble_{a.dataset}_{name}.json")
