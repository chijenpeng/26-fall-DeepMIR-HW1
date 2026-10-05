"""End-to-end inference: audio folders in, top-3 predictions out.

    python src/inference.py --data_A /path/to/dataset_A --data_B /path/to/dataset_B --out r14725022.json

Each data folder is expected to look like the released datasets (manifest.csv + audio/*.wav);
without a manifest every .wav below the folder is used. By default only rows with split == "test"
are predicted (use --split all for every clip).

Steps (all with the released checkpoints in checkpoints/best/):
  Task 1  Whisper-tiny encoder layer 1  +  MERT-v1-330M layers 5, 6, 14      -> logistic regression
  Task 2  Whisper-medium encoder layer 12  +  MERT-v1-330M layer 8  +  Whisper-small language ID -> logistic regression
Frame features are pooled with mean + std over time. The four pretrained models are downloaded
from the Hugging Face Hub on first use.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def run(script, *args, env):
    """Run one src/ script as a subprocess from the repository root; raise if it fails."""
    cmd = [sys.executable, str(SRC / script), *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_A", default=None, help="Task 1 data folder (release-decade)")
    ap.add_argument("--data_B", default=None, help="Task 2 data folder (release-market)")
    ap.add_argument("--out", required=True, help="output JSON, e.g. r14725022.json")
    ap.add_argument("--split", default="test", help='manifest split to predict, or "all"')
    ap.add_argument("--work_dir", default=str(ROOT / "features_inference"), help="where extracted features are cached")
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints" / "best"))
    ap.add_argument("--skip_extract", action="store_true", help="reuse features already in --work_dir")
    a = ap.parse_args()
    assert a.data_A or a.data_B, "give --data_A and/or --data_B"
    work = Path(a.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HW1_FEATURES=str(work), PYTORCH_ENABLE_MPS_FALLBACK="1")
    if a.split != "all":
        env["HW1_ONLY_SPLIT"] = a.split
    else:
        env.pop("HW1_ONLY_SPLIT", None)

    if not a.skip_extract:
        if a.data_A:
            dA = str(Path(a.data_A).resolve())
            run("extract_whisper.py", "--dataset", "A", "--size", "tiny", "--data_root", dA, env=env)
            run("extract_mert.py", "--dataset", "A", "--model", "m-a-p/MERT-v1-330M", "--data_root", dA, env=env)
        if a.data_B:
            dB = str(Path(a.data_B).resolve())
            run("extract_whisper.py", "--dataset", "B", "--size", "medium", "--data_root", dB, env=env)
            run("extract_mert.py", "--dataset", "B", "--model", "m-a-p/MERT-v1-330M", "--data_root", dB, env=env)
            run("extract_lang.py", "--dataset", "B", "--data_root", dB, env=env)

    # ---- classification (same feature-assembly code as training) ----
    os.environ["HW1_FEATURES"] = str(work)            # read by config.py at import time
    sys.path.insert(0, str(SRC))
    import numpy as np, joblib
    from config import LABELS
    from train import build_X, feature_spec

    spec = {
        "A": dict(ckpt="task1_decade.joblib",
                  features=f"mert@{work}/A_mert_MERT-v1-330M_5s.npz:5,6,14+mert@{work}/A_mert_whisper-tiny_30s.npz:1"),
        "B": dict(ckpt="task2_market.joblib",
                  features=f"mert@{work}/B_mert_whisper-medium_30s.npz:12+mert@{work}/B_mert_MERT-v1-330M_5s.npz:8+lang"),
    }
    out = {}
    for ds, given in (("A", a.data_A), ("B", a.data_B)):
        if not given:
            continue
        X, meta = build_X(feature_spec(ds, spec[ds]["features"]), [7])   # [7] = layers of the default `mert` block, unused here: every block carries its own layers
        clf = joblib.load(Path(a.ckpt_dir) / spec[ds]["ckpt"])["clf"]
        proba = clf.predict_proba(X)
        sel = np.ones(len(X), bool) if a.split == "all" else (meta["split"] == a.split)
        labels = LABELS[ds]
        out[f"dataset_{ds}"] = {str(sid): [labels[j] for j in np.argsort(-p)[:3]]
                                for sid, p in zip(meta["sample_id"][sel], proba[sel])}
        print(f"dataset_{ds}: {int(sel.sum())} clips, feature dim {X.shape[1]}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
