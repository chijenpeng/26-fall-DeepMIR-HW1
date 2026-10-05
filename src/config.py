"""Paths and label definitions shared by all scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = {"A": ROOT / "dataset_A", "B": ROOT / "dataset_B"}
FEATURES = Path(os.environ.get("HW1_FEATURES", ROOT / "features"))   # inference.py points this at its own work dir
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
LABELS = {
    "A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    "B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
}
SR = 24000
MERT_MODEL = "m-a-p/MERT-v1-95M"
CHUNK_SEC = 5          # MERT was trained on 5 s inputs; 30 s excerpt -> 6 chunks


def load_manifest(ds, data_root=None):
    """Read <root>/manifest.csv (sample_id, split, label, audio_path, ...).

    If the folder has no manifest, every .wav below it is treated as an unlabeled test clip.
    Set HW1_ONLY_SPLIT=test to keep a single split (used by inference.py)."""
    import pandas as pd
    root = Path(data_root) if data_root else DATA[ds]
    if (root / "manifest.csv").exists():
        m = pd.read_csv(root / "manifest.csv")
    else:
        wavs = sorted(root.rglob("*.wav"))
        m = pd.DataFrame({"sample_id": [w.stem for w in wavs], "split": "test", "label": "",
                          "audio_path": [str(w.relative_to(root)) for w in wavs]})
    only = os.environ.get("HW1_ONLY_SPLIT")
    if only:
        m = m[m["split"] == only].reset_index(drop=True)
    m["path"] = [str(root / p) for p in m.audio_path]
    return m
