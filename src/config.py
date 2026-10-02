"""Paths and label definitions shared by all scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = {"A": ROOT / "dataset_A", "B": ROOT / "dataset_B"}
FEATURES = ROOT / "features"
RESULTS = ROOT / "results"
LABELS = {
    "A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    "B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
}
SR = 24000
MERT_MODEL = "m-a-p/MERT-v1-95M"
CHUNK_SEC = 5          # MERT was trained on 5 s inputs; 30 s excerpt -> 6 chunks


def load_manifest(ds, data_root=None):
    import pandas as pd
    root = Path(data_root) if data_root else DATA[ds]
    m = pd.read_csv(root / "manifest.csv")
    m["path"] = [str(root / p) for p in m.audio_path]
    return m
