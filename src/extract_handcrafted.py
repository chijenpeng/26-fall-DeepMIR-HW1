"""Extract hand-crafted features for every file in a dataset -> features/<ds>_handcrafted.npz"""
import argparse, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from config import FEATURES, load_manifest
from handcrafted import extract

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--data_root", default=None, help="override dataset folder (for TA inference)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = load_manifest(a.dataset, a.data_root)
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(extract, m.path.tolist(), chunksize=16))
    df = pd.DataFrame(rows)
    out = a.out or FEATURES / f"{a.dataset}_handcrafted.npz"
    FEATURES.mkdir(exist_ok=True)
    np.savez(out, X=df.values.astype(np.float32), names=np.array(df.columns), sample_id=m.sample_id.values,
             split=m.split.values, label=m.label.fillna("").values)
    print(f"saved {out}: X={df.shape}")
