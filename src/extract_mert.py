"""Extract MERT hidden states for every file -> features/<ds>_mert_<model>_<chunk_sec>s.npz

Each 30 s excerpt is cut into CHUNK_SEC chunks; for every chunk and every transformer layer we
store the time-mean and time-std of the frame embeddings (float16). Recording-level vectors and
the 5/10/15/30 s ablation are built later by averaging subsets of chunks.
Output arrays: mean[N, n_chunks, L, D], std[N, n_chunks, L, D].
"""
import argparse, os, numpy as np, torch
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from config import FEATURES, SR, MERT_MODEL, CHUNK_SEC, load_manifest
from handcrafted import load_mono

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device():
    """Best available torch device: cuda, else Apple mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def embed_file(path, model, proc, device, chunk_sec):
    """Time-mean and time-std of every hidden layer for each chunk of one file: two arrays [n_chunks, L, D]."""
    x, sr = load_mono(path)
    assert sr == SR, f"expected {SR} Hz, got {sr}"
    n = chunk_sec * sr
    n_chunks = len(x) // n
    chunks = x[: n_chunks * n].reshape(n_chunks, n)
    inp = proc(list(chunks), sampling_rate=sr, return_tensors="pt")
    out = model(inp["input_values"].to(device), output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=1)          # [n_chunks, L, T, D]
    return hs.mean(2).float().cpu().numpy(), hs.std(2).float().cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--model", default=MERT_MODEL)
    ap.add_argument("--chunk_sec", type=int, default=CHUNK_SEC)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = get_device()
    print("device:", device, "model:", a.model)
    proc = Wav2Vec2FeatureExtractor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True).to(device).eval()
    m = load_manifest(a.dataset, a.data_root)
    means, stds = [], []
    for p in tqdm(m.path.tolist(), desc=f"MERT {a.dataset}"):
        mu, sd = embed_file(p, model, proc, device, a.chunk_sec)
        means.append(mu.astype(np.float16))
        stds.append(sd.astype(np.float16))
    tag = a.model.split("/")[-1]
    out = a.out or FEATURES / f"{a.dataset}_mert_{tag}_{a.chunk_sec}s.npz"
    FEATURES.mkdir(exist_ok=True)
    np.savez(out, mean=np.stack(means), std=np.stack(stds), sample_id=m.sample_id.values,
             split=m.split.values, label=m.label.fillna("").values, model=tag, chunk_sec=a.chunk_sec)
    print(f"saved {out}: mean={np.stack(means).shape}")
