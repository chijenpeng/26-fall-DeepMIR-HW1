"""Frozen Whisper-encoder features at several model sizes.

  CUDA_VISIBLE_DEVICES=0 python src/extract_whisper.py --dataset A --size small

Stored in the same layout as the MERT files (mean[N,1,L,D], std[N,1,L,D], one 30 s "chunk"),
under features/<ds>_mert_whisper-<size>_30s.npz, so train.py / layer_sweep.py consume them with
  --mert_model whisper-<size> --chunk_sec 30
"""
import argparse, numpy as np, torch, librosa
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel
from config import FEATURES, load_manifest
from handcrafted import load_mono
from extract_mert import get_device

SIZES = {"tiny": "openai/whisper-tiny", "base": "openai/whisper-base", "small": "openai/whisper-small",
         "medium": "openai/whisper-medium", "large-v3": "openai/whisper-large-v3"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"]); ap.add_argument("--size", required=True, choices=list(SIZES))
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--data_root", default=None)
    a = ap.parse_args(); device = get_device()
    fe = WhisperFeatureExtractor.from_pretrained(SIZES[a.size])
    enc = WhisperModel.from_pretrained(SIZES[a.size], torch_dtype=torch.float16).encoder.to(device).eval()
    m = load_manifest(a.dataset, a.data_root); paths = m.path.tolist(); means, stds = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), a.batch), desc=f"whisper-{a.size} {a.dataset}"):
            xs = [librosa.resample(load_mono(p)[0], orig_sr=24000, target_sr=16000) for p in paths[i:i + a.batch]]
            feats = fe(xs, sampling_rate=16000, return_tensors="pt").input_features.to(device, torch.float16)
            hs = torch.stack(enc(feats, output_hidden_states=True).hidden_states, 1).float()   # [B, L, T, D]
            means.append(hs.mean(2).cpu().numpy().astype(np.float16)); stds.append(hs.std(2).cpu().numpy().astype(np.float16))
    FEATURES.mkdir(exist_ok=True)
    out = FEATURES / f"{a.dataset}_mert_whisper-{a.size}_30s.npz"
    np.savez(out, mean=np.concatenate(means)[:, None], std=np.concatenate(stds)[:, None], sample_id=m.sample_id.values,
             split=m.split.values, label=m.label.fillna("").values, model=f"whisper-{a.size}", chunk_sec=30)
    print(f"saved {out}: mean={np.concatenate(means).shape}")
