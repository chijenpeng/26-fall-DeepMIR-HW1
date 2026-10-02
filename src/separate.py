"""Demucs source separation -> sibling datasets that reuse the original manifest.

  python src/separate.py --dataset B            # writes dataset_B_vocals/ and dataset_B_accomp/

Each output folder has manifest.csv (copied) and audio/<id>.wav (24 kHz mono 16-bit), so
extract_mert.py can be pointed at it with --data_root / --out.
"""
import argparse, shutil, numpy as np, torch, torchaudio, soundfile as sf
from pathlib import Path
from tqdm import tqdm
from demucs.pretrained import get_model
from demucs.apply import apply_model
from config import DATA, SR, load_manifest
from handcrafted import load_mono
from extract_mert import get_device


@torch.no_grad()
def separate_file(path, model, device):
    x, sr = load_mono(path)
    wav = torch.from_numpy(x)[None]                                        # [1, T]
    wav = torchaudio.functional.resample(wav, sr, model.samplerate)
    mix = wav.repeat(2, 1)[None].to(device)                                # [1, 2, T] fake stereo
    ref = mix.mean(0); mix = (mix - ref.mean()) / (ref.std() + 1e-8)
    out = apply_model(model, mix, device=device, shifts=0, split=True, overlap=0.25, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()                            # [S, 2, T]
    vi = model.sources.index("vocals")
    vocals = out[vi].mean(0)
    accomp = torch.stack([out[i] for i in range(len(model.sources)) if i != vi]).sum(0).mean(0)
    back = lambda y: torchaudio.functional.resample(y.cpu()[None], model.samplerate, sr)[0].numpy()
    return back(vocals), back(accomp), sr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--data_root", default=None)
    a = ap.parse_args()
    device = get_device(); print("device:", device)
    model = get_model(a.model).to(device).eval()
    m = load_manifest(a.dataset, a.data_root)
    src_root = Path(a.data_root) if a.data_root else DATA[a.dataset]
    outs = {k: src_root.parent / f"{src_root.name}_{k}" for k in ("vocals", "accomp")}
    for d in outs.values():
        (d / "audio").mkdir(parents=True, exist_ok=True)
        shutil.copy(src_root / "manifest.csv", d / "manifest.csv")
    for p, rel in tqdm(list(zip(m.path, m.audio_path)), desc=f"demucs {a.dataset}"):
        if all((outs[k] / rel).exists() for k in outs):
            continue
        v, acc, sr = separate_file(p, model, device)
        for k, y in (("vocals", v), ("accomp", acc)):
            sf.write(outs[k] / rel, np.clip(y, -1, 1), sr, subtype="PCM_16")
    print("done:", {k: str(v) for k, v in outs.items()})
