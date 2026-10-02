"""Whisper language-ID posteriors per clip -> features/<ds>_lang_<model>.npz

Only the language-detection step is run (encoder + one decoder step), no transcription.
Output: probs[N, len(LANGS)+1] = P(en, pt, es, de, it, fr, other) renormalised over all
Whisper language tokens; top[N] = argmax language code over all languages.
"""
import argparse, os, numpy as np, torch, librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from config import FEATURES, load_manifest
from handcrafted import load_mono
from extract_mert import get_device

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
LANGS = ["en", "pt", "es", "de", "it", "fr"]


@torch.no_grad()
def lang_probs(paths, model, proc, device, lang_ids, sot):
    xs = [librosa.resample(load_mono(p)[0], orig_sr=24000, target_sr=16000) for p in paths]
    feats = proc(xs, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    dec = torch.full((len(xs), 1), sot, dtype=torch.long, device=device)
    logits = model(input_features=feats, decoder_input_ids=dec).logits[:, -1, :]
    return torch.softmax(logits[:, lang_ids], -1).cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = get_device(); print("device:", device, "model:", a.model)
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model).to(device).eval()
    tok = proc.tokenizer
    codes = list(LANGUAGES.keys())
    lang_ids = [tok.convert_tokens_to_ids(f"<|{c}|>") for c in codes]
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    m = load_manifest(a.dataset, a.data_root); paths = m.path.tolist()
    P = []
    for i in tqdm(range(0, len(paths), a.batch), desc=f"whisper-lang {a.dataset}"):
        P.append(lang_probs(paths[i:i + a.batch], model, proc, device, lang_ids, sot))
    P = np.concatenate(P)                                    # [N, n_all_langs]
    sel = [codes.index(c) for c in LANGS]
    probs = np.concatenate([P[:, sel], 1 - P[:, sel].sum(1, keepdims=True)], 1).astype(np.float32)
    top = np.array([codes[j] for j in P.argmax(1)])
    tag = a.model.split("/")[-1]
    FEATURES.mkdir(exist_ok=True)
    out = a.out or FEATURES / f"{a.dataset}_lang_{tag}.npz"
    np.savez(out, probs=probs, names=np.array(LANGS + ["other"]), top=top, sample_id=m.sample_id.values,
             split=m.split.values, label=m.label.fillna("").values, model=tag)
    print(f"saved {out}: probs={probs.shape}")
