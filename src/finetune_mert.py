"""Fine-tune MERT (default MERT-v1-95M, see --model) end-to-end on random 5 s chunks; evaluate by averaging the 6 chunks.

  CUDA_VISIBLE_DEVICES=0 python src/finetune_mert.py --dataset A --epochs 8

Head: softmax-weighted sum over all hidden layers -> time mean -> dropout -> linear.
Reads   the dataset manifest and audio.
Writes  results/<name>_best.pt (head + encoder weights), _history.json, _val.json, _cm.png, _proba.npz
        (validation + test probabilities), _curves.png and results/ckpt/<name>/epXX.pt; <name> defaults
        to ft_<ds>. See finetune_common.py, which holds the code shared with finetune_whisper.py.
"""
import argparse, numpy as np, torch, torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from config import RESULTS, LABELS, SR, MERT_MODEL, CHUNK_SEC, load_manifest
from handcrafted import load_mono
from utils import label_indices
from finetune_common import split_indices, train_step, end_epoch, write_final_results


class MertClassifier(nn.Module):
    """MERT encoder + learned softmax weighting of its hidden layers + linear classification head."""

    def __init__(self, name, n_classes, dropout=0.2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, trust_remote_code=True)
        L = self.enc.config.num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(L))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.enc.config.hidden_size, n_classes))

    def forward(self, x):                                   # x: [B, T] normalised waveform
        hs = torch.stack(self.enc(x, output_hidden_states=True).hidden_states, 1)   # [B, L, T', D]
        w = torch.softmax(self.layer_w, 0)[None, :, None, None]
        return self.head((hs * w).sum(1).mean(1))


def load_all(m, proc):
    """Every clip of the manifest as a normalised waveform tensor (kept in memory)."""
    wavs = []
    for p in tqdm(m.path, desc="loading audio"):
        x, sr = load_mono(p)
        assert sr == SR
        wavs.append(proc(x, sampling_rate=sr, return_tensors="pt")["input_values"][0].float())
    return wavs


@torch.no_grad()
def predict(model, wavs, idx, device, n=CHUNK_SEC * SR, bs=8):
    """Class probabilities [len(idx), n_classes] of the clips `idx`: softmax averaged over each clip's chunks of n samples."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        batch = [wavs[j] for j in idx[i:i + bs]]
        chunks = torch.stack([w[: len(w) // n * n].reshape(-1, n) for w in batch])   # [b, 6, n]
        b, k = chunks.shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(chunks.reshape(b * k, n).to(device)).float()
        out.append(torch.softmax(logits, -1).reshape(b, k, -1).mean(1).cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--model", default=MERT_MODEL)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--freeze_feature_extractor", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ckpt_every", type=int, default=1, help="save a bf16 checkpoint every k epochs (0 = only best)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    labels = LABELS[a.dataset]
    m = load_manifest(a.dataset)
    proc = Wav2Vec2FeatureExtractor.from_pretrained(a.model, trust_remote_code=True)
    wavs = load_all(m, proc)
    y = label_indices(m.label.fillna(""), labels)
    tr, va, te = split_indices(m.split.values)
    model = MertClassifier(a.model, len(labels)).to(device)
    if a.freeze_feature_extractor:
        for p in model.enc.feature_extractor.parameters():
            p.requires_grad = False
    enc_params = [p for n, p in model.enc.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc},
                             {"params": list(model.head.parameters()) + [model.layer_w], "lr": a.lr_head}], weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_enc, a.lr_head], total_steps=steps, pct_start=0.1)
    n = CHUNK_SEC * SR
    name = a.name or f"ft_{a.dataset}"
    best = (-1, None)                                       # (best validation top-1, its epoch)
    hist = []
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(tr)
        losses = []
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[i:i + a.bs]
            starts = [np.random.randint(0, len(wavs[j]) - n + 1) for j in idx]     # one random 5 s chunk per clip
            x = torch.stack([wavs[j][s:s + n] for j, s in zip(idx, starts)]).to(device)
            losses.append(train_step(model, x, y[idx], opt, sched, device))
        best = end_epoch(model, name, ep, losses, predict(model, wavs, va, device), y[va], labels, hist, best, a.ckpt_every)
    model.load_state_dict(torch.load(RESULTS / f"{name}_best.pt"))
    pv, pt = predict(model, wavs, va, device), predict(model, wavs, te, device)
    # the stored method string is "finetune_mert95m" whatever --model is (known labelling quirk, left as is)
    res = write_final_results(model, name, "finetune_mert95m", a, labels, hist, best[1], m, y, va, te, pv, pt)
    print(np.array(res["confusion_counts"]))
