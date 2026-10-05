"""Fine-tune a Whisper encoder (tiny / base / small / medium) end-to-end on the full 30 s clip.

  CUDA_VISIBLE_DEVICES=0 python src/finetune_whisper.py --dataset A --size small --epochs 20

Head: softmax-weighted sum over encoder layers -> time mean -> dropout -> linear.
Reads   the dataset manifest and audio.
Writes  the same set of files as finetune_mert.py (results/<name>_best.pt, _history.json, _val.json,
        _cm.png, _proba.npz, _curves.png, results/ckpt/<name>/epXX.pt); <name> defaults to
        ftw_<size>_<ds>. The shared logging / checkpointing / result code is in finetune_common.py.
"""
import argparse, numpy as np, torch, torch.nn as nn, librosa
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel
from config import RESULTS, LABELS, load_manifest
from handcrafted import load_mono
from utils import label_indices
from extract_whisper import SIZES
from finetune_common import split_indices, train_step, end_epoch, write_final_results


class WhisperClassifier(nn.Module):
    """Whisper encoder + learned softmax weighting of its layers + linear classification head."""

    def __init__(self, name, n_classes, dropout=0.2):
        super().__init__()
        self.enc = WhisperModel.from_pretrained(name).encoder
        L = self.enc.config.encoder_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(L))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.enc.config.d_model, n_classes))

    def forward(self, feats):                                   # feats: [B, n_mels, 3000]
        hs = torch.stack(self.enc(feats, output_hidden_states=True).hidden_states, 1)   # [B, L, T, D]
        w = torch.softmax(self.layer_w, 0)[None, :, None, None]
        return self.head((hs * w).sum(1).mean(1))


@torch.no_grad()
def predict(model, X, idx, device, bs=16):
    """Class probabilities [len(idx), n_classes] for the rows `idx` of the log-mel tensor X."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(X[idx[i:i + bs]].to(device).float()).float()
        out.append(torch.softmax(logits, -1).cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--size", default="small", choices=list(SIZES))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ckpt_every", type=int, default=5)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    labels = LABELS[a.dataset]
    m = load_manifest(a.dataset)
    fe = WhisperFeatureExtractor.from_pretrained(SIZES[a.size])
    X = torch.stack([fe(librosa.resample(load_mono(p)[0], orig_sr=24000, target_sr=16000), sampling_rate=16000, return_tensors="pt")
                     .input_features[0].to(torch.float16) for p in tqdm(m.path, desc="log-mel")])      # [N, n_mels, 3000]
    y = label_indices(m.label.fillna(""), labels)
    tr, va, te = split_indices(m.split.values)
    model = WhisperClassifier(SIZES[a.size], len(labels)).to(device)
    for p in model.enc.conv1.parameters():
        p.requires_grad = False
    for p in model.enc.conv2.parameters():
        p.requires_grad = False
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc},
                             {"params": list(model.head.parameters()) + [model.layer_w], "lr": a.lr_head}], weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_enc, a.lr_head], total_steps=steps, pct_start=0.1)
    name = a.name or f"ftw_{a.size}_{a.dataset}"
    best = (-1, None)                                           # (best validation top-1, its epoch)
    hist = []
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(tr)
        losses = []
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[i:i + a.bs]
            x = X[idx].to(device).float()
            if np.random.rand() < 0.5:                                   # light SpecAugment: one time mask up to 3 s
                t0 = np.random.randint(0, 3000 - 300)
                x[:, :, t0:t0 + np.random.randint(50, 300)] = x.mean()
            losses.append(train_step(model, x, y[idx], opt, sched, device))
        best = end_epoch(model, name, ep, losses, predict(model, X, va, device), y[va], labels, hist, best, a.ckpt_every)
    model.load_state_dict(torch.load(RESULTS / f"{name}_best.pt"))
    pv, pt = predict(model, X, va, device), predict(model, X, te, device)
    write_final_results(model, name, f"finetune_whisper_{a.size}", a, labels, hist, best[1], m, y, va, te, pv, pt)
