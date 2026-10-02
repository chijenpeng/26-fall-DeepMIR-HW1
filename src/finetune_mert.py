"""Fine-tune MERT-v1-95M end-to-end on random 5 s chunks; evaluate by averaging the 6 chunks.

  CUDA_VISIBLE_DEVICES=0 python src/finetune_mert.py --dataset A --epochs 8

Head: softmax-weighted sum over the 13 hidden layers -> time mean -> dropout -> linear.
Saves results/ft_<ds>_best.pt (head + encoder weights), results/ft_<ds>_val.json and
results/ft_<ds>_proba.npz (validation + test probabilities, for the submission file).
"""
import argparse, json, numpy as np, torch, torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from config import RESULTS, LABELS, SR, MERT_MODEL, CHUNK_SEC, load_manifest
from handcrafted import load_mono
from utils import evaluate, neighbour_error_rate, dump


class MertClassifier(nn.Module):
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
    wavs = []
    for p in tqdm(m.path, desc="loading audio"):
        x, sr = load_mono(p); assert sr == SR
        wavs.append(proc(x, sampling_rate=sr, return_tensors="pt")["input_values"][0].float())
    return wavs


@torch.no_grad()
def predict(model, wavs, idx, device, n=CHUNK_SEC * SR, bs=8):
    model.eval(); out = []
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
    ap.add_argument("--dataset", required=True, choices=["A", "B"]); ap.add_argument("--model", default=MERT_MODEL)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr_enc", type=float, default=2e-5); ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--freeze_feature_extractor", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default=None)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda"; labels = LABELS[a.dataset]; lab2i = {l: i for i, l in enumerate(labels)}
    m = load_manifest(a.dataset); proc = Wav2Vec2FeatureExtractor.from_pretrained(a.model, trust_remote_code=True)
    wavs = load_all(m, proc)
    y = np.array([lab2i.get(l, -1) for l in m.label.fillna("")]); split = m.split.values
    tr = np.flatnonzero(split == "train"); va = np.flatnonzero(split == "validation"); te = np.flatnonzero(split == "test")
    model = MertClassifier(a.model, len(labels)).to(device)
    if a.freeze_feature_extractor:
        for p in model.enc.feature_extractor.parameters():
            p.requires_grad = False
    enc_params = [p for n, p in model.enc.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc}, {"params": list(model.head.parameters()) + [model.layer_w], "lr": a.lr_head}], weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_enc, a.lr_head], total_steps=steps, pct_start=0.1)
    n = CHUNK_SEC * SR; name = a.name or f"ft_{a.dataset}"; best = (-1, None); hist = []
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr); losses = []
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[i:i + a.bs]
            starts = [np.random.randint(0, len(wavs[j]) - n + 1) for j in idx]
            x = torch.stack([wavs[j][s:s + n] for j, s in zip(idx, starts)]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(x).float(), torch.tensor(y[idx], device=device), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            losses.append(loss.item())
        pv = predict(model, wavs, va, device); res = evaluate(pv, y[va], labels, "")
        hist.append(dict(epoch=ep, loss=float(np.mean(losses)), val_top1=res["top1"], val_top3=res["top3"]))
        print(f"epoch {ep}: loss={np.mean(losses):.3f} val top1={res['top1']:.3f} top3={res['top3']:.3f}", flush=True)
        if res["top1"] > best[0]:
            best = (res["top1"], ep); torch.save(model.state_dict(), RESULTS / f"{name}_best.pt")
    model.load_state_dict(torch.load(RESULTS / f"{name}_best.pt"))
    pv, pt = predict(model, wavs, va, device), predict(model, wavs, te, device)
    res = evaluate(pv, y[va], labels, f"{name} (best epoch {best[1]})", RESULTS / f"{name}_cm.png")
    res.update(dict(dataset=a.dataset, method="finetune_mert95m", best_epoch=best[1], history=hist, args=vars(a),
                    neighbour_error_rate=neighbour_error_rate(res["confusion_counts"]),
                    layer_weights=torch.softmax(model.layer_w, 0).tolist()))
    dump(res, RESULTS / f"{name}_val.json")
    np.savez(RESULTS / f"{name}_proba.npz", sample_id=np.concatenate([m.sample_id.values[va], m.sample_id.values[te]]),
             split=np.concatenate([split[va], split[te]]), proba=np.concatenate([pv, pt]), labels=np.array(labels))
    print(json.dumps({k: v for k, v in res.items() if k not in ("confusion_counts", "history", "args")}, indent=1))
    print(np.array(res["confusion_counts"]))
