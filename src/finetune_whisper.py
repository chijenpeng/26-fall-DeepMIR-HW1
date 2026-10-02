"""Fine-tune a Whisper encoder (tiny / base / small / medium) end-to-end on the full 30 s clip.

  CUDA_VISIBLE_DEVICES=0 python src/finetune_whisper.py --dataset A --size small --epochs 20

Head: softmax-weighted sum over encoder layers -> time mean -> dropout -> linear. Same logging,
per-epoch bf16 checkpoints, history json and curve plot as finetune_mert.py.
"""
import argparse, json, numpy as np, torch, torch.nn as nn, librosa
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel
from config import RESULTS, LABELS, load_manifest
from handcrafted import load_mono
from utils import evaluate, neighbour_error_rate, dump
from extract_whisper import SIZES


class WhisperClassifier(nn.Module):
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
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(X[idx[i:i + bs]].to(device).float()).float()
        out.append(torch.softmax(logits, -1).cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"]); ap.add_argument("--size", default="small", choices=list(SIZES))
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr_enc", type=float, default=1e-5); ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default=None); ap.add_argument("--ckpt_every", type=int, default=5)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda"; labels = LABELS[a.dataset]; lab2i = {l: i for i, l in enumerate(labels)}
    m = load_manifest(a.dataset); fe = WhisperFeatureExtractor.from_pretrained(SIZES[a.size])
    X = torch.stack([fe(librosa.resample(load_mono(p)[0], orig_sr=24000, target_sr=16000), sampling_rate=16000, return_tensors="pt")
                     .input_features[0].to(torch.float16) for p in tqdm(m.path, desc="log-mel")])      # [N, n_mels, 3000]
    y = np.array([lab2i.get(l, -1) for l in m.label.fillna("")]); split = m.split.values
    tr = np.flatnonzero(split == "train"); va = np.flatnonzero(split == "validation"); te = np.flatnonzero(split == "test")
    model = WhisperClassifier(SIZES[a.size], len(labels)).to(device)
    for p in model.enc.conv1.parameters(): p.requires_grad = False
    for p in model.enc.conv2.parameters(): p.requires_grad = False
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc}, {"params": list(model.head.parameters()) + [model.layer_w], "lr": a.lr_head}], weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_enc, a.lr_head], total_steps=steps, pct_start=0.1)
    name = a.name or f"ftw_{a.size}_{a.dataset}"; best = (-1, None); hist = []
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr); losses = []
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[i:i + a.bs]
            x = X[idx].to(device).float()
            if np.random.rand() < 0.5:                                   # light SpecAugment: one time mask up to 3 s
                t0 = np.random.randint(0, 3000 - 300); x[:, :, t0:t0 + np.random.randint(50, 300)] = x.mean()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(x).float(), torch.tensor(y[idx], device=device), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            losses.append(loss.item())
        pv = predict(model, X, va, device); res = evaluate(pv, y[va], labels, "")
        val_loss = float(-np.log(pv[np.arange(len(va)), y[va]] + 1e-9).mean())
        hist.append(dict(epoch=ep, loss=float(np.mean(losses)), val_loss=val_loss, val_top1=res["top1"], val_top3=res["top3"]))
        print(f"epoch {ep:2d}: train_loss={np.mean(losses):.3f} val_loss={val_loss:.3f} val top1={res['top1']:.3f} top3={res['top3']:.3f}", flush=True)
        if res["top1"] > best[0]:
            best = (res["top1"], ep); torch.save(model.state_dict(), RESULTS / f"{name}_best.pt")
        if a.ckpt_every and (ep + 1) % a.ckpt_every == 0:
            ck = RESULTS / "ckpt" / name; ck.mkdir(parents=True, exist_ok=True)
            torch.save({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()}, ck / f"ep{ep:02d}.pt")
        dump(dict(history=hist, best_epoch=best[1]), RESULTS / f"{name}_history.json")
    model.load_state_dict(torch.load(RESULTS / f"{name}_best.pt"))
    pv, pt = predict(model, X, va, device), predict(model, X, te, device)
    res = evaluate(pv, y[va], labels, f"{name} (best epoch {best[1]})", RESULTS / f"{name}_cm.png")
    res.update(dict(dataset=a.dataset, method=f"finetune_whisper_{a.size}", best_epoch=best[1], history=hist, args=vars(a),
                    neighbour_error_rate=neighbour_error_rate(res["confusion_counts"]), layer_weights=torch.softmax(model.layer_w, 0).tolist()))
    dump(res, RESULTS / f"{name}_val.json")
    np.savez(RESULTS / f"{name}_proba.npz", sample_id=np.concatenate([m.sample_id.values[va], m.sample_id.values[te]]),
             split=np.concatenate([split[va], split[te]]), proba=np.concatenate([pv, pt]), labels=np.array(labels))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]; fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["loss"] for h in hist], label="train loss"); ax[0].plot(ep, [h["val_loss"] for h in hist], label="val loss (NLL)")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [h["val_top1"] for h in hist], label="val top-1"); ax[1].plot(ep, [h["val_top3"] for h in hist], label="val top-3")
    ax[1].axvline(best[1], ls="--", c="gray"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"{name}  (best epoch {best[1]})"); plt.tight_layout(); plt.savefig(RESULTS / f"{name}_curves.png", dpi=130)
    print(json.dumps({k: v for k, v in res.items() if k not in ("confusion_counts", "history", "args")}, indent=1))
