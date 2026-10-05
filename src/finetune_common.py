"""Training-step, logging, checkpointing and result writing shared by finetune_mert.py and finetune_whisper.py.

The two scripts keep what differs between them (model class, data loading, batch construction,
optimizer setup, the epoch loop) and call the functions below for everything else. Files written
under results/, for a run called <name>:
  <name>_best.pt           state dict of the epoch with the best validation top-1
  ckpt/<name>/epXX.pt      bf16 copy of the state dict every --ckpt_every epochs
  <name>_history.json      per-epoch train loss, validation loss / top-1 / top-3 (rewritten every epoch)
  <name>_val.json          validation metrics of the best epoch, history, args, layer weights
  <name>_cm.png            validation confusion matrix of the best epoch
  <name>_proba.npz         validation + test probabilities of the best epoch
  <name>_curves.png        loss and accuracy curves
Reads nothing.
"""
import json, numpy as np, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS
from utils import evaluate, neighbour_error_rate, dump


def split_indices(split):
    """Row indices of the train, validation and test clips, in that order."""
    return [np.flatnonzero(split == s) for s in ("train", "validation", "test")]


def train_step(model, x, targets, opt, sched, device):
    """One optimizer + scheduler step on a batch: cross-entropy with label smoothing 0.1 under bf16
    autocast, gradient norm clipped at 1.0. `targets` are class indices (numpy). Returns the loss value."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = nn.functional.cross_entropy(model(x).float(), torch.tensor(targets, device=device), label_smoothing=0.1)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    return loss.item()


def end_epoch(model, name, ep, losses, val_proba, y_val, labels, hist, best, ckpt_every):
    """Bookkeeping after training epoch `ep`, given its batch losses and validation probabilities.

    Appends the epoch to `hist`, prints the log line, saves <name>_best.pt when validation top-1
    improves, saves a bf16 checkpoint every `ckpt_every` epochs (0 = never) and rewrites
    <name>_history.json. `best` is (best validation top-1, its epoch); the updated pair is returned."""
    res = evaluate(val_proba, y_val, labels, "")
    val_loss = float(-np.log(val_proba[np.arange(len(y_val)), y_val] + 1e-9).mean())
    hist.append(dict(epoch=ep, loss=float(np.mean(losses)), val_loss=val_loss, val_top1=res["top1"], val_top3=res["top3"]))
    print(f"epoch {ep:2d}: train_loss={np.mean(losses):.3f} val_loss={val_loss:.3f} val top1={res['top1']:.3f} top3={res['top3']:.3f}", flush=True)
    if res["top1"] > best[0]:
        best = (res["top1"], ep)
        torch.save(model.state_dict(), RESULTS / f"{name}_best.pt")
    if ckpt_every and (ep + 1) % ckpt_every == 0:
        ck = RESULTS / "ckpt" / name
        ck.mkdir(parents=True, exist_ok=True)
        torch.save({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()}, ck / f"ep{ep:02d}.pt")
    dump(dict(history=hist, best_epoch=best[1]), RESULTS / f"{name}_history.json")
    return best


def plot_curves(hist, best_epoch, name):
    """Two panels (train / validation loss; validation top-1 / top-3) -> <name>_curves.png."""
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["loss"] for h in hist], label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val loss (NLL)")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(ep, [h["val_top1"] for h in hist], label="val top-1")
    ax[1].plot(ep, [h["val_top3"] for h in hist], label="val top-3")
    ax[1].axvline(best_epoch, ls="--", c="gray")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.suptitle(f"{name}  (best epoch {best_epoch})")
    plt.tight_layout()
    plt.savefig(RESULTS / f"{name}_curves.png", dpi=130)


def write_final_results(model, name, method, args, labels, hist, best_epoch, m, y, va, te, val_proba, test_proba):
    """Write the results of the best epoch and print their summary. Returns the validation metrics dict.

    `model` must already hold the best weights (its layer weights are stored) and `val_proba` /
    `test_proba` are its predictions for the manifest rows `va` / `te`. Writes <name>_cm.png,
    <name>_val.json (with `method`, history and vars(args)), <name>_proba.npz and <name>_curves.png."""
    res = evaluate(val_proba, y[va], labels, f"{name} (best epoch {best_epoch})", RESULTS / f"{name}_cm.png")
    res.update(dict(dataset=args.dataset, method=method, best_epoch=best_epoch, history=hist, args=vars(args),
                    neighbour_error_rate=neighbour_error_rate(res["confusion_counts"]),
                    layer_weights=torch.softmax(model.layer_w, 0).tolist()))
    dump(res, RESULTS / f"{name}_val.json")
    sample_id, split = m.sample_id.values, m.split.values
    np.savez(RESULTS / f"{name}_proba.npz", sample_id=np.concatenate([sample_id[va], sample_id[te]]),
             split=np.concatenate([split[va], split[te]]), proba=np.concatenate([val_proba, test_proba]), labels=np.array(labels))
    plot_curves(hist, best_epoch, name)
    print(json.dumps({k: v for k, v in res.items() if k not in ("confusion_counts", "history", "args")}, indent=1))
    return res
