"""Audio-language-model baseline: Qwen2-Audio-7B-Instruct, zero-shot.

For every clip in the chosen splits we
  (1) let the model answer freely (greedy, <=16 new tokens) and parse a label from the text;
  (2) score each of the six candidate labels by teacher-forced log-probability of the label
      tokens given the prompt -> a full ranking, so top-3 is always well defined and there
      are no invalid outputs in that mode.

  CUDA_VISIBLE_DEVICES=1 python src/alm_qwen2audio.py --dataset A --splits validation,test --prompt naive

Output: results/alm_<model>_<dataset>_<prompt>.json  (per-sample records + validation metrics)
"""
import argparse, json, re, numpy as np, torch, librosa
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from config import RESULTS, LABELS, load_manifest
from handcrafted import load_mono
from utils import evaluate

PROMPTS = {
    "A": {
        "naive": ("This is a 30-second excerpt of a music recording released in the United States. "
                  "Our goal is to identify the decade in which the recording was released. "
                  "Which decade is it? Answer with exactly one of: 1960s, 1970s, 1980s, 1990s, 2000s, 2010s."),
    },
    "B": {
        "naive": ("This is a 30-second excerpt of a music recording released in the 1980s. "
                  "Our goal is to identify the country (release market) in which this record was issued. "
                  "Which market is it? Answer with exactly one of: US, UK, Brazil, Spain, Germany, Italy."),
    },
}
SYNONYMS = {"B": {"united states": "US", "usa": "US", "america": "US", "united kingdom": "UK", "britain": "UK",
                  "england": "UK", "brasil": "Brazil", "deutschland": "Germany", "italia": "Italy", "españa": "Spain"}}


def parse_label(text, ds):
    t = text.lower()
    if ds == "A":
        m = re.search(r"(19[6-9]0|20[01]0)\s*'?s", t) or re.search(r"(19[6-9]|20[01])\d", t)
        if m:
            return f"{m.group(1)[:3]}0s" if len(m.group(1)) == 3 else f"{m.group(1)}s"
        return None
    for syn, lab in SYNONYMS["B"].items():
        if syn in t:
            return lab
    for lab in LABELS["B"]:
        if re.search(rf"\b{lab.lower()}\b", t):
            return lab
    return None


def build_inputs(proc, audio16k, prompt, device, suffix=""):
    conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False) + suffix
    inputs = proc(text=text, audios=[audio16k], sampling_rate=16000, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def run_clip(model, proc, audio16k, prompt, labels, device):
    inputs = build_inputs(proc, audio16k, prompt, device)
    gen = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    gen_text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    scores = {}
    for lab in labels:                      # teacher-forced log p(label tokens | prompt)
        k = len(proc.tokenizer(lab, add_special_tokens=False).input_ids)
        inp = build_inputs(proc, audio16k, prompt, device, suffix=lab)
        logits = model(**inp).logits[0].float()
        ids = inp["input_ids"][0]
        lp = torch.log_softmax(logits[-k - 1:-1], -1).gather(1, ids[-k:, None]).sum().item()
        scores[lab] = lp
    return gen_text, scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"])
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--prompt", default="naive")
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--device_map", default="cuda:0")
    ap.add_argument("--data_root", default=None)
    a = ap.parse_args()
    labels = LABELS[a.dataset]; prompt = PROMPTS[a.dataset][a.prompt]
    proc = AutoProcessor.from_pretrained(a.model)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map=a.device_map).eval()
    device = model.device
    m = load_manifest(a.dataset, a.data_root); m = m[m.split.isin(a.splits.split(","))].reset_index(drop=True)
    tag = a.model.split("/")[-1]; RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"alm_{tag}_{a.dataset}_{a.prompt}.json"
    records = {}
    if out_path.exists():                                     # resume
        records = json.load(open(out_path)).get("records", {})
    for _, r in tqdm(list(m.iterrows()), desc=f"ALM {a.dataset} {a.prompt}"):
        if r.sample_id in records:
            continue
        x, sr = load_mono(r.path); x16 = librosa.resample(x, orig_sr=sr, target_sr=16000)
        gen_text, scores = run_clip(model, proc, x16, prompt, labels, device)
        ranking = sorted(labels, key=lambda l: -scores[l])
        records[r.sample_id] = dict(split=r.split, label=r.label if isinstance(r.label, str) else "",
                                    gen_text=gen_text, gen_label=parse_label(gen_text, a.dataset),
                                    scores=scores, ranking=ranking)
        if len(records) % 25 == 0:
            json.dump(dict(prompt=prompt, records=records), open(out_path, "w"), indent=1)
    # validation metrics
    va = [v for v in records.values() if v["split"] == "validation" and v["label"]]
    metrics = {}
    if va:
        y = np.array([labels.index(v["label"]) for v in va])
        proba = np.array([[v["scores"][l] for l in labels] for v in va]); proba = np.exp(proba - proba.max(1, keepdims=True))
        res = evaluate(proba, y, labels, f"ALM {tag} {a.dataset} {a.prompt} (logprob ranking)", RESULTS / f"alm_{tag}_{a.dataset}_{a.prompt}_cm.png")
        gen_ok = [v["gen_label"] == v["label"] for v in va]; invalid = [v["gen_label"] is None for v in va]
        metrics = dict(score_top1=res["top1"], score_top3=res["top3"], confusion_counts=res["confusion_counts"],
                       gen_top1=float(np.mean(gen_ok)), gen_invalid_rate=float(np.mean(invalid)), n_val=len(va),
                       gen_label_dist={l: int(sum(v["gen_label"] == l for v in va)) for l in labels + [None]})
        print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_counts"}, indent=1))
        print(np.array(res["confusion_counts"]))
    json.dump(dict(prompt=prompt, metrics=metrics, records=records), open(out_path, "w"), indent=1)
    print("saved", out_path)
