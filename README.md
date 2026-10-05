# DeepMIR 2026 Fall, Homework 1: Music Era and Release-Market Classification

Chi-Jen Peng (R14725022)

Two six-class tasks on 30-second clips (24 kHz mono):

| Task | Labels | Final model | Val top-1 | Val top-3 |
|---|---|---|---|---|
| 1. Release decade | 1960s ... 2010s | Whisper-tiny encoder layer 1 + MERT-v1-330M layers 5, 6, 14, logistic regression | 52.3% | 86.4% |
| 2. Release market | US, UK, Brazil, Spain, Germany, Italy | Whisper-large-v3 encoder layers 23, 30 + MERT-v1-95M layer 7 + Whisper-small language ID, logistic regression | 60.8% | 90.2% |

All encoders are frozen. Frame features are pooled with mean + std over time, standardized, and
classified with an L2 logistic regression. The report is `submission/r14725022_report.pdf`; the
submitted predictions are `submission/r14725022.json`.

## Inference

Tested with Python 3.11 on Linux with CUDA 12.

```bash
pip install -r requirements.txt

python src/inference.py \
    --data_A /path/to/dataset_A \
    --data_B /path/to/dataset_B \
    --out r14725022.json
```

- Each data folder should look like the released datasets: `manifest.csv` and `audio/*.wav`.
  Rows with `split == test` are predicted. Use `--split all` to predict every row.
  If a folder has no `manifest.csv`, every `.wav` below it is predicted.
- The output has the required format: for every test `sample_id`, the top-3 labels in descending
  confidence order, under `dataset_A` and `dataset_B`.
- The script extracts features into `features_inference/` and then applies the two classifiers in
  `checkpoints/best/`. Add `--skip_extract` to reuse features from a previous run.
- Five pretrained models are downloaded from the Hugging Face Hub on first use (about 6 GB in total):
  `openai/whisper-tiny`, `openai/whisper-small`, `openai/whisper-large-v3`,
  `m-a-p/MERT-v1-95M`, `m-a-p/MERT-v1-330M`. MERT uses `trust_remote_code=True`.
- A CUDA GPU is used when available. The code also falls back to Apple MPS or the CPU, but only the CUDA path was tested.

**Verified:** running this command on the released audio reproduces `submission/r14725022.json` exactly
(132 / 132 Task 1 and 102 / 102 Task 2 clips with identical top-3 lists). It took 56 seconds on one
NVIDIA RTX A4500 with the models already downloaded.

## Repository layout

```
README.md                       this file
requirements.txt                packages needed for inference (install this one)
requirements-experiments.txt    extra packages for re-running the experiments (includes requirements.txt)

submission/
  r14725022_report.pdf          the report
  r14725022.json                the submitted test predictions

checkpoints/best/
  task1_decade.joblib           final Task 1 classifier (StandardScaler + logistic regression)
  task2_market.joblib           final Task 2 classifier

src/
  inference.py                  end-to-end inference (the command above)
  config.py                     paths, labels, manifest loading
  utils.py                      top-k accuracy, confusion-matrix plots, label / split helpers, JSON output

  # feature extraction
  extract_whisper.py            Whisper encoder hidden states, mean + std per layer
  extract_mert.py               MERT hidden states, mean + std per 5 s chunk and layer
  extract_lang.py               Whisper language-ID posteriors (7-d)
  extract_handcrafted.py        24 hand-crafted production features per clip
  handcrafted.py                the feature definitions (band energies, crest factor, limiter statistics)
  separate.py                   Demucs vocals / accompaniment stems
  features.py                   assemble design matrices from the feature files

  # training and evaluation
  train.py                      logistic regression / SVM / MLP on any combination of feature blocks
  layer_sweep.py                one classifier per encoder layer
  train_chunks.py               chunk-level training (multiple excerpts per recording)
  regression.py                 Task 1 as ridge regression / hierarchical prediction
  finetune_mert.py              end-to-end fine-tuning of MERT
  finetune_whisper.py           end-to-end fine-tuning of a Whisper encoder
  finetune_common.py            training step, logging, checkpoints and result files shared by the two fine-tuning scripts
  alm_qwen2audio.py             zero-shot Qwen2-Audio with three prompt designs

  # analysis and figures
  tsne.py                       t-SNE of features or classifier logits
  make_slide_figs.py            figures used in the report

scripts/
  reproduce_all.sh              every command behind the numbers in the report

eda/                            spectral and dynamics analysis of the Task 1 training split (scripts, tables, plots)
results/                        validation metrics (*_val.json), confusion matrices (*_cm.png), layer sweeps, ALM outputs
slides/                         LaTeX source of the report (main.tex, figs/)
notes/                          homework handout, prediction format example, working notes (in Chinese)
```

Not in the repository; created locally:

```
dataset_A/, dataset_B/          the released data (manifest.csv + audio/*.wav)
features/                       cached features written by the extract_*.py scripts
features_inference/             cached features written by inference.py
results/*.joblib                classifiers written by train.py
```

## Reproducing the final models

Put the released data in `dataset_A/` and `dataset_B/`, install the experiment requirements, and
run from the repository root:

```bash
pip install -r requirements-experiments.txt

# Task 1
python src/extract_whisper.py --dataset A --size tiny
python src/extract_mert.py    --dataset A --model m-a-p/MERT-v1-330M
python src/train.py --dataset A --layers 7 --C 0.0003 0.001 0.003 0.01 --name A_wt_L1_m330 \
    --features "mert@features/A_mert_whisper-tiny_30s.npz:1+mert@features/A_mert_MERT-v1-330M_5s.npz:5,6,14"

# Task 2
python src/extract_whisper.py --dataset B --size large-v3
python src/extract_mert.py    --dataset B
python src/extract_lang.py    --dataset B
python src/train.py --dataset B --layers 7 --C 0.0003 0.001 0.003 0.01 --name B_wl_L23-30_mert7_lang \
    --features "mert@features/B_mert_whisper-large-v3_30s.npz:23,30+mert+lang"
```

`train.py` fits on the training split, selects `C` by validation top-1, and writes
`results/<name>.joblib`, `results/<name>_val.json` and `results/<name>_cm.png`.
A feature block is `mert` (default MERT-v1-95M file, layers from `--layers`),
`mert@<npz>:<layers>` (any encoder feature file with its own layers), `lang`, or `handcrafted`,
joined with `+`.

## Reproducing the experiments

`scripts/reproduce_all.sh` lists every command, grouped as in the report:

| Report section | Scripts |
|---|---|
| Results, error analysis | `train.py`, `eda/spectral_eda.py`, `eda/dynamics_eda.py`, `extract_lang.py` |
| Input length | `extract_whisper.py --seconds`, `extract_lang.py --seconds`, `train.py --n_chunks` |
| Multiple excerpts | `train_chunks.py` |
| Regression | `regression.py` |
| Mixture / vocals / accompaniment | `separate.py`, then the extractors with `--data_root` |
| t-SNE | `tsne.py` |
| Ablation: features, layers, sizes | `layer_sweep.py`, `train.py` |
| Ablation: fine-tuning | `finetune_mert.py`, `finetune_whisper.py` |
| Audio language model | `alm_qwen2audio.py` |

The script consolidates the commands that were actually run during the project. Only the
inference path above was re-run end to end after the consolidation.

## Checkpoints

- `checkpoints/best/` (in this repository): the two scikit-learn pipelines used for the submission.
  Each file is a dict with the fitted `StandardScaler + LogisticRegression` under `"clf"`.
- Other checkpoints are not needed for inference and are not part of the submission:
  the classifiers of the frozen-feature ablations are re-created in seconds by `train.py`, and the
  fine-tuned encoders by `finetune_mert.py` / `finetune_whisper.py`.

## Environment

| | |
|---|---|
| GPU | 2 x NVIDIA RTX A4500 (20 GB); inference needs one |
| CPU / RAM | Intel i9-14900K, 125 GB |
| OS | Ubuntu 26.04 |
| Python | 3.11 |
| Key packages | torch 2.14.1, transformers 4.45.2, scikit-learn 1.9.1, librosa 0.11.0 |

`scikit-learn` and `transformers` are pinned because the classifiers are stored as pickles and the
MERT remote code was tested with this `transformers` version.

## Model selection and limitations

- Training split for fitting, validation split for every choice (encoder, layers, `C`, fine-tuning
  epoch). Test labels are hidden; the test split is only predicted.
- More than 100 configurations were compared on 132 and 102 validation clips, so the validation
  scores above are optimistic estimates of test accuracy.
- The two final models are logistic regressions fitted with L-BFGS, which is deterministic. The RBF
  SVM used in the ablation is seeded (`random_state=0`); its three numbers in the report come from
  the seeded runs. Fine-tuning runs are seeded but GPU results can still vary slightly between runs.

## Use of AI assistance

The code and the report slides were written with the help of Claude Code (Claude Fable 5.1,
Anthropic), which also ran the experiments. The ideas and hypotheses, the choice of experiments
and the selection of the final models are the author's. Every number comes from running the code
in this repository.

## References

Pretrained models, datasets, libraries and methods are cited on the last two slides of the report.
