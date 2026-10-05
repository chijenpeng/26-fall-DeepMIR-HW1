#!/bin/bash
# Every command behind the numbers in the report, grouped by report section.
# Run from the repository root with dataset_A/ and dataset_B/ in place:
#     pip install -r requirements-experiments.txt
#     bash scripts/reproduce_all.sh
# GPU steps are marked [GPU]. The full script takes a few hours on one 20 GB GPU;
# most of that is fine-tuning (section 7) and the audio language model (section 8).
set -e
PY=${PY:-python}
T(){ name=$1; shift; $PY src/train.py "$@" --name "$name"; }      # train + validate one classifier
C4="--C 0.0003 0.001 0.003 0.01"

# ------------------------------------------------------------------ 0. data analysis (Results: why neighbors get confused)
$PY eda/spectral_eda.py            # long-term spectrum, band energies per decade   -> eda/
$PY eda/dynamics_eda.py            # crest factor, limiter statistics per decade    -> eda/

# ------------------------------------------------------------------ 1. feature extraction
for d in A B; do
  $PY src/extract_handcrafted.py --dataset $d                                   # 24-d production features
  $PY src/extract_mert.py --dataset $d                                          # [GPU] MERT-v1-95M
  $PY src/extract_mert.py --dataset $d --model m-a-p/MERT-v1-330M               # [GPU]
  for s in tiny base small medium large-v3; do $PY src/extract_whisper.py --dataset $d --size $s; done   # [GPU]
done
$PY src/extract_lang.py --dataset B                                             # [GPU] Whisper-small language ID

# ------------------------------------------------------------------ 2. final models (Results)
WT=mert@features/A_mert_whisper-tiny_30s.npz:1;  M330=mert@features/A_mert_MERT-v1-330M_5s.npz:5,6,14
WL=mert@features/B_mert_whisper-large-v3_30s.npz
T A_wt_L1_m330            --dataset A --features $WT+$M330 --layers 7 $C4
T B_wl_L23-30_mert7_lang  --dataset B --features $WL:23,30+mert+lang --layers 7 $C4
$PY src/predict.py --model_A results/A_wt_L1_m330.joblib --model_B results/B_wl_L23-30_mert7_lang.joblib --out submission/r14725022.json

# ------------------------------------------------------------------ 3. ablation: features, layers, sizes, pooling, classifier
for d in A B; do
  $PY src/train.py --dataset $d --features mert --layers all                                        # MERT-v1-95M, one classifier per layer
  $PY src/layer_sweep.py --dataset $d --mert_model MERT-v1-330M
  for s in tiny base small medium large-v3; do $PY src/layer_sweep.py --dataset $d --mert_model whisper-$s --chunk_sec 30; done
done
T A_handcrafted_svm        --dataset A --features handcrafted --clf svm --C 0.3 1 3 10
T A_mert_L8_smallC         --dataset A --features mert --layers 8 --C 0.0003 0.001 0.003 0.01
T A_mert_L6-8-10           --dataset A --features mert --layers 6,8,10 --C 0.0003 0.001 0.003 0.01
T A_wt_L1                  --dataset A --features $WT --layers 7 --C 0.001 0.003 0.01 0.03 0.1
T A_mert330_L5-6-14        --dataset A --features mert --mert_model MERT-v1-330M --layers 5,6,14 $C4
T A_mert330_L5-6-14_nostd  --dataset A --features mert --mert_model MERT-v1-330M --layers 5,6,14 --no_std --C 0.001 0.003 0.01 0.03
T A_mert330_L5-6-14_svm    --dataset A --features mert --mert_model MERT-v1-330M --layers 5,6,14 --clf svm --C 0.3 1 3
T A_wt_L1_m330_hc          --dataset A --features $WT+$M330+handcrafted --layers 7 $C4
T B_handcrafted_svm        --dataset B --features handcrafted --clf svm --C 0.3 1 3 10
T B_mert_L7                --dataset B --features mert --layers 7 --C 0.001 0.003 0.01 0.03
T B_mert330_L9             --dataset B --features mert --mert_model MERT-v1-330M --layers 9 --C 0.001 0.003 0.01 0.03
T B_lang_only              --dataset B --features lang --C 0.01 0.1 1 10
T B_wl_L30                 --dataset B --features $WL:30 --layers 7 --C 0.001 0.003 0.01 0.03
T B_wl_L30_mert7           --dataset B --features $WL:30+mert --layers 7 $C4
T B_wl_L30_mert7_lang      --dataset B --features $WL:30+mert+lang --layers 7 $C4

# ------------------------------------------------------------------ 4. optional experiment: input length (best models, first k seconds)
for s in 5 10 15; do
  $PY src/extract_whisper.py --dataset A --size tiny --seconds $s               # [GPU]
  $PY src/extract_whisper.py --dataset B --size large-v3 --seconds $s           # [GPU]
  $PY src/extract_lang.py --dataset B --seconds $s                              # [GPU]
done
for s in 5 10 15 30; do
  k=$((s/5)); LM=whisper-small_${s}s; [ $s = 30 ] && LM=whisper-small
  T A_best_${s}s --dataset A --features mert@features/A_mert_whisper-tiny_${s}s.npz:1+$M330 --layers 7 --n_chunks $k $C4
  T B_best_${s}s --dataset B --features mert@features/B_mert_whisper-large-v3_${s}s.npz:23,30+mert+lang --lang_model $LM --layers 7 --n_chunks $k $C4
done

# ------------------------------------------------------------------ 5. optional experiments: multiple excerpts, regression, t-SNE
$PY src/train_chunks.py --dataset A --layers 6,8,10
$PY src/train_chunks.py --dataset A --mert_model MERT-v1-330M --layers 5,6,14 --name A_chunks330
$PY src/train_chunks.py --dataset B --layers 7
$PY src/train_chunks.py --dataset B --mert_model MERT-v1-330M --layers 8,9,12 --name B_chunks330
T B_mert330_L8-9-12 --dataset B --features mert --mert_model MERT-v1-330M --layers 8,9,12 $C4
$PY src/regression.py --features $WT+$M330 --layers 7 --name A_best          # ridge regression and hierarchical prediction
$PY src/tsne.py --dataset A --features $WT+$M330 --layers 7 --model results/A_wt_L1_m330.joblib --name A_best_logits
$PY src/tsne.py --dataset B --features $WL:23,30+mert+lang --layers 7 --name B_best

# ------------------------------------------------------------------ 6. optional experiment: mixture / vocals / accompaniment (Task 2)
$PY src/separate.py --dataset B                                                 # [GPU] -> dataset_B_vocals/, dataset_B_accomp/
for st in vocals accomp; do
  $PY src/extract_whisper.py --dataset B --size large-v3 --data_root dataset_B_$st --out features/B_wl_${st}_30s.npz      # [GPU]
  $PY src/extract_mert.py --dataset B --data_root dataset_B_$st --out features/B_mert_${st}_MERT-v1-95M_5s.npz            # [GPU]
  $PY src/extract_lang.py --dataset B --data_root dataset_B_$st --out features/B_lang_${st}_whisper-small.npz             # [GPU]
done
WV=mert@features/B_wl_vocals_30s.npz:23,30; WA=mert@features/B_wl_accomp_30s.npz:23,30
MV=mert@features/B_mert_vocals_MERT-v1-95M_5s.npz:7; MA=mert@features/B_mert_accomp_MERT-v1-95M_5s.npz:7
LV=lang@features/B_lang_vocals_whisper-small.npz; LA=lang@features/B_lang_accomp_whisper-small.npz
T Bsep_mixture    --dataset B --features $WL:23,30+mert+lang --layers 7 $C4
T Bsep_vocals     --dataset B --features $WV+$MV+$LV --layers 7 $C4
T Bsep_accomp     --dataset B --features $WA+$MA+$LA --layers 7 $C4
T Bsep_mix+vocals --dataset B --features $WL:23,30+mert+lang+$WV+$MV+$LV --layers 7 $C4
T Bsep_all3       --dataset B --features $WL:23,30+mert+lang+$WV+$MV+$LV+$WA+$MA+$LA --layers 7 $C4

# ------------------------------------------------------------------ 7. ablation: fine-tuning   [GPU, hours]
for d in A B; do
  $PY src/finetune_mert.py --dataset $d --epochs 8  --name ft_$d
  $PY src/finetune_mert.py --dataset $d --epochs 40 --name ft40_$d
  $PY src/finetune_mert.py --dataset $d --model m-a-p/MERT-v1-330M --epochs 40 --ckpt_every 5 --name ft40_330M_$d
  for s in tiny base small; do $PY src/finetune_whisper.py --dataset $d --size $s --epochs 20; done
  $PY src/finetune_whisper.py --dataset $d --size medium --epochs 20 --bs 2
done

# ------------------------------------------------------------------ 8. audio language model   [GPU, 20 GB]
$PY src/alm_qwen2audio.py --dataset A --splits validation,test --prompt naive
$PY src/alm_qwen2audio.py --dataset B --splits validation,test --prompt naive
$PY src/alm_qwen2audio.py --dataset A --splits validation,test --prompt cues
$PY src/alm_qwen2audio.py --dataset B --splits validation,test --prompt cues
$PY src/alm_qwen2audio.py --dataset A --splits validation,test --prompt production

# ------------------------------------------------------------------ 9. report figures
$PY src/make_slide_figs.py
(cd slides && latexmk -xelatex main.tex)
