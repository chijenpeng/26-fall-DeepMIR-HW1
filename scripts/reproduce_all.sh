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
WM=mert@features/B_mert_whisper-medium_30s.npz:12; M8=mert@features/B_mert_MERT-v1-330M_5s.npz:8
T A_wt_L1_m330            --dataset A --features $WT+$M330 --layers 7 $C4
T B_wm_L12_m330L8_lang    --dataset B --features $WM+$M8+lang --layers 7 $C4
# export the two classifiers and predict the test split from the features extracted above
mkdir -p checkpoints/best submission
cp results/A_wt_L1_m330.joblib checkpoints/best/task1_decade.joblib
cp results/B_wm_L12_m330L8_lang.joblib checkpoints/best/task2_market.joblib
$PY src/inference.py --data_A dataset_A --data_B dataset_B --work_dir features --skip_extract --out submission/r14725022.json

# ------------------------------------------------------------------ 3. ablation: features, layers, sizes, pooling, classifier
# Every logistic regression below uses the same grid C4 = {3e-4, 1e-3, 3e-3, 1e-2}; the SVM rows use {0.3, 1, 3, 10}.
for d in A B; do
  $PY src/layer_sweep.py --dataset $d --mert_model MERT-v1-95M $C4
  $PY src/layer_sweep.py --dataset $d --mert_model MERT-v1-330M $C4
  for s in tiny base small medium large-v3; do $PY src/layer_sweep.py --dataset $d --mert_model whisper-$s --chunk_sec 30 $C4; done
done
# Reference rows and the rows used by Ablation (3)
T A_handcrafted_svm        --dataset A --features handcrafted --clf svm --C 0.3 1 3 10
T A_final_nostd            --dataset A --features $WT+$M330 --layers 7 --no_std $C4              # final model, mean pooling only
T A_final_svm              --dataset A --features $WT+$M330 --layers 7 --clf svm --C 0.3 1 3 10   # final model, RBF SVM
T B_final_nostd            --dataset B --features $WM+$M8+lang --layers 7 --no_std $C4
T B_final_svm              --dataset B --features $WM+$M8+lang --layers 7 --clf svm --C 0.3 1 3 10
T A_wt_L1_m330_hc          --dataset A --features $WT+$M330+handcrafted --layers 7 $C4
T B_handcrafted_svm        --dataset B --features handcrafted --clf svm --C 0.3 1 3 10
T B_lang_only              --dataset B --features lang $C4
T B_wm_L12_m330L8         --dataset B --features $WM+$M8 --layers 7 $C4              # the final model without language ID

# Model selection grid, Ablation (1a) and (1b): the best Whisper layer combined with three MERT layer sets per MERT size.
# best1 = best layer, top3 = three best layers, third = best layer of the shallow, middle and deep third (all read off the layer sweeps above).
# Task 1: Whisper-tiny layer 1 is the best single Whisper layer of all five sizes.
T sel_A_W --dataset A --features $WT --layers 7 $C4
for s in 95M:7:best1 95M:6,7,9:top3 95M:2,7,9:third 330M:5:best1 330M:5,6,14:top3 330M:5,14,23:third; do
  IFS=: read m l tag <<< "$s"; M=mert@features/A_mert_MERT-v1-${m}_5s.npz:$l
  T sel_A_${m}_${tag}   --dataset A --features $M --layers 7 $C4
  T sel_A_${m}_${tag}_W --dataset A --features $WT+$M --layers 7 $C4
done
# Task 2: eight Whisper layers tie in top-1; the tie is broken by top-3 -> Whisper-medium layer 12. Every combined cell also has the language ID.
T sel_B_Wm12      --dataset B --features $WM --layers 7 $C4
T sel_B_Wm12_lang --dataset B --features $WM+lang --layers 7 $C4
for s in 95M:7:best1 95M:3,4,7:top3 95M:4,7,10:third 330M:8:best1 330M:8,9,12:top3 330M:8,9,17:third; do
  IFS=: read m l tag <<< "$s"; M=mert@features/B_mert_MERT-v1-${m}_5s.npz:$l
  T sel_B_${m}_${tag}           --dataset B --features $M --layers 7 $C4
  T sel_B_${m}_${tag}_Wm12_lang --dataset B --features $WM+$M+lang --layers 7 $C4
done

# ------------------------------------------------------------------ 4. optional experiment: input length (best models, first k seconds)
for s in 5 10 15; do
  $PY src/extract_whisper.py --dataset A --size tiny --seconds $s               # [GPU]
  $PY src/extract_whisper.py --dataset B --size medium --seconds $s             # [GPU]
  $PY src/extract_lang.py --dataset B --seconds $s                              # [GPU]
done
for s in 5 10 15 30; do
  k=$((s/5)); LM=whisper-small_${s}s; [ $s = 30 ] && LM=whisper-small
  T A_best_${s}s --dataset A --features mert@features/A_mert_whisper-tiny_${s}s.npz:1+$M330 --layers 7 --n_chunks $k $C4
  T B_best_${s}s --dataset B --features mert@features/B_mert_whisper-medium_${s}s.npz:12+$M8+lang --lang_model $LM --layers 7 --n_chunks $k $C4
done

# ------------------------------------------------------------------ 5. optional experiments: multiple excerpts, regression, t-SNE
# chunk-level training on the MERT layer set of each final model, for both MERT sizes (clip-level counterparts: sel_* runs above)
$PY src/train_chunks.py --dataset A --layers 6,7,9 $C4
$PY src/train_chunks.py --dataset A --mert_model MERT-v1-330M --layers 5,6,14 --name A_chunks330 $C4
$PY src/train_chunks.py --dataset B --layers 7 $C4
$PY src/train_chunks.py --dataset B --mert_model MERT-v1-330M --layers 8 --name B_chunks330_L8 $C4
$PY src/regression.py --features $WT+$M330 --layers 7 --name A_best          # ridge regression and hierarchical prediction
$PY src/tsne.py --dataset A --features $WT+$M330 --layers 7 --model results/A_wt_L1_m330.joblib --name A_best_logits
$PY src/tsne.py --dataset B --features $WM+$M8+lang --layers 7 --name B_best

# ------------------------------------------------------------------ 6. optional experiment: mixture / vocals / accompaniment (Task 2)
$PY src/separate.py --dataset B                                                 # [GPU] -> dataset_B_vocals/, dataset_B_accomp/
for st in vocals accomp; do
  $PY src/extract_whisper.py --dataset B --size medium --data_root dataset_B_$st --out features/B_wm_${st}_30s.npz        # [GPU]
  $PY src/extract_mert.py --dataset B --model m-a-p/MERT-v1-330M --data_root dataset_B_$st --out features/B_mert_${st}_MERT-v1-330M_5s.npz   # [GPU]
  $PY src/extract_lang.py --dataset B --data_root dataset_B_$st --out features/B_lang_${st}_whisper-small.npz             # [GPU]
done
WV=mert@features/B_wm_vocals_30s.npz:12; WA=mert@features/B_wm_accomp_30s.npz:12
MV=mert@features/B_mert_vocals_MERT-v1-330M_5s.npz:8; MA=mert@features/B_mert_accomp_MERT-v1-330M_5s.npz:8
LV=lang@features/B_lang_vocals_whisper-small.npz; LA=lang@features/B_lang_accomp_whisper-small.npz
T Bsep_mixture    --dataset B --features $WM+$M8+lang --layers 7 $C4
T Bsep_vocals     --dataset B --features $WV+$MV+$LV --layers 7 $C4
T Bsep_accomp     --dataset B --features $WA+$MA+$LA --layers 7 $C4
T Bsep_mix+vocals --dataset B --features $WM+$M8+lang+$WV+$MV+$LV --layers 7 $C4
T Bsep_all3       --dataset B --features $WM+$M8+lang+$WV+$MV+$LV+$WA+$MA+$LA --layers 7 $C4

# ------------------------------------------------------------------ 7. ablation: fine-tuning   [GPU, hours]
for d in A B; do
  $PY src/finetune_mert.py --dataset $d --epochs 8  --name ft_$d
  $PY src/finetune_mert.py --dataset $d --epochs 40 --name ft40_$d
  $PY src/finetune_mert.py --dataset $d --model m-a-p/MERT-v1-330M --epochs 40 --ckpt_every 5 --name ft40_330M_$d
  for s in tiny base small; do $PY src/finetune_whisper.py --dataset $d --size $s --epochs 20; done
  $PY src/finetune_whisper.py --dataset $d --size medium --epochs 20 --bs 2
done

# averaging a fine-tuned model with the final frozen model (validation only)
$PY src/ensemble.py --dataset A --frozen results/A_wt_L1_m330.joblib --features $WT+$M330 --layers 7 --finetuned results/ft40_A_proba.npz
$PY src/ensemble.py --dataset B --frozen results/B_wm_L12_m330L8_lang.joblib --features $WM+$M8+lang --layers 7 --finetuned results/ft40_B_proba.npz
$PY src/ensemble.py --dataset B --frozen results/B_wm_L12_m330L8_lang.joblib --features $WM+$M8+lang --layers 7 --finetuned results/ftw_medium_B_proba.npz

# ------------------------------------------------------------------ 8. audio language model   [GPU, 20 GB]
$PY src/alm_qwen2audio.py --dataset A --splits validation,test --prompt naive
$PY src/alm_qwen2audio.py --dataset B --splits validation,test --prompt naive
$PY src/alm_qwen2audio.py --dataset A --splits validation,test --prompt cues
$PY src/alm_qwen2audio.py --dataset B --splits validation,test --prompt cues
$PY src/alm_qwen2audio.py --dataset A --splits validation,test --prompt production

# ------------------------------------------------------------------ 9. report figures
$PY src/make_slide_figs.py
(cd slides && latexmk -xelatex main.tex)
