# 報告大綱與素材對照(paper style)

所有數字都是 validation(A 132 首、B 102 首)。詳細表格與討論在 `experiments.md`,
建模前的資料觀察在 `observations_taskA.md`。圖檔都在 `results/`。

## 0. 實驗設定

| | Server(主要) | Mac(前期) |
|---|---|---|
| CPU / RAM | Intel i9-14900K(32 執行緒),125 GB | Apple M3,16 GB |
| GPU | NVIDIA RTX A4500 20 GB × 2 | MPS |
| 系統 | Ubuntu 26.04,driver 595.84 | macOS 26.2 |
| 套件 | Python 3.11,torch 2.14.1+cu126,transformers 4.45.2,scikit-learn 1.9.1,librosa 0.11,demucs 4.1 | torch 2.5.0,transformers 4.45.2,scikit-learn 1.5.2 |
| 用途 | MERT-330M、Whisper、Demucs、ALM、所有 fine-tune、最終模型訓練與預測 | 手工特徵、MERT-95M 抽特徵、前期分類器 |

共通流程:30 秒音訊(24 kHz mono)→ 凍結預訓練編碼器 → 時間軸 mean + std pooling →
StandardScaler → L2 logistic regression(C 在 validation 上挑)→ softmax 前三名。
train 擬合、validation 選模型、test 只產生預測檔。

## 1. 主結果(只放兩個最佳模型)

| 任務 | 模型 | top-1 | top-3 | 圖 |
|---|---|---|---|---|
| Task 1 年代 | Whisper-tiny 第 1 層 + MERT-330M 第 5、6、14 層(6,912 維,C=0.0003) | .523 | .864 | `A_wt_L1_m330_cm.png` |
| Task 2 市場 | Whisper-large-v3 第 23、30 層 + MERT-95M 第 7 層 + Whisper-small 語言後驗(6,663 維,C=0.001) | .608 | .902 | `B_wl_L23-30_mert7_lang_cm.png` |

混淆矩陣(counts,rows = true):

```
Task 1    60s 70s 80s 90s 00s 10s  recall     Task 2     US  UK  BR  ES  DE  IT  recall
1960s  [  17   2   0   3   0   0 ]  .77        US      [  11   4   0   0   1   1 ]  .65
1970s  [   5   9   3   2   2   1 ]  .41        UK      [   5   9   0   0   2   1 ]  .53
1980s  [   2   2  13   2   3   0 ]  .59        Brazil  [   0   0  15   0   0   2 ]  .88
1990s  [   1   3   2   9   4   3 ]  .41        Spain   [   0   2   0  14   1   0 ]  .82
2000s  [   2   2   1   5   5   7 ]  .23        Germany [   3   4   1   2   6   1 ]  .35
2010s  [   0   2   0   0   4  16 ]  .73        Italy   [   1   4   0   2   3   7 ]  .41
```

錯誤分析:

- **Task 1**:63 個錯誤中 57% 是相鄰年代、22% 差兩個年代、21% 差三個以上。兩端(1960s、2010s)最準,
  2000s 最差(recall .23),被 2010s 吸走 7 首、1990s 吸走 5 首;1970s 有 5 首被判成 1960s。
  對應 EDA:製作線索只能分出「舊 / 新」兩群,中間年代靠內容特徵。
- **Task 2**:40 個錯誤中 75% 落在 US / UK / Germany / Italy 之間,最大的一對是 US ↔ UK(9 首)。
  Brazil(.88)和 Spain(.82)幾乎不出錯。佐證:train 的語言列聯表,Germany 83%、Italy 55% 被判為英語發行。
  所以錯誤確實集中在特定市場之間:英文歌比例高的市場。

## 2. 選做實驗(PDF 第 14 頁)

| PDF 項目 | 做了什麼 | 結果 | 位置 |
|---|---|---|---|
| 5 / 10 / 15 / 30 秒 | 兩個任務,用最佳模型,對截短音訊重抽特徵並重訓 | A top-1:.394 / .470 / .492 / .523;B top-1:.451 / .520 / .578 / .608。單調上升,尚未飽和 | experiments.md「用最佳模型重做」 |
| 同一錄音多段訓練 | 兩個任務,每段 5 秒當獨立樣本,測試時平均機率,MERT-95M 與 330M | A .477 到 .500,B .431 到 .451,沒有比整首平均好 | 「chunk-level」 |
| 年份回歸 / 階層式 | Task 1,MERT-95M 三層。Ridge 回歸;三群再二分 | 回歸 .311 / .788(MAE 1.06 年代);階層式 .477 / .879 | 「把年代當序數」 |
| mixture / vocal / accompaniment | Task 2,Demucs htdemucs,各抽 MERT-95M,另對人聲跑 Whisper 語言偵測 | 混音 .422、人聲 .441、伴奏 .382;加語言後人聲 .559 | 「Source separation」 |
| t-SNE | 最佳模型的特徵空間與分類器 logit 空間,A、B 各兩張 | A 的 logit 空間呈 1960s → 2010s 的序數弧線,2000s 與 2010s 重疊;B 的特徵空間直接看得到 Brazil、Spain、Italy 三群,英文市場混成一團 | `results/{A,B}_best_tsne.png`、`{A,B}_best_logits_tsne.png` |
| 混淆與資料集偏差 | A:頻譜與動態 EDA(31 個手工特徵的 ANOVA);B:語言 × 市場列聯表 | 超低頻隨年代升 8.6 dB;2000s 後 crest 少 3 dB;手工特徵在 B 上是隨機水準 | observations_taskA.md、「語言特徵」 |

## 3. Ablation

| 主題 | 做了什麼 | 重點結果 |
|---|---|---|
| 特徵來源 | 手工 24 維、MERT-95M、MERT-330M、Whisper 五種尺寸、語言後驗,以及它們的拼接 | A:手工 .409 → MERT-95M .485 → 330M .500 → 加 Whisper-tiny .523。B:手工 .216 → MERT .422 → 語言 .539 → Whisper-large .578 → 組合 .608 |
| 逐層 sweep | MERT-95M 13 層、MERT-330M 25 層、Whisper 五種尺寸全部層 | A 的最佳在中層(MERT)或最淺層(Whisper);B 的最佳在最深層 |
| 編碼器尺寸 | Whisper tiny / base / small / medium / large-v3 凍結 | A:最小的最好(tiny L1 .470);B:最大的最好(large L30 .578,top-3 .902) |
| Frozen 對 fine-tune | 14 次微調:MERT-95M(8、40 epoch)、MERT-330M(40)、Whisper tiny 到 medium(20),A、B 都跑,每個 epoch 記 loss 並存 checkpoint | 全部在 2 到 12 epoch 過擬合,全部輸給凍結特徵組合;曲線圖 `ft40_*_curves.png`、`ftw_*_curves.png` |
| Pooling | mean 對 mean + std | 加 std 有幫助(A .462 → .500) |
| 分類器 | logreg、RBF SVM、MLP | logreg 最穩 |

## 4. Audio Language Model

- 模型:Qwen2-Audio-7B-Instruct,零樣本,bf16,單張 A4500。
- 範圍:validation + test 全部樣本(A 264 首、B 204 首)。
- 輸出:(1) 自由生成 16 token,regex 解析,解析不出記為 invalid;(2) 對六個標籤算 teacher-forced
  log-prob 排序取前三,定義上每個樣本都有合法答案。
- Prompt:`naive`(A、B)、`cues`(A、B)、`production`(A)。全文在 `src/alm_qwen2audio.py`。

| Prompt | Task 1 top-1 / top-3 | 自由生成無效率 | Task 2 top-1 / top-3 | 自由生成無效率 |
|---|---|---|---|---|
| naive | .341 / .697 | 1.5% | .529 / .735 | 1.0% |
| cues | .265 / .621 | 1.5% | .225 / .618 | 11.8% |
| production | .174 / .583 | 6.1% | 未跑 | |

- 重點:prompt 越長越被文字先驗牽著走(A 的答案塌向 1970s 或 2010s;B 的 log-prob 被 prompt 中出現的標籤字串拉高)。
- 與 Whisper 凍結 sweep 呼應:ALM 的語言模型只拿到編碼器最後一層,而製作線索只存在於最淺層。
- 混淆矩陣圖:`alm_Qwen2-Audio-7B-Instruct_{A,B}_{naive,cues,production}_cm.png`。

## 5. AI 協作聲明(Disclosure)

草稿,中英各一版,用字由你定:

> 本作業的所有程式碼(`src/`、`eda/`、`scripts/`)皆在 Claude Code(Anthropic)的協助下撰寫,
> 實驗的執行、紀錄整理與分析文字的初稿也由其協助完成。研究假設與方向(例如以超低頻、響度與動態作為年代線索、
> 以語言作為市場線索)、實驗的取捨與最終模型的選擇由作者決定,所有數字皆來自實際執行本 repo 程式碼的結果,
> 並由作者檢視確認。

> All code in this submission (`src/`, `eda/`, `scripts/`) was written with the assistance of Claude Code
> (Anthropic), which also helped run the experiments, keep the experiment log and draft the analysis text.
> The hypotheses and research direction (e.g. sub-bass, loudness and dynamics as decade cues; language as a
> market cue), the choice of experiments and the selection of the final models were made by the author.
> Every number reported here was produced by running the code in this repository and was reviewed by the author.

注意:PDF 的 Rules 只寫「可使用公開程式碼與預訓練模型並引用」,沒有提 AI 工具。若課程另有規定,以課程規定為準。

## 6. 引用(每個用到的預訓練模型、程式庫、資料集、方法)

以下書目資訊是憑記憶整理,**繳交前請對照各 model card / 論文頁面核對年份與編號**。

預訓練模型

| 用途 | 模型 | 引用 |
|---|---|---|
| A、B 的音樂特徵;fine-tune | `m-a-p/MERT-v1-95M`、`m-a-p/MERT-v1-330M` | Li et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training," ICLR 2024. arXiv:2306.00107 |
| A、B 的編碼器特徵;B 的語言後驗;fine-tune | `openai/whisper-{tiny,base,small,medium,large-v3}` | Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," ICML 2023. arXiv:2212.04356 |
| ALM | `Qwen/Qwen2-Audio-7B-Instruct` | Chu et al., "Qwen2-Audio Technical Report," 2024. arXiv:2407.10759 |
| Source separation | Demucs `htdemucs` | Rouard, Massa, Défossez, "Hybrid Transformers for Music Source Separation," ICASSP 2023. arXiv:2211.08553 |

資料集

- Discogs-VI:Araz, Serra, Bogdanov, "Discogs-VI: A Musical Version Identification Dataset Based on Public Editorial Metadata," ISMIR 2024. arXiv:2410.17400(作業資料集的來源,由助教整理釋出)

程式庫(公開 codebase)

- PyTorch:Paszke et al., NeurIPS 2019。torchaudio:Yang et al., ICASSP 2022。
- Hugging Face Transformers:Wolf et al., EMNLP 2020 (System Demonstrations)。
- scikit-learn:Pedregosa et al., JMLR 2011。
- librosa:McFee et al., SciPy 2015。
- nnAudio(MERT 的 CQT 前端依賴):Cheuk et al., IEEE Access 2020。
- Demucs 程式庫:github.com/facebookresearch/demucs(後續維護 github.com/adefossez/demucs)。
- NumPy(Harris et al., Nature 2020)、SciPy(Virtanen et al., Nature Methods 2020)、pandas、matplotlib(Hunter 2007)、seaborn(Waskom 2021)、soundfile / libsndfile、tqdm、joblib。

方法

- t-SNE:van der Maaten & Hinton, JMLR 2008。
- Welch 功率譜估計(EDA 的長期平均頻譜):Welch, IEEE Trans. Audio Electroacoust., 1967。
- Fine-tune 用到的技巧:AdamW(Loshchilov & Hutter, ICLR 2019)、one-cycle 學習率(Smith & Topin, 2019)、
  label smoothing(Szegedy et al., CVPR 2016)、SpecAugment 的時間遮罩(Park et al., Interspeech 2019)。
- 若報告討論 loudness war,可引 Vickers, "The Loudness War: Background, Speculation, and Recommendations," AES Convention 129, 2010。

工具

- Claude Code(Anthropic):見第 5 節。

沒有用到、不需引用:short-chunk CNN、Open-Unmix、Spleeter、Audio Flamingo、UMAP。

## 已知限制(報告要照實寫)

- test 標籤隱藏,只能報 validation 數字並繳交預測檔 `results/submission_best_v2.json`。
- validation 很小,A 一首 0.76%、B 一首 0.98%,前幾名之間的差距多在雜訊內。
- 回歸、階層式、多段訓練、source separation 用的是 MERT 特徵,不是最終的 Whisper + MERT 組合(輸入長度與 t-SNE 已用最佳模型重做)。
- `production` prompt 沒跑 Task 2;Task 1 沒做 source separation。
- 最終模型用 scikit-learn 1.9.1 訓練,requirements.txt 需釘版本。
