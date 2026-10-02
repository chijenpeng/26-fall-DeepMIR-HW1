# 實驗紀錄(持續更新)

所有數字都是 **validation split** 的結果(A 132 首、B 102 首)。
A 一首 = 0.76%,B 一首 = 0.98%,所以 2 到 3% 的差距都在雜訊範圍內。
每組實驗在 `results/` 都有 `<名稱>_val.json`(含混淆矩陣 counts)和 `<名稱>_cm.png`(左 counts、右 row-normalized)。

## 特徵定義

| 代號 | 內容 | 維度 | 腳本 |
|---|---|---|---|
| handcrafted | 24 維製作特徵:8 個頻帶能量比例、100 Hz 以下比例、低頻 tilt、頻譜重心、rolloff 85/99%、RMS、crest、貼近峰值比例、波形 kurtosis、連續貼頂長度、區塊 crest、DR、LRA proxy、400 ms 音量標準差、低頻帶與高頻帶 crest | 24 | `src/handcrafted.py`、`src/extract_handcrafted.py` |
| mert Lk | MERT-v1-95M 凍結,30 秒切成 6 段 5 秒,每段取第 k 層 frame 的時間平均和時間標準差,再把 6 段平均。每層 768 × 2 = 1536 維 | 1536 × 層數 | `src/extract_mert.py`、`src/features.py` |
| mert Lk(nostd) | 同上但只取時間平均 | 768 × 層數 | 同上 |
| lang | Whisper-small 語言偵測的後驗機率(en, pt, es, de, it, fr, other),取 log | 7 | `src/extract_lang.py` |
| mert@vocals / mert@accomp | Demucs(htdemucs)分離出的人聲 / 伴奏,再各自過 MERT | 同 mert | `src/separate.py` |

分類器一律先 StandardScaler,再接 logistic regression(L2)、RBF SVM 或單隱藏層 MLP;
C 在 validation 上挑。Top-3 直接取 softmax 機率前三名。

## Task A:發行年代

### 單層 sweep(mert,logreg)

| 層 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top-1 | .333 | .364 | .371 | .356 | .364 | .402 | .424 | .424 | .439 | .439 | .424 | .417 | .402 |
| top-3 | .833 | .795 | .795 | .773 | .795 | .811 | .833 | .818 | .811 | .818 | .811 | .758 | .750 |

中間層(6 到 10)最好,符合 MERT 論文的觀察:中層偏音色和製作,高層偏音高和內容。

### 主要比較

| 實驗名稱 | 特徵 | 分類器 | top-1 | top-3 | 相鄰年代錯誤比例 |
|---|---|---|---|---|---|
| A_handcrafted_logreg | handcrafted | logreg | .379 | .848 | .49 |
| A_handcrafted_svm | handcrafted | SVM | .409 | .856 | – |
| A_mert_L8_smallC | mert L8 | logreg | .439 | .848 | .59 |
| A_mert_L8_nostd | mert L8(只有平均) | logreg | .455 | .811 | .58 |
| A_mert_L8_svm | mert L8 | SVM | .424 | .826 | .54 |
| **A_mert_L6-8-10** | **mert L6+L8+L10** | **logreg** | **.485** | .818 | .59 |
| A_mert_L6-8-10_nostd | mert L6+L8+L10(只有平均) | logreg | .462 | .886 | .62 |
| A_mert_L6-8-10_svm | mert L6+L8+L10 | SVM | .477 | .826 | .54 |
| A_mert_L5-12 | mert L5 到 L12 | logreg | .470 | .856 | .61 |
| A_mert_all13 | mert 全部 13 層 | logreg | .470 | .841 | .60 |
| A_both_L8_logreg | handcrafted + mert L8 | logreg | .439 | .795 | .61 |
| A_both_L8_svm | handcrafted + mert L8 | SVM | .455 | .841 | .53 |
| A_both_L8_mlp | handcrafted + mert L8 | MLP | .424 | .780 | .63 |
| A_both_L6-8-10 | handcrafted + mert L6+L8+L10 | logreg | .477 | .811 | .62 |

### 輸入長度(mert L8,logreg,只用前 k 段 5 秒)

| 長度 | 5 s | 10 s | 15 s | 30 s |
|---|---|---|---|---|
| top-1 | .356 | .432 | .455 | .439 |
| top-3 | .727 | .750 | .765 | .848 |

### 同一錄音多段訓練(chunk-level,`src/train_chunks.py`)

把每首的 6 段 5 秒各自當一個訓練樣本(train 從 1,026 變 6,156 筆),validation 時平均六段的機率。

| 實驗名稱 | 特徵 | top-1 | top-3 |
|---|---|---|---|
| A_chunks_L6-8-10 | mert L6+L8+L10,chunk-level | .477 | .848 |
| A_chunks_L8 | mert L8,chunk-level | .477 | .841 |
| B_chunks_L4-7 | mert L4+L7,chunk-level | .431 | .755 |
| B_chunks_L7 | mert L7,chunk-level | .451 | .765 |

沒有比整首平均特徵好(A 48.5%、B 42.2%)。資料量乘六但多出來的樣本高度相關,
對線性分類器等於沒有新資訊;而且單段 5 秒的特徵比 30 秒平均更吵,訓練時反而引入雜訊。

### 把年代當序數(mert L6+L8+L10)

| 方法 | top-1 | top-3 | 備註 |
|---|---|---|---|
| Ridge 回歸到年代索引,四捨五入 | .311 | .788 | MAE 1.06 個年代;top-3 用距離排序 |
| 階層式:先分 {60s,70s} / {80s,90s} / {00s,10s} 三群,群內再二分 | .477 | .879 | 和平的六類 logreg 差不多,但 top-3 較高 |
| 階層式(handcrafted) | .394 | .856 | |

### 觀察

- 手工特徵 41% → MERT 單層 44% → 多層拼接 48.5%。手工特徵加進 MERT 沒有再進步,
  代表 MERT 中層已經含有製作年代的資訊。
- 錯誤有 54 到 62% 落在相鄰年代。混淆矩陣(A_mert_L6-8-10):
  1960s 最準(22 首對 16 首),1970s 和 1990s 最差,1970s 多被判成 1960s,1990s 散到 1980s 和 2000s。
- 回歸明顯比分類差:年代索引不是等距的「聲音距離」(1960s 到 70s 的差距遠大於 2000s 到 10s)。
- t-SNE(`results/A_mert_L6-8-10_tsne.png`):只有微弱的梯度,1960s/70s 偏一邊、2000s/10s 偏另一邊,
  中間大量混合,右下角有一小團 1960s/70s 的緊密群集。整體沒有清楚的年代群。
- MERT-95M 凍結特徵在 46 到 49% 之間是平台期。待做:MERT-330M(server)。

## Task B:發行市場(全部是 1980s)

### 單層 sweep(mert,logreg)

| 層 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top-1 | .382 | .363 | .373 | .392 | .402 | .373 | .353 | .422 | .373 | .294 | .343 | .294 | .324 |
| top-3 | .676 | .706 | .716 | .696 | .765 | .784 | .745 | .725 | .716 | .686 | .618 | .686 | .657 |

### 主要比較

| 實驗名稱 | 特徵 | 分類器 | top-1 | top-3 |
|---|---|---|---|---|
| B_handcrafted_svm | handcrafted | SVM | .216 | .588 |
| B_mert_L7 | mert L7 | logreg | .422 | .725 |
| **B_mert_L4-7** | **mert L4+L7** | **logreg** | **.422** | **.794** |
| B_mert_L3-4-5-7 | mert L3+L4+L5+L7 | logreg | .392 | .814 |
| B_mert_L0-12_all | mert 全部 13 層 | logreg | .392 | .755 |
| B_mert_L7_svm | mert L7 | SVM | .412 | .804 |
| B_mert_L4-7_svm | mert L4+L7 | SVM | .422 | .784 |
| B_both_L4-7 | handcrafted + mert L4+L7 | logreg | .422 | .794 |

### 語言特徵(Whisper-small 對混音做語言偵測)

train 上的「Whisper 判定語言 × 市場」列聯表(只看 en/pt/es/de/it 五種的 argmax):

| 市場 | 本地語言 | 判為本地語言的比例 | 判為英語的比例 |
|---|---|---|---|
| US | en | 0.92 | – |
| UK | en | 1.00 | – |
| Brazil | pt | 0.87 | 0.11 |
| Spain | es | 0.57 | 0.39 |
| Germany | de | 0.17 | 0.83 |
| Italy | it | 0.41 | 0.55 |

語言能乾淨地切出 Brazil,對 Spain 和 Italy 有一半幫助,Germany 的 1980s 發行八成是英文歌,語言幾乎幫不上。

| 實驗名稱 | 特徵 | 分類器 | top-1 | top-3 |
|---|---|---|---|---|
| B_lang_only | lang(7 維) | logreg | .539 | .794 |
| B_mert_L4-7_lang | mert L4+L7 + lang | logreg | .510 | .843 |
| **B_mert_L7_lang** | **mert L7 + lang** | **logreg** | **.559** | .775 |
| B_mert_L4-7_lang_svm | mert L4+L7 + lang | SVM | .451 | .804 |

- 7 維語言後驗單獨就贏過 1536 維的 MERT(53.9% 對 42.2%)。
- B_lang_only 混淆矩陣:Brazil 13、Spain 13、US 12 準確;Germany 有 8 首被判成 US,Italy 有 7 首被判成 UK,
  全是英文歌的問題。MERT 加語言後 Italy 升到 9,Germany 仍只有 5。
- 待做:對 Demucs 人聲 stem 重跑 Whisper(混音的伴奏會干擾語言判斷)、伴奏 stem 的 MERT 看能否分 US / UK。

### 觀察

- **手工製作特徵在 B 上等於隨機**(21.6%,隨機 16.7%)。同一年代的歌,loudness 和頻譜平衡沒有市場差異。
  所以 B 的最終模型不含手工特徵。
- 混淆矩陣(B_mert_L4-7,每類 17 首):Brazil 最準(10 首),Spain 7,Italy 7,Germany 6,
  US 8,UK 5。**US 和 UK 互混最嚴重**(UK 有 8 首被判成 US)。Brazil 和 Spain 之間也有混淆。
- t-SNE(`results/B_mert_L4-7_tsne.png`):Brazil 有一團明顯的群集,Italy 在最右邊有一小團緊密群集,
  其餘四類大量混合。
- 這支持「語言是主要線索」的假設:葡語和義語市場有可見的結構,英語市場(US、UK)分不開。
- 待做(server):Whisper 語言後驗、Demucs 人聲 / 伴奏分開過 MERT、MERT-330M。

## Audio Language Model(ALM)

### 討論:ALM 對我們找到的製作線索敏感嗎?

幾乎不敏感。ALM 回答問題的方式比較像一個人聽了之後憑曲風和常識猜,原因在它的輸入管線:

- Qwen2-Audio 的音訊編碼器是 Whisper-large。輸入是 16 kHz 的 128-bin log-mel 頻譜圖,8 kHz 以上沒有。
  每個 clip 以自己的峰值正規化,動態範圍截到 80 dB,所以**絕對音量(RMS)完全消失**,loudness war 最強的線索沒了。
- 時間解析度 10 ms 一格。我們抓到的 limiter 簽名(20 幾個 sample 貼頂,約 1 ms)在 mel 頻譜圖上看不到,
  只剩包絡層級的「整首都很平」。
- 超低頻勉強有:mel 刻度低頻 bin 很密,20 到 60 Hz 大概落在前一兩個 bin,但很粗。
- 編碼器是為語音辨識訓練的,Qwen2-Audio 再用音訊問答、描述、音樂標籤微調。學到的表徵偏「什麼語言、什麼樂器、
  什麼曲風、什麼情緒」,不是母帶特徵。
- 語言模型那層靠文字先驗:拿到「像 disco、有合成器、英文演唱」後,用讀過的文字知識推「disco 大概 1970 年代」。
  它沒被明確教過年代標籤,是在用常識推論。

預期:Task A 會比 MERT 加 logreg 差,而且會系統性把復古風格的新歌判成舊歌;Task B 會做得還可以,
因為語言和曲風正是它擅長的,但英文歌一律猜 US 或 UK,Germany 和 Italy 的英文發行分不出來,和語言特徵的弱點相同。
Audio Flamingo 用 CLAP 類編碼器(音訊配文字描述訓練),更偏語意,訓練描述裡可能有「vintage」「lo-fi」這類詞,
對錄音質感有一點敏感,但同樣看不到 sample 層級的動態。

ALM 和手工特徵是互補的兩端:一個靠語意常識,一個靠製作指紋。

### 硬體可行性(A4500 20 GB × 2、3090 24 GB)

| 模型 | 參數量 | bf16 VRAM | 備註 |
|---|---|---|---|
| Qwen2-Audio-7B-Instruct | 約 8.4B | 約 17 到 18 GB | A4500 勉強、3090 可以;transformers 4.45 原生支援 |
| Qwen2-Audio 4-bit | 同上 | 約 6 到 7 GB | 需 bitsandbytes |
| Audio Flamingo 2 | 約 3B | 約 7 到 8 GB | NVIDIA 自家 codebase,要獨立環境 |
| Audio Flamingo 3 | 約 7B | 約 15 到 17 GB | 同上 |

選 Qwen2-Audio-7B-Instruct:30 秒剛好是 Whisper 編碼器的原生視窗,環境不用另建。

### 實驗設計(`src/alm_qwen2audio.py`)

- Prompt `naive`:樸素陳述目標(「這是一段美國發行的 30 秒錄音,我們要判斷發行年代,請回答六個選項之一」),附上音訊。
- 每個 clip 做兩種輸出:(1) 自由回答(greedy,16 token 內)再用 regex 解析標籤,無法解析記為 invalid;
  (2) 對六個標籤做 teacher-forced log-prob,排序得 top-3,這種模式永遠有合法輸出。
- 先跑 validation 和 test;prompt 比較(至少兩種)之後再加。
- 硬體實測:bf16 在單張 A4500 20 GB 跑得動,沒有 OOM,每首約 2.2 秒(一次自由生成加六次標籤評分)。

### 結果:Task A,prompt `naive`(validation 132 首)

| 模式 | top-1 | top-3 | 無效輸出 |
|---|---|---|---|
| log-prob 排序六個標籤 | .341 | .697 | 0(定義上不會有) |
| 自由回答再解析 | .311 | – | 1.5%(2 首;一首答 1950s,一首無法解析) |

自由回答的分布:1970s 73 首、1960s 28、2010s 21、1980s 7、2000s 1、**1990s 0**。
混淆矩陣(log-prob 排序,rows = true):

```
          60s 70s 80s 90s 00s 10s
1960s  [  14   8   0   0   0   0 ]
1970s  [   9  13   0   0   0   0 ]
1980s  [   3  16   3   0   0   0 ]
1990s  [   2  14   4   0   1   1 ]
2000s  [   0  14   1   0   0   7 ]
2010s  [   1   6   0   0   0  15 ]
```

- 比 MERT 加 logreg(48.5%)差很多,只比手工特徵的 CV(32%)略好,top-3 甚至比手工特徵低。
- 偏誤非常明顯:模型的先驗把一切都往 1970s 推,1980s 到 2000s 幾乎全被判成 1970s,
  只有 1960s 和 2010s 兩端還能認出來。這和「靠曲風常識推論」的預期一致:
  它認得出「很舊」和「很新」的語意特徵,但沒有年代中段的製作線索。
- 兩種模式差距小(34.1% 對 31.1%),代表自由回答大致就是 log-prob 最高的那個標籤,
  但 log-prob 模式多了可靠的 top-3 和零無效輸出,報告用這個當主結果。

### 結果:Task B,prompt `naive`(validation 102 首)

| 模式 | top-1 | top-3 | 無效輸出 |
|---|---|---|---|
| log-prob 排序六個標籤 | .529 | .735 | 0 |
| 自由回答再解析 | .471 | – | 1.0%(1 首) |

自由回答的分布:US 55 首、Brazil 18、Spain 17、UK 7、Germany 2、Italy 2。
混淆矩陣(log-prob 排序,rows = true,每類 17 首):

```
          US  UK  BR  ES  DE  IT
US     [  16   1   0   0   0   0 ]
UK     [   8   9   0   0   0   0 ]
Brazil [   1   1  15   0   0   0 ]
Spain  [   2   2   2  11   0   0 ]
Germany[   8   4   1   3   1   0 ]
Italy  [   7   2   1   5   0   2 ]
```

- 比 Task A 好很多(52.9% 對 34.1%),和我們 7 維語言特徵的 53.9% 幾乎一樣,
  證實 ALM 在這題也是靠「聽出語言」在答。
- 弱點和語言特徵一模一樣:Germany 和 Italy 的英文發行全被判成 US(各 8 首和 7 首),UK 有一半被判成 US。
  它的先驗是「英文歌就是美國歌」。Italy 還有 5 首被判成 Spain,是義語和西語的混淆。
- 兩個任務合起來看:ALM 擅長語意層的線索(語言、曲風),對製作層的線索(動態、頻譜平衡)沒反應,
  所以在 A 上輸給 MERT,在 B 上和語言特徵打平但輸給 MERT 加語言(55.9%)。
### Prompt 設計(`src/alm_qwen2audio.py` 的 `PROMPTS`)

| 名稱 | 任務 | 內容 | 目的 |
|---|---|---|---|
| naive | A、B | 樸素陳述目標、附音訊、要求回答六選一 | 基準 |
| cues | A、B | 提示該聽的線索(A:製作品質、樂器、曲風;B:先辨語言再看曲風),並聲明六類等機率 | 壓掉「全答 1970s」「英文歌就是 US」的先驗 |
| production | A | 271 字的聽音指南,把 EDA 找到的四組製作線索寫給模型:60 Hz 以下超低頻與黑膠刻片的關係、8 kHz 以上高頻延伸與磁帶嘶聲、1980s 的數位亮度與 gated drums、2000s 以後的 loudness war 與 limiter 壓平 transient,最後補曲風與樂器線索 | 測試「明確告訴模型製作線索」有沒有用。依前面的分析,Whisper 編碼器看不到絕對音量和 sample 層級的動態,預期幫助有限,但這正是要驗證的 |

- `cues` 和 `production` 的結果待補(server GPU 1 排程中)。

## 繳交相關

- 保底預測檔:`results/submission_baseline.json`(A 用 A_mert_L6-8-10,B 用 B_mert_L4-7),
  格式已和官方範例核對,繳交時改名為 `<學號>.json`。
- 待辦:報告 PDF、README(推論步驟)、requirements.txt、開放存取的雲端連結。
