# Task A (release-decade) — observations before modelling

Scope: **train split only** (1,026 clips, 171 per decade). Validation and test were not
touched. Scripts and raw numbers live in `eda/` (`spectral_eda.py`, `dynamics_eda.py`,
`trainA_*_feats.csv`, `ltas_by_decade.png`, `boxplots_by_decade.png`).

## Data facts

- Middle 30 s of each recording, WAV mono 16-bit 24 kHz, decoded from an Opus (YouTube) source.
  Nothing above 12 kHz exists, and Opus has already thrown away some high-frequency detail.
- Balanced classes, artist-disjoint splits. Train / val / test = 1,026 / 132 / 132.
- README: no loudness normalisation was applied, so absolute level (RMS) is a real property of
  the master, not of the preprocessing.

## Hypothesis 1 — "older music has less low end" (vinyl cutting limits)

Long-term average spectrum (Welch, 4096-pt) per decade, energy per band relative to total.

| band (median, dB re. total) | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s |
|---|---|---|---|---|---|---|
| 20–60 Hz (sub-bass)         | -21.7 | -17.4 | -16.0 | -14.5 | -15.0 | -13.1 |
| 60–120 Hz (bass)            | -8.6  | -8.0  | -8.1  | -7.6  | -7.6  | -7.1  |
| 8–12 kHz (air)              | -28.6 | -28.8 | -24.1 | -24.2 | -24.8 | -25.2 |

- **Partly true, and only for sub-bass.** 20–60 Hz rises ~8.6 dB from the 1960s to the 2010s,
  roughly monotonically (Spearman ρ = 0.30 vs decade index). The 1960s LTAS rolls off steeply
  below ~80 Hz.
- **60–120 Hz does not differ** (ANOVA p = 0.45). There is no sign that old masters compensate
  for missing sub-bass with more upper bass; the 1960s simply have less low end overall.
- High end: 1960s/70s have 4–5 dB less 8–12 kHz than 1980s+. The 1980s are the brightest decade.
- Caveat: within-decade spread (σ ≈ 6–9 dB) is larger than the between-decade shift, so the
  sub-bass feature alone gives only 22.7 % CV top-1 (chance 16.7 %).

## Hypothesis 2 — loudness war / dynamics

| feature (median) | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s | ANOVA F |
|---|---|---|---|---|---|---|---|
| RMS (dBFS)                             | -14.0 | -14.7 | -14.5 | -13.0 | -11.0 | -10.6 | 33.8 |
| crest factor (dB)                      | 13.9  | 14.5  | 14.5  | 13.0  | 11.0  | 10.6  | 38.9 |
| DR, top-20 % blocks (peak−RMS, dB)     | 10.6  | 11.2  | 11.3  | 10.7  | 8.9   | 8.7   | 47.9 |
| 400 ms block crest, median (dB)        | 11.7  | 12.1  | 12.2  | 11.7  | 10.6  | 10.3  | 46.5 |
| longest run pinned at ceiling (samples)| 5     | 5     | 6     | 9     | 22    | 26    | 43.4 |
| samples within 0.5 dB of peak (%)      | 0.02  | 0.01  | 0.02  | 0.06  | 0.42  | 0.50  | 26.1 |
| waveform kurtosis                      | 0.96  | 1.21  | 1.37  | 1.06  | 0.52  | 0.52  | 9.7  |

- **The transient loss is in the *new* decades, not the old ones.** 2000s/2010s masters are
  ~3 dB louder, have ~3 dB less crest, and show a brick-wall-limiter signature: many samples
  pinned at the ceiling for 20+ consecutive samples, flattened amplitude distribution.
- These are the strongest single features in the whole EDA (F ≈ 40–48).
- **Onset / attack-shape features are *not* discriminative** (onset-strength mean, peak/median,
  kurtosis, attack rise rate, spectral-flux statistics: F = 2–10, and the sign is opposite to the
  intuition). Drum hits in modern masters are still sharp; they are just clipped to the same
  height as everything else. What matters is peak-to-RMS headroom, not attack shape.
- **Macro-dynamics are useless here** (loudness range, 3 s-block loudness std: F < 3). A
  30 s middle excerpt does not contain verse/chorus contrast.

## What the hand-crafted features can and cannot do

5-fold CV on train, StandardScaler + logistic regression:

| feature set | top-1 | top-3 |
|---|---|---|
| sub-bass ratio only              | 0.227 | – |
| crest factor only                | 0.239 | 0.665 |
| 15 spectral features             | 0.314 | 0.727 |
| 16 dynamics features             | 0.307 | 0.716 |
| spectral + dynamics (31)         | 0.317 | 0.750 |
| chance                           | 0.167 | 0.500 |

- Spectral and dynamics features are **redundant**: both measure "production era". Combining
  them adds only +0.3 % top-1.
- Confusion structure: two clusters — {1960s, 1970s, 1980s} and {2000s, 2010s}. Errors are
  mostly between neighbouring decades. **1990s is the hardest class** (28–34 / 171 correct);
  it overlaps both clusters.
- Conclusion: production features plateau around 32 % top-1. Going further requires features
  that describe musical *content* (timbre of instruments, arrangement, genre), i.e. a
  pretrained audio encoder (MERT). The hand-crafted block is kept as a small complementary
  feature group and as an interpretable baseline.

## Selected hand-crafted features (24 dims, `src/handcrafted.py`)

8 band energies (incl. sub 20–60 and bass 60–120), below-100 Hz ratio, 60–120 / 20–60 tilt,
centroid, rolloff 85/99 %, RMS, crest, near-peak ratio, waveform kurtosis, longest ceiling
run, block crest median, DR top-20 %, LRA proxy, 400 ms loudness std, band-limited crest
(60–250 Hz and 2–8 kHz). Onset-based features were dropped.

## Planned experiments

1. MERT-v1-95M frozen, 6 × 5 s chunks, per-layer mean + std pooled to one vector per recording.
   Per-layer sweep on validation; logistic regression / SVM / MLP.
2. Ablation: handcrafted only vs MERT only vs concatenation.
3. Input-length ablation: first 1 / 2 / 3 / 6 chunks (5 / 10 / 15 / 30 s).
4. Confusion analysis: neighbour-error rate, t-SNE of the chosen layer coloured by decade.
5. Optional (GPU box): MERT-v1-330M, short-chunk CNN on log-mel as a from-scratch comparison.
