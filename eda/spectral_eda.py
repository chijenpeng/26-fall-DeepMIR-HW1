import numpy as np, pandas as pd, soundfile as sf, sys
from scipy.signal import welch
from concurrent.futures import ProcessPoolExecutor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = str(HERE.parent / "dataset_A")
OUT  = sys.argv[1] if len(sys.argv) > 1 else str(HERE)      # output folder (default: eda/)
man = pd.read_csv(f"{ROOT}/manifest.csv")
tr = man[man.split == "train"].reset_index(drop=True)

BANDS = {"sub_20_60":(20,60),"bass_60_120":(60,120),"low_120_250":(120,250),
         "lowmid_250_500":(250,500),"mid_500_2k":(500,2000),"himid_2k_4k":(2000,4000),
         "high_4k_8k":(4000,8000),"air_8k_12k":(8000,12000)}

def feats(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    f, P = welch(x, fs=sr, nperseg=4096, noverlap=2048)
    tot = P[(f>=20)].sum()
    d = {}
    for k,(lo,hi) in BANDS.items():
        d[k] = 10*np.log10(P[(f>=lo)&(f<hi)].sum()/tot + 1e-12)   # dB rel. to total
    d["below100_vs_rest_dB"] = 10*np.log10(P[(f>=20)&(f<100)].sum()/P[(f>=100)].sum()+1e-12)
    pf = P/ P.sum()
    d["centroid_hz"] = float((f*pf).sum())
    c = np.cumsum(pf); d["rolloff85_hz"] = float(f[np.searchsorted(c,0.85)]); d["rolloff99_hz"] = float(f[np.searchsorted(c,0.99)])
    rms = np.sqrt((x**2).mean()); d["rms_dBFS"] = 20*np.log10(rms+1e-9)
    d["crest_dB"] = 20*np.log10(np.abs(x).max()/(rms+1e-9))
    # short-term loudness variance (dynamics): 400ms RMS blocks
    n = int(0.4*sr); blk = x[:len(x)//n*n].reshape(-1,n); brms = 20*np.log10(np.sqrt((blk**2).mean(1))+1e-9)
    d["dyn_std_dB"] = float(brms.std())
    return d, np.log10(P+1e-14)

if __name__ == "__main__":
    paths = [f"{ROOT}/{p}" for p in tr.audio_path]
    with ProcessPoolExecutor() as ex: res = list(ex.map(feats, paths, chunksize=16))
    F = pd.DataFrame([r[0] for r in res]); F.insert(0,"label",tr.label.values); F.insert(0,"sample_id",tr.sample_id.values)
    F.to_csv(f"{OUT}/trainA_spectral_feats.csv", index=False)
    spec = np.stack([r[1] for r in res]); f = welch(np.zeros(24000*30), fs=24000, nperseg=4096, noverlap=2048)[0]
    decs = sorted(F.label.unique())
    pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
    print("\n=== MEAN per decade (band energies in dB relative to total; higher = more of that band) ===")
    print(F.groupby("label").mean(numeric_only=True).round(2).T)
    print("\n=== MEDIAN per decade ===")
    print(F.groupby("label").median(numeric_only=True).round(2).T)
    print("\n=== STD per decade ===")
    print(F.groupby("label").std(numeric_only=True).round(2).T)
    # effect sizes
    from scipy.stats import f_oneway, spearmanr
    idx = F.label.map({d:i for i,d in enumerate(decs)})
    rows=[]
    for c in F.columns[2:]:
        Fst,p = f_oneway(*[F.loc[F.label==d,c] for d in decs]); rho,_ = spearmanr(idx, F[c])
        rows.append((c, round(Fst,1), f"{p:.1e}", round(rho,3)))
    print("\n=== ANOVA F / p across decades, Spearman rho vs decade order ===")
    print(pd.DataFrame(rows, columns=["feature","F","p","spearman_rho"]).sort_values("F",ascending=False).to_string(index=False))
    # CV with handcrafted features only
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
    from sklearn.metrics import confusion_matrix, top_k_accuracy_score
    X = F.iloc[:,2:].values; y = idx.values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(clf, X, y, cv=cv).mean()
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    print(f"\n=== 5-fold CV on train, {X.shape[1]} handcrafted features, logistic regression ===")
    print(f"top1 = {acc:.3f}  top3 = {top_k_accuracy_score(y, proba, k=3):.3f}   (chance top1 = 0.167, top3 = 0.5)")
    print("confusion (rows=true, cols=pred):", decs); print(confusion_matrix(y, proba.argmax(1)))
    low_only = F[["below100_vs_rest_dB"]].values
    print(f"low-end feature ALONE top1 = {cross_val_score(clf, low_only, y, cv=cv).mean():.3f}")
    # plot
    fig,ax = plt.subplots(1,2,figsize=(14,5))
    for d in decs:
        m = spec[F.label==d].mean(0); m = m - m[(f>=20)].max()
        ax[0].semilogx(f, 10*m, label=d); ax[1].semilogx(f, 10*m, label=d)
    ax[0].set_xlim(20,12000); ax[0].set_title("Mean long-term spectrum per decade (train, dB, peak-normalized)"); ax[0].set_xlabel("Hz"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xlim(20,400); ax[1].set_title("Zoom: 20-400 Hz"); ax[1].set_xlabel("Hz"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/ltas_by_decade.png", dpi=110)
    fig,ax = plt.subplots(1,3,figsize=(15,4))
    for a,c in zip(ax,["below100_vs_rest_dB","rolloff85_hz","crest_dB"]):
        a.boxplot([F.loc[F.label==d,c] for d in decs], labels=decs); a.set_title(c); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/boxplots_by_decade.png", dpi=110)
