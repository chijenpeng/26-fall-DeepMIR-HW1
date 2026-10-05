import numpy as np, pandas as pd, soundfile as sf, librosa, sys, warnings
warnings.filterwarnings("ignore")
from scipy.stats import kurtosis, f_oneway, spearmanr
from scipy.signal import butter, sosfilt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=str(HERE.parent/"dataset_A"); OUT=sys.argv[1] if len(sys.argv)>1 else str(HERE)   # output folder (default: eda/)
man=pd.read_csv(f"{ROOT}/manifest.csv"); tr=man[man.split=="train"].reset_index(drop=True)

def db(v): return 20*np.log10(v+1e-9)
def feats(path):
    x,sr=sf.read(path,dtype="float32");
    if x.ndim>1: x=x.mean(1)
    pk=np.abs(x).max(); rms=np.sqrt((x**2).mean()); d={}
    d["crest_dB"]=db(pk)-db(rms)
    # --- brickwall / clipping signatures ---
    d["near_peak_ratio_pct"]=100*np.mean(np.abs(x)>pk*10**(-0.5/20))     # samples within 0.5 dB of max
    d["amp_kurtosis"]=kurtosis(x)                                           # Gaussian=0; limited -> lower
    s=(np.abs(x)>pk*0.98).astype(np.int8); e=np.diff(np.r_[0,s,0]); st=np.flatnonzero(e==1); en=np.flatnonzero(e==-1)
    d["max_flat_run_samples"]=float((en-st).max()) if len(st) else 0.0   # consecutive samples pinned at ceiling
    # --- macro / micro dynamics (400 ms & 3 s blocks) ---
    def blocks(n): b=x[:len(x)//n*n].reshape(-1,n); return b
    b=blocks(int(0.4*sr)); brms=db(np.sqrt((b**2).mean(1))); bpk=db(np.abs(b).max(1))
    d["block_crest_median_dB"]=np.median(bpk-brms)                         # micro-dynamics (PSR-like)
    d["LRA_proxy_dB"]=np.percentile(brms,95)-np.percentile(brms,10)         # macro loudness range
    top=np.argsort(brms)[-len(brms)//5:]; d["DR_top20_dB"]=np.mean(bpk[top]-brms[top])   # TT-DR-meter style
    b3=blocks(3*sr); d["macro_std_3s_dB"]=db(np.sqrt((b3**2).mean(1))).std()
    # --- transient / onset features ---
    oenv=librosa.onset.onset_strength(y=x,sr=sr,hop_length=256)
    d["onset_mean"]=oenv.mean(); d["onset_peak_to_median"]=oenv.max()/(np.median(oenv)+1e-6)
    d["onset_kurtosis"]=kurtosis(oenv)                                     # spiky onsets -> high
    # attack sharpness: envelope rise rate at onsets (dB per ms)
    env=librosa.feature.rms(y=x,frame_length=512,hop_length=128)[0]; envdb=db(env); fr=sr/128
    on=librosa.onset.onset_detect(onset_envelope=oenv,sr=sr,hop_length=256,units="frames")*2
    rises=[]
    for o in on:
        lo=max(0,o-int(0.05*fr)); hi=min(len(envdb),o+int(0.05*fr))
        if hi-lo>3: rises.append((envdb[lo:hi].max()-envdb[lo:hi].min())/(0.1*1000))
    d["attack_rise_dB_per_ms"]=np.mean(rises) if rises else 0.0
    # --- multi-band crest: transients live in highs ---
    for name,(lo,hi) in {"crest_low_60_250":(60,250),"crest_high_2k_8k":(2000,8000)}.items():
        sos=butter(4,[lo,hi],btype="band",fs=sr,output="sos"); xb=sosfilt(sos,x)
        d[name]=db(np.abs(xb).max())-db(np.sqrt((xb**2).mean()))
    # --- spectral flux spikiness ---
    S=np.abs(librosa.stft(x,n_fft=1024,hop_length=256)); flux=np.sqrt((np.diff(S,axis=1).clip(0)**2).sum(0))
    d["flux_kurtosis"]=kurtosis(flux); d["flux_p95_over_median"]=np.percentile(flux,95)/(np.median(flux)+1e-9)
    return d

if __name__=="__main__":
    with ProcessPoolExecutor() as ex: res=list(ex.map(feats,[f"{ROOT}/{p}" for p in tr.audio_path],chunksize=8))
    F=pd.DataFrame(res); F.insert(0,"label",tr.label.values); F.insert(0,"sample_id",tr.sample_id.values)
    F.to_csv(f"{OUT}/trainA_dynamics_feats.csv",index=False)
    decs=sorted(F.label.unique()); idx=F.label.map({d:i for i,d in enumerate(decs)})
    pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
    print("=== MEDIAN per decade ===\n", F.groupby("label").median(numeric_only=True).round(2).T)
    rows=[]
    for c in F.columns[2:]:
        Fs,p=f_oneway(*[F.loc[F.label==d,c] for d in decs]); rho,_=spearmanr(idx,F[c]); rows.append((c,round(Fs,1),f"{p:.1e}",round(rho,3)))
    print("\n=== ANOVA F / Spearman rho vs decade ===\n", pd.DataFrame(rows,columns=["feature","F","p","rho"]).sort_values("F",ascending=False).to_string(index=False))
    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline; from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
    from sklearn.metrics import top_k_accuracy_score, confusion_matrix
    cv=StratifiedKFold(5,shuffle=True,random_state=0); y=idx.values
    def ev(X,name):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)); pr=cross_val_predict(clf,X,y,cv=cv,method="predict_proba")
        print(f"{name:45s} top1={np.mean(pr.argmax(1)==y):.3f} top3={top_k_accuracy_score(y,pr,k=3):.3f}"); return pr
    ev(F[["crest_dB"]].values,"crest only")
    pr=ev(F.iloc[:,2:].values,f"dynamics features ({F.shape[1]-2})")
    old=pd.read_csv(f"{OUT}/trainA_spectral_feats.csv"); assert (old.sample_id==F.sample_id).all()
    ev(old.iloc[:,2:].values,"spectral features (15, previous)")
    pr=ev(np.hstack([old.iloc[:,2:].values,F.iloc[:,2:].values]),"spectral + dynamics")
    print(decs); print(confusion_matrix(y,pr.argmax(1)))
