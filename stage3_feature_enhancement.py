
"""
STAGE 3 — FEATURE ENHANCEMENT
Same 109 DEPRESJON + PSYKOSE subjects.

Adds:
- Interdaily Stability (IS)
- Intradaily Variability (IV)
- Relative Amplitude (RA)
- entropy / complexity features
- morning / afternoon / evening / night activity

Compares:
A) Existing 18 features
B) Enhanced features
C) Enhanced + Borderline-SMOTE

All preprocessing/resampling occurs inside each training fold.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from scipy.stats import entropy as scipy_entropy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED=42
DEP="data/depresjon"; PSY="data/psykose"
OUT="outputs/stage3_feature_enhancement"
os.makedirs(OUT,exist_ok=True)

BASE_FEATURES=[
"mean_activity","std_activity","median_activity","max_activity",
"q25_activity","q75_activity","iqr_activity","entropy","zero_crossings",
"sleep_hours","active_minutes","day_activity","night_activity",
"day_night_ratio","circ_regularity","prop_inactive","prop_active","n_days"
]

# ---------- circadian features ----------
def circadian_features(d):
    x=d["activity"].to_numpy(dtype=float)
    if "timestamp" not in d.columns:
        return dict(IS=np.nan,IV=np.nan,RA=np.nan,
                    morning_activity=np.nan,afternoon_activity=np.nan,
                    evening_activity=np.nan,night_activity=np.nan)

    t=pd.to_datetime(d["timestamp"],errors="coerce")
    tmp=pd.DataFrame({"activity":x,"timestamp":t}).dropna()
    if tmp.empty:
        return dict(IS=np.nan,IV=np.nan,RA=np.nan,
                    morning_activity=np.nan,afternoon_activity=np.nan,
                    evening_activity=np.nan,night_activity=np.nan)

    tmp["hour"]=tmp.timestamp.dt.hour
    hourly=tmp.groupby("hour")["activity"].mean()
    overall=tmp.activity.mean()

    # IS: variance of mean hourly profile / overall variance.
    hourly_complete=hourly.reindex(range(24))
    IS=hourly_complete.var(skipna=True)/(tmp.activity.var()+1e-12)

    # IV: mean squared successive differences / overall variance.
    IV=np.mean(np.diff(tmp.activity)**2)/(tmp.activity.var()+1e-12)

    # RA: relative amplitude using average activity in most active
    # 10-hour period versus least active 5-hour period.
    prof=hourly_complete.interpolate(limit_direction="both").to_numpy()
    if np.isfinite(prof).all():
        rolling10=np.array([prof[np.arange(i,i+10)%24].mean() for i in range(24)])
        rolling5=np.array([prof[np.arange(i,i+5)%24].mean() for i in range(24)])
        RA=(rolling10.max()-rolling5.min())/(rolling10.max()+rolling5.min()+1e-12)
    else:
        RA=np.nan

    def mean_hours(a,b):
        z=tmp[tmp.hour.between(a,b-1)].activity
        return z.mean() if len(z) else np.nan

    return dict(
        IS=IS, IV=IV, RA=RA,
        morning_activity=mean_hours(6,12),
        afternoon_activity=mean_hours(12,17),
        evening_activity=mean_hours(17,22),
        night_activity=(
            tmp[(tmp.hour>=22)|(tmp.hour<6)].activity.mean()
            if len(tmp[(tmp.hour>=22)|(tmp.hour<6)]) else np.nan
        )
    )


def extract(fp,label,label_str):
    d=pd.read_csv(fp)
    a=d["activity"].to_numpy(dtype=float)
    q25,q75=np.percentile(a,[25,75])
    hist,_=np.histogram(a,bins=20,density=True)

    if "timestamp" in d.columns:
        t=pd.to_datetime(d["timestamp"],errors="coerce")
        hour=t.dt.hour.fillna(0).astype(int)
        day=a[hour.between(6,21).to_numpy()] if len(a) else np.array([])
        night=a[~hour.between(6,21).to_numpy()] if len(a) else np.array([])
        dayv=np.mean(day) if len(day) else np.nan
        nightv=np.mean(night) if len(night) else np.nan
        ratio=(dayv+1e-5)/(nightv+1e-5)
        circ=d.assign(hour=hour).groupby("hour")["activity"].mean().std()
    else:
        dayv=nightv=ratio=circ=np.nan

    sid=os.path.splitext(os.path.basename(fp))[0]
    r={
        "subject_id":sid,"label":label,"label_str":label_str,
        "mean_activity":a.mean(),"std_activity":a.std(),"median_activity":np.median(a),
        "max_activity":a.max(),"q25_activity":q25,"q75_activity":q75,
        "iqr_activity":q75-q25,"entropy":scipy_entropy(hist+1e-10),
        "zero_crossings":np.sum(np.diff(np.sign(a-a.mean()))!=0),
        "sleep_hours":np.sum(a<5)/60,"active_minutes":np.sum(a>100),
        "day_activity":dayv,"night_activity":nightv,"day_night_ratio":ratio,
        "circ_regularity":circ,"prop_inactive":np.mean(a<5),
        "prop_active":np.mean(a>100),
        "n_days":len(d["date"].unique()) if "date" in d.columns else 1
    }
    r.update(circadian_features(d))
    return r


def load():
    dep=[]
    for f in sorted(glob.glob(os.path.join(DEP,"condition","*.csv"))):
        dep.append(extract(f,1,"depression"))
    for f in sorted(glob.glob(os.path.join(DEP,"control","*.csv"))):
        dep.append(extract(f,0,"healthy"))

    psy=[]
    for f in sorted(glob.glob(os.path.join(PSY,"patient","*.csv"))):
        psy.append(extract(f,2,"schizophrenia"))
    for f in sorted(glob.glob(os.path.join(PSY,"control","*.csv"))):
        psy.append(extract(f,0,"healthy"))
    return pd.DataFrame(dep),pd.DataFrame(psy)


def model(name):
    if name=="Random Forest":
        return RandomForestClassifier(n_estimators=300,max_depth=8,random_state=SEED,n_jobs=-1)
    if name=="CatBoost":
        return CatBoostClassifier(iterations=300,depth=6,learning_rate=.05,
            random_seed=SEED,verbose=False,allow_writing_files=False)
    return LGBMClassifier(n_estimators=300,learning_rate=.05,max_depth=6,
                          random_state=SEED,verbosity=-1,n_jobs=-1)


def evaluate(X,y,feature_set,imbalance):
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
    rows=[]
    for name in ["Random Forest","CatBoost","LightGBM"]:
        for fold,(tr,te) in enumerate(cv.split(X,y),1):
            if imbalance=="Borderline-SMOTE":
                pipe=ImbPipeline([
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler()),
                    ("sampler",BorderlineSMOTE(random_state=SEED,k_neighbors=3)),
                    ("model",model(name))
                ])
            else:
                pipe=Pipeline([
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler()),
                    ("model",model(name))
                ])
            pipe.fit(X[tr],y[tr])
            yp=pipe.predict(X[te]); prob=pipe.predict_proba(X[te])
            auc=roc_auc_score(label_binarize(y[te],classes=[0,1,2]),prob,
                              average="weighted",multi_class="ovr")
            rows.append({
                "feature_set":feature_set,"imbalance":imbalance,"model":name,"fold":fold,
                "accuracy":accuracy_score(y[te],yp),
                "precision":precision_score(y[te],yp,average="weighted",zero_division=0),
                "recall":recall_score(y[te],yp,average="weighted",zero_division=0),
                "f1":f1_score(y[te],yp,average="weighted",zero_division=0),
                "auc":auc
            })
    return pd.DataFrame(rows)


dep,psy=load()
data=pd.concat([dep,psy],ignore_index=True)
print(f"DEPRESJON: {len(dep)} | PSYKOSE: {len(psy)} | TOTAL: {len(data)}")
print(data.label_str.value_counts())

ENHANCED=BASE_FEATURES+[
"IS","IV","RA","morning_activity","afternoon_activity",
"evening_activity","night_activity"
]

Xbase=data[BASE_FEATURES].to_numpy()
Xenh=data[ENHANCED].to_numpy()
y=data.label.to_numpy()

print("\nNEW FEATURES:")
print(ENHANCED[len(BASE_FEATURES):])

results=[]
print("\n"+"="*70)
print("A) BASELINE FEATURES")
print("="*70)
results.append(evaluate(Xbase,y,"18 baseline features","None"))

print("\n"+"="*70)
print("B) ENHANCED FEATURES")
print("="*70)
results.append(evaluate(Xenh,y,"25 enhanced features","None"))

print("\n"+"="*70)
print("C) ENHANCED + BORDERLINE-SMOTE")
print("="*70)
results.append(evaluate(Xenh,y,"25 enhanced features","Borderline-SMOTE"))

df=pd.concat(results,ignore_index=True)
df.to_csv(os.path.join(OUT,"stage3_by_fold.csv"),index=False)

summary=df.groupby(["feature_set","imbalance","model"]).agg(
 accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),
 precision_mean=("precision","mean"),precision_std=("precision","std"),
 recall_mean=("recall","mean"),recall_std=("recall","std"),
 f1_mean=("f1","mean"),f1_std=("f1","std"),
 auc_mean=("auc","mean"),auc_std=("auc","std")
).reset_index()
summary.to_csv(os.path.join(OUT,"stage3_summary.csv"),index=False)

print("\n"+"="*100)
print("STAGE 3 SUMMARY — MEAN ± STD")
print("="*100)
print(summary.to_string(index=False))
print("\nPrimary metrics: recall and F1. Accuracy/AUC remain secondary.")
