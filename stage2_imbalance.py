
"""
STAGE 2 — CLASS IMBALANCE EXPERIMENTS
DEPRESJON + PSYKOSE

Compares:
1) Class weighting
2) Borderline-SMOTE
3) ADASYN

All imbalance handling happens INSIDE each training fold.
No resampling is ever applied to the outer test fold.

Uses the same 109 subjects and 18 features as trial.py / Stage 1.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
N_SPLITS = 5
DEPRESJON_DIR = "data/depresjon"
PSYKOSE_DIR = "data/psykose"
OUT = "outputs/stage2_imbalance"
os.makedirs(OUT, exist_ok=True)

FEATURE_COLS = [
    "mean_activity","std_activity","median_activity","max_activity",
    "q25_activity","q75_activity","iqr_activity","entropy",
    "zero_crossings","sleep_hours","active_minutes",
    "day_activity","night_activity","day_night_ratio",
    "circ_regularity","prop_inactive","prop_active","n_days"
]


def extract_features(df, subject_id, label_int, label_str):
    act = df["activity"].values.astype(float)
    mean_act, std_act = np.mean(act), np.std(act)
    median_act, max_act = np.median(act), np.max(act)
    q25, q75 = np.percentile(act,25), np.percentile(act,75)
    hist,_ = np.histogram(act,bins=20,density=True)
    ent = scipy_entropy(hist + 1e-10)
    zero = np.sum(np.diff(np.sign(act-mean_act)) != 0)
    sleep = np.sum(act < 5)/60.0
    active = np.sum(act > 100)
    pinactive, pact = np.mean(act < 5), np.mean(act > 100)

    if "timestamp" in df.columns:
        d = df.copy()
        d["hour"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.hour.fillna(0).astype(int)
        day = d[d["hour"].between(6,21)]["activity"].mean()
        night = d[~d["hour"].between(6,21)]["activity"].mean()
        ratio = (day+1e-5)/(night+1e-5)
        circ = d.groupby("hour")["activity"].mean().std()
    else:
        day=night=ratio=circ=np.nan

    days = len(df["date"].unique()) if "date" in df.columns else 1

    return {
        "subject_id":subject_id,"label":label_int,"label_str":label_str,
        "mean_activity":mean_act,"std_activity":std_act,"median_activity":median_act,
        "max_activity":max_act,"q25_activity":q25,"q75_activity":q75,
        "iqr_activity":q75-q25,"entropy":ent,"zero_crossings":zero,
        "sleep_hours":sleep,"active_minutes":active,"day_activity":day,
        "night_activity":night,"day_night_ratio":ratio,"circ_regularity":circ,
        "prop_inactive":pinactive,"prop_active":pact,"n_days":days
    }


def load_depresjon():
    rows=[]
    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR,"condition","*.csv"))):
        rows.append(extract_features(pd.read_csv(fp),os.path.splitext(os.path.basename(fp))[0],1,"depression"))
    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR,"control","*.csv"))):
        rows.append(extract_features(pd.read_csv(fp),os.path.splitext(os.path.basename(fp))[0],0,"healthy"))
    print(f"DEPRESJON: {len(rows)} subjects")
    return pd.DataFrame(rows)


def load_psykose():
    rows=[]
    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR,"patient","*.csv"))):
        rows.append(extract_features(pd.read_csv(fp),os.path.splitext(os.path.basename(fp))[0],2,"schizophrenia"))
    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR,"control","*.csv"))):
        rows.append(extract_features(pd.read_csv(fp),os.path.splitext(os.path.basename(fp))[0],0,"healthy"))
    print(f"PSYKOSE: {len(rows)} subjects")
    return pd.DataFrame(rows)


def models(method):
    # Class weighting is built into the classifier and only affects training.
    weighted = method == "Class Weight"

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,max_depth=8,random_state=SEED,n_jobs=-1,
            class_weight="balanced" if weighted else None
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,depth=6,learning_rate=0.05,
            random_seed=SEED,verbose=False,allow_writing_files=False,
            auto_class_weights="Balanced" if weighted else None
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,learning_rate=0.05,max_depth=6,
            random_state=SEED,verbosity=-1,n_jobs=-1,
            class_weight="balanced" if weighted else None
        )
    }


def metrics(model,Xte,yte):
    yp=model.predict(Xte)
    prob=model.predict_proba(Xte)
    try:
        auc=roc_auc_score(
            label_binarize(yte,classes=[0,1,2]),prob,
            average="weighted",multi_class="ovr"
        )
    except Exception:
        auc=np.nan
    return {
        "accuracy":accuracy_score(yte,yp),
        "precision":precision_score(yte,yp,average="weighted",zero_division=0),
        "recall":recall_score(yte,yp,average="weighted",zero_division=0),
        "f1":f1_score(yte,yp,average="weighted",zero_division=0),
        "auc":auc
    }


def run():
    dep,psy=load_depresjon(),load_psykose()
    data=pd.concat([dep,psy],ignore_index=True)
    X=data[FEATURE_COLS].values
    y=data["label"].values

    print(f"\nMerged: {len(data)} subjects")
    print(data["label_str"].value_counts())

    cv=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
    all_rows=[]

    for method in ["Class Weight","Borderline-SMOTE","ADASYN"]:
        print("\n"+"="*70)
        print(method)
        print("="*70)

        for name,base in models(method).items():
            for fold,(tr,te) in enumerate(cv.split(X,y),1):
                # Imputer + scaler are fit only on the training fold.
                if method == "Class Weight":
                    pipe=Pipeline([
                        ("imputer",SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler()),
                        ("model",clone(base))
                    ])
                else:
                    sampler = BorderlineSMOTE(random_state=SEED,k_neighbors=3) if method=="Borderline-SMOTE" else ADASYN(random_state=SEED,n_neighbors=3)
                    pipe=ImbPipeline([
                        ("imputer",SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler()),
                        ("sampler",sampler),
                        ("model",clone(base))
                    ])

                pipe.fit(X[tr],y[tr])
                m=metrics(pipe,X[te],y[te])

                all_rows.append({
                    "method":method,"model":name,"fold":fold,**m
                })
                print(
                    f"{name:<14} Fold {fold}: "
                    f"Acc={m['accuracy']*100:.1f}% | "
                    f"Prec={m['precision']:.3f} | "
                    f"Recall={m['recall']:.3f} | "
                    f"F1={m['f1']:.3f} | AUC={m['auc']:.3f}"
                )

    df=pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT,"by_fold.csv"),index=False)

    summary=df.groupby(["method","model"]).agg(
        accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),
        precision_mean=("precision","mean"),precision_std=("precision","std"),
        recall_mean=("recall","mean"),recall_std=("recall","std"),
        f1_mean=("f1","mean"),f1_std=("f1","std"),
        auc_mean=("auc","mean"),auc_std=("auc","std")
    ).reset_index()

    summary.to_csv(os.path.join(OUT,"summary.csv"),index=False)

    print("\n"+"="*70)
    print("STAGE 2 SUMMARY — MEAN ± STD")
    print("="*70)
    print(summary.to_string(index=False))

if __name__=="__main__":
    run()
