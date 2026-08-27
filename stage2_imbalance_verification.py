
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

SEED=42
DEPRESJON_DIR="data/depresjon"; PSYKOSE_DIR="data/psykose"
OUT="outputs/stage2_imbalance_verification"; os.makedirs(OUT,exist_ok=True)
FEATURE_COLS=["mean_activity","std_activity","median_activity","max_activity","q25_activity","q75_activity","iqr_activity","entropy","zero_crossings","sleep_hours","active_minutes","day_activity","night_activity","day_night_ratio","circ_regularity","prop_inactive","prop_active","n_days"]

def feat(df,sid,label,name):
    a=df.activity.values.astype(float); q25,q75=np.percentile(a,[25,75]); h,_=np.histogram(a,bins=20,density=True)
    if "timestamp" in df:
        d=df.copy(); d["hour"]=pd.to_datetime(d.timestamp,errors="coerce").dt.hour.fillna(0).astype(int)
        day=d[d.hour.between(6,21)].activity.mean(); night=d[~d.hour.between(6,21)].activity.mean()
        ratio=(day+1e-5)/(night+1e-5); circ=d.groupby("hour").activity.mean().std()
    else: day=night=ratio=circ=np.nan
    return {"subject_id":sid,"label":label,"label_str":name,"mean_activity":a.mean(),"std_activity":a.std(),"median_activity":np.median(a),"max_activity":a.max(),"q25_activity":q25,"q75_activity":q75,"iqr_activity":q75-q25,"entropy":scipy_entropy(h+1e-10),"zero_crossings":np.sum(np.diff(np.sign(a-a.mean()))!=0),"sleep_hours":np.sum(a<5)/60,"active_minutes":np.sum(a>100),"day_activity":day,"night_activity":night,"day_night_ratio":ratio,"circ_regularity":circ,"prop_inactive":np.mean(a<5),"prop_active":np.mean(a>100),"n_days":len(df.date.unique()) if "date" in df else 1}

def load():
    r=[]
    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR,"condition","*.csv"))): r.append(feat(pd.read_csv(fp),os.path.splitext(os.path.basename(fp))[0],1,"depression"))
    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR,"control","*.csv"))): r.append(feat(pd.read_csv(fp),os.path.splitext(os.path.basename(fp))[0],0,"healthy"))
    dep=pd.DataFrame(r); r=[]
    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR,"patient","*.csv"))): r.append(feat(pd.read_csv(fp),os.path.splitext(os.path.basename(fp))[0],2,"schizophrenia"))
    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR,"control","*.csv"))): r.append(feat(pd.read_csv(fp),os.path.splitext(os.path.basename(fp))[0],0,"healthy"))
    psy=pd.DataFrame(r)
    return dep,psy,pd.concat([dep,psy],ignore_index=True)

def make_models(weighted=False):
    return {
      "Random Forest":RandomForestClassifier(n_estimators=300,max_depth=8,random_state=SEED,n_jobs=-1,class_weight="balanced" if weighted else None),
      "CatBoost":CatBoostClassifier(iterations=300,depth=6,learning_rate=.05,random_seed=SEED,verbose=False,allow_writing_files=False,auto_class_weights="Balanced" if weighted else None),
      "LightGBM":LGBMClassifier(n_estimators=300,learning_rate=.05,max_depth=6,random_state=SEED,verbosity=-1,n_jobs=-1,class_weight="balanced" if weighted else None)
    }

def run_method(X,y,method):
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED); rows=[]
    for name,base in make_models(method=="Class Weight").items():
        for fold,(tr,te) in enumerate(cv.split(X,y),1):
            print(f"\n{method} | {name} | Fold {fold}")
            print("  TRAIN BEFORE:", dict(zip(*np.unique(y[tr],return_counts=True))))
            if method=="Class Weight":
                pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",clone(base))])
                after_note="No synthetic resampling (class weights only)"
            else:
                sampler=BorderlineSMOTE(random_state=SEED,k_neighbors=3) if method=="Borderline-SMOTE" else ADASYN(random_state=SEED,n_neighbors=3)
                pipe=ImbPipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("sampler",sampler),("model",clone(base))])
                # Explicitly inspect what the sampler creates, using the same train-only preprocessing.
                imp=SimpleImputer(strategy="median"); sc=StandardScaler()
                Xt=sc.fit_transform(imp.fit_transform(X[tr]))
                _,yr=sampler.fit_resample(Xt,y[tr])
                print("  TRAIN AFTER :", dict(zip(*np.unique(yr,return_counts=True))))
                after_note=str(dict(zip(*np.unique(yr,return_counts=True))))
            pipe.fit(X[tr],y[tr]); yp=pipe.predict(X[te]); prob=pipe.predict_proba(X[te])
            auc=roc_auc_score(label_binarize(y[te],classes=[0,1,2]),prob,average="weighted",multi_class="ovr")
            rows.append({"method":method,"model":name,"fold":fold,"accuracy":accuracy_score(y[te],yp),"precision":precision_score(y[te],yp,average="weighted",zero_division=0),"recall":recall_score(y[te],yp,average="weighted",zero_division=0),"f1":f1_score(y[te],yp,average="weighted",zero_division=0),"auc":auc})
    return pd.DataFrame(rows)

dep,psy,data=load()
print(f"DEPRESJON: {len(dep)} | PSYKOSE: {len(psy)} | TOTAL: {len(data)}")
print("\nPER-DATASET CLASS DISTRIBUTION")
print("DEPRESJON:",dep.label_str.value_counts().to_dict())
print("PSYKOSE:",psy.label_str.value_counts().to_dict())
X=data[FEATURE_COLS].values; y=data.label.values

frames=[]
for method in ["Class Weight","Borderline-SMOTE","ADASYN"]:
    frames.append(run_method(X,y,method))
df=pd.concat(frames,ignore_index=True)
df.to_csv(os.path.join(OUT,"by_fold.csv"),index=False)

summary=df.groupby(["method","model"]).agg(
 accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),
 precision_mean=("precision","mean"),precision_std=("precision","std"),
 recall_mean=("recall","mean"),recall_std=("recall","std"),
 f1_mean=("f1","mean"),f1_std=("f1","std"),auc_mean=("auc","mean"),auc_std=("auc","std")
).reset_index()
summary.to_csv(os.path.join(OUT,"comparison_summary.csv"),index=False)

print("\n"+"="*100)
print("FINAL COMPARISON — MEAN ± STD")
print("="*100)
print(summary.to_string(index=False))
print("\nPrimary screening metrics: RECALL and F1; accuracy/AUC are reported as secondary metrics.")
