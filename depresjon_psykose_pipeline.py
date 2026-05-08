"""
Mental Health Monitoring — Combined Pipeline
Datasets: DEPRESJON + PSYKOSE + Your collected survey data
Tasks:
  1. Binary:    healthy vs condition (depression or schizophrenia)
  2. 3-class:   healthy / depression / schizophrenia
  3. Survey:    PHQ-4 severity from your collected data (as second component)
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")

from scipy.stats import entropy as scipy_entropy
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, accuracy_score)
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

os.makedirs("outputs/results", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/shap",   exist_ok=True)
os.makedirs("outputs/eda",    exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG — update these paths to match your PC
# ─────────────────────────────────────────────
DEPRESJON_DIR = "data/depresjon"   # folder containing condition/ control/ scores.csv
PSYKOSE_DIR   = "data/psykose"     # folder containing patient/ control/ patients_info.csv
SURVEY_CSV    = "data/data_collection.csv"

# ─────────────────────────────────────────────
# FEATURE EXTRACTION from minute-level activity
# ─────────────────────────────────────────────
def extract_features(df, subject_id, label_int, label_str):
    """Extract statistical + circadian features from raw actigraphy CSV."""
    act = df["activity"].values.astype(float)

    # Basic stats
    mean_act   = np.mean(act)
    std_act    = np.std(act)
    median_act = np.median(act)
    max_act    = np.max(act)
    min_act    = np.min(act)
    q25        = np.percentile(act, 25)
    q75        = np.percentile(act, 75)
    iqr        = q75 - q25

    # Entropy (irregularity of movement)
    hist, _    = np.histogram(act, bins=20, density=True)
    hist       = hist + 1e-10
    act_entropy= scipy_entropy(hist)

    # Zero crossings (restlessness)
    zero_cross = np.sum(np.diff(np.sign(act - mean_act)) != 0)

    # Sleep estimate: activity < 5 = likely sleep (per-minute)
    sleep_mins = np.sum(act < 5)
    sleep_hrs  = sleep_mins / 60.0

    # Active periods: activity > 100
    active_mins = np.sum(act > 100)

    # Circadian: split into day (6:00–22:00) and night (22:00–6:00)
    if "timestamp" in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        day_act   = df[df["hour"].between(6, 21)]["activity"].mean()
        night_act = df[~df["hour"].between(6, 21)]["activity"].mean()
        day_night_ratio = (day_act + 1e-5) / (night_act + 1e-5)

        # Circadian regularity: std of hourly mean activity
        hourly_mean = df.groupby("hour")["activity"].mean()
        circ_regularity = hourly_mean.std()
    else:
        day_act = night_act = day_night_ratio = circ_regularity = np.nan

    # Prop of inactive minutes
    prop_inactive = np.mean(act < 5)
    prop_active   = np.mean(act > 100)

    # Total days
    n_days = len(df["date"].unique()) if "date" in df.columns else np.nan

    return {
        "subject_id":       subject_id,
        "label":            label_int,
        "label_str":        label_str,
        "mean_activity":    mean_act,
        "std_activity":     std_act,
        "median_activity":  median_act,
        "max_activity":     max_act,
        "min_activity":     min_act,
        "q25_activity":     q25,
        "q75_activity":     q75,
        "iqr_activity":     iqr,
        "entropy":          act_entropy,
        "zero_crossings":   zero_cross,
        "sleep_hours":      sleep_hrs,
        "active_minutes":   active_mins,
        "day_activity":     day_act,
        "night_activity":   night_act,
        "day_night_ratio":  day_night_ratio,
        "circ_regularity":  circ_regularity,
        "prop_inactive":    prop_inactive,
        "prop_active":      prop_active,
        "n_days":           n_days,
    }

# ─────────────────────────────────────────────
# LOAD DEPRESJON
# ─────────────────────────────────────────────
def load_depresjon():
    print("\n── Loading DEPRESJON ──")
    rows = []

    for fpath in sorted(glob.glob(os.path.join(DEPRESJON_DIR, "condition", "*.csv"))):
        sid = os.path.splitext(os.path.basename(fpath))[0]
        df  = pd.read_csv(fpath)
        rows.append(extract_features(df, sid, label_int=1, label_str="depression"))

    for fpath in sorted(glob.glob(os.path.join(DEPRESJON_DIR, "control", "*.csv"))):
        sid = os.path.splitext(os.path.basename(fpath))[0]
        df  = pd.read_csv(fpath)
        rows.append(extract_features(df, sid, label_int=0, label_str="healthy"))

    feat_df = pd.DataFrame(rows)

    # Merge MADRS scores
    scores = pd.read_csv(os.path.join(DEPRESJON_DIR, "scores.csv"))
    scores = scores.rename(columns={"number": "subject_id"})
    feat_df = feat_df.merge(scores[["subject_id","age","gender","madrs1","madrs2"]],
                            on="subject_id", how="left")
    feat_df["source"] = "depresjon"
    print(f"  DEPRESJON: {len(feat_df)} subjects  "
          f"| depression={sum(feat_df.label==1)}  healthy={sum(feat_df.label==0)}")
    return feat_df

# ─────────────────────────────────────────────
# LOAD PSYKOSE
# ─────────────────────────────────────────────
def load_psykose():
    print("\n── Loading PSYKOSE ──")
    rows = []

    # patient folder → schizophrenia (label=2)
    pat_folder = os.path.join(PSYKOSE_DIR, "patient")
    for fpath in sorted(glob.glob(os.path.join(pat_folder, "*.csv"))):
        sid = os.path.splitext(os.path.basename(fpath))[0]
        df  = pd.read_csv(fpath)
        rows.append(extract_features(df, sid, label_int=2, label_str="schizophrenia"))

    # control folder → healthy (label=0)
    ctrl_folder = os.path.join(PSYKOSE_DIR, "control")
    for fpath in sorted(glob.glob(os.path.join(ctrl_folder, "*.csv"))):
        sid = os.path.splitext(os.path.basename(fpath))[0]
        df  = pd.read_csv(fpath)
        rows.append(extract_features(df, sid, label_int=0, label_str="healthy"))

    feat_df = pd.DataFrame(rows)

    # Merge patients_info
    info_path = os.path.join(PSYKOSE_DIR, "patients_info.csv")
    if os.path.exists(info_path):
        info = pd.read_csv(info_path)
        print(f"  patients_info columns: {info.columns.tolist()}")
        # rename first column to subject_id for merge
        info = info.rename(columns={info.columns[0]: "subject_id"})
        feat_df = feat_df.merge(info, on="subject_id", how="left")

    feat_df["source"] = "psykose"
    print(f"  PSYKOSE: {len(feat_df)} subjects  "
          f"| schizophrenia={sum(feat_df.label==2)}  healthy={sum(feat_df.label==0)}")
    return feat_df

# ─────────────────────────────────────────────
# MERGE DATASETS
# ─────────────────────────────────────────────
def merge_datasets(dep_df, psy_df):
    print("\n── Merging datasets ──")

    # PSYKOSE control subjects overlap with DEPRESJON control — deduplicate
    # Keep DEPRESJON controls, drop duplicate control IDs from PSYKOSE
    dep_ctrl_ids = set(dep_df[dep_df.label==0]["subject_id"])
    psy_no_ctrl  = psy_df[psy_df.label != 0]  # keep only schizophrenia from PSYKOSE
    psy_ctrl     = psy_df[psy_df.label == 0]   # PSYKOSE controls (same people)

    # Use PSYKOSE controls as additional healthy samples (different recording sessions)
    merged = pd.concat([dep_df, psy_no_ctrl, psy_ctrl], ignore_index=True)

    # Drop duplicated control subjects if same ID appears in both
    merged = merged.drop_duplicates(subset=["subject_id"], keep="first")

    print(f"  Merged total: {len(merged)} subjects")
    print(f"  healthy={sum(merged.label==0)} | "
          f"depression={sum(merged.label==1)} | "
          f"schizophrenia={sum(merged.label==2)}")
    return merged

# ─────────────────────────────────────────────
# MODEL TRAINING + EVALUATION
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "mean_activity","std_activity","median_activity","max_activity",
    "q25_activity","q75_activity","iqr_activity","entropy",
    "zero_crossings","sleep_hours","active_minutes",
    "day_activity","night_activity","day_night_ratio",
    "circ_regularity","prop_inactive","prop_active","n_days"
]

def evaluate(model, X_te, y_te, name, task, n_classes):
    yp   = model.predict(X_te)
    yprb = model.predict_proba(X_te)
    acc  = accuracy_score(y_te, yp)
    f1   = f1_score(y_te, yp, average="weighted")
    if n_classes == 2:
        auc = roc_auc_score(y_te, yprb[:,1])
    else:
        yte_b = label_binarize(y_te, classes=sorted(np.unique(y_te)))
        auc   = roc_auc_score(yte_b, yprb, average="weighted", multi_class="ovr")

    print(f"  [{name}]  Acc={acc*100:.1f}%  F1={f1:.3f}  AUC={auc:.3f}")
    print(classification_report(y_te, yp, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_te, yp)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(f"{name} — {task}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/results/cm_{task}_{name.replace(' ','_')}.png", dpi=150)
    plt.close()

    return {"Task":task,"Model":name,
            "Accuracy":round(acc*100,2),"F1":round(f1,4),"AUC":round(auc,4)}

def run_models(X_tr, X_te, y_tr, y_te, task):
    n_classes = len(np.unique(y_tr))
    obj = "multiclass" if n_classes > 2 else "binary"
    print(f"\n{'='*52}\n  TASK: {task.upper()}  ({n_classes} classes)\n{'='*52}")

    # SMOTE
    Xr, yr = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_tr))-1)
                   ).fit_resample(X_tr, y_tr)

    lgbm = LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                           num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=1.0,
                           objective=obj, n_jobs=-1, random_state=42, verbose=-1)
    cb   = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                               random_seed=42, verbose=0)
    rf   = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)

    for m in [lgbm, cb, rf]:
        m.fit(Xr, yr)

    ensemble = VotingClassifier(
        estimators=[("lgbm",lgbm),("cb",cb),("rf",rf)], voting="soft"
    )
    ensemble.fit(Xr, yr)

    results = []
    best_f1, best_model = 0, None
    for name, model in [("LightGBM",lgbm),("CatBoost",cb),
                         ("Random Forest",rf),("Ensemble",ensemble)]:
        r = evaluate(model, X_te, y_te, name, task, n_classes)
        results.append(r)
        if r["F1"] > best_f1:
            best_f1, best_model = r["F1"], model

    joblib.dump(best_model, f"outputs/models/best_{task}.pkl")
    return results, best_model, lgbm   # return lgbm for SHAP

# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────
def run_shap(model, X_te, feature_names, task):
    print(f"\n  Running SHAP for {task}...")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te)
        if isinstance(shap_values, list):
            sv = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            sv = np.abs(shap_values)
        mean_imp = sv.mean(axis=0)
        idx = np.argsort(mean_imp)[::-1]

        plt.figure(figsize=(8,5))
        plt.barh([feature_names[i] for i in idx[::-1]],
                  mean_imp[idx[::-1]], color="#5C6BC0", edgecolor="white")
        plt.title(f"SHAP Feature Importance — {task}", fontweight="bold")
        plt.xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        plt.savefig(f"outputs/shap/shap_{task}.png", dpi=150)
        plt.close()
        print(f"  SHAP saved for {task}")
    except Exception as e:
        print(f"  SHAP skipped ({e})")

# ─────────────────────────────────────────────
# EDA PLOTS
# ─────────────────────────────────────────────
def plot_eda(merged):
    print("\n── EDA plots ──")
    colors = {"healthy":"#66BB6A","depression":"#42A5F5","schizophrenia":"#AB47BC"}

    # 1. Mean activity by class
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    for feat, ax in zip(["mean_activity","sleep_hours","entropy"], axes):
        for lstr, grp in merged.groupby("label_str"):
            ax.hist(grp[feat].dropna(), bins=15, alpha=0.6,
                    label=lstr, color=colors.get(lstr,"gray"), edgecolor="white")
        ax.set_title(feat.replace("_"," ").title(), fontweight="bold")
        ax.legend()
    plt.suptitle("Feature Distributions by Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/eda/feature_distributions.png", dpi=150)
    plt.close()

    # 2. Class balance
    fig, ax = plt.subplots(figsize=(6,4))
    counts = merged["label_str"].value_counts()
    ax.bar(counts.index, counts.values,
           color=[colors.get(k,"gray") for k in counts.index], edgecolor="white")
    ax.set_title("Class Distribution — Merged Dataset", fontweight="bold")
    ax.set_ylabel("Number of subjects")
    for i,(k,v) in enumerate(counts.items()):
        ax.text(i, v+0.3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/eda/class_distribution.png", dpi=150)
    plt.close()
    print("  EDA plots saved.")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load & merge
    dep_df  = load_depresjon()
    psy_df  = load_psykose()
    merged  = merge_datasets(dep_df, psy_df)

    merged.to_csv("data/merged_features.csv", index=False)
    print("\n  Saved: data/merged_features.csv")

    # 2. EDA
    plot_eda(merged)

    # 3. Prepare feature matrix
    X = merged[FEATURE_COLS].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    joblib.dump(scaler, "outputs/models/scaler_actigraphy.pkl")

    all_results = []

    # ── TASK A: Binary (healthy=0 vs condition=1) ──
    y_bin = (merged["label"] > 0).astype(int)
    Xtr,Xte,ytr,yte = train_test_split(Xs, y_bin, test_size=0.2,
                                        random_state=42, stratify=y_bin)
    res, best, lgbm_bin = run_models(Xtr, Xte, ytr, yte, "binary_healthy_vs_condition")
    all_results.extend(res)
    run_shap(lgbm_bin, Xte, FEATURE_COLS, "binary")

    # ── TASK B: 3-class (healthy / depression / schizophrenia) ──
    y_multi = merged["label"].values
    Xtr,Xte,ytr,yte = train_test_split(Xs, y_multi, test_size=0.2,
                                        random_state=42, stratify=y_multi)
    res, best, lgbm_3 = run_models(Xtr, Xte, ytr, yte, "3class_health_dep_schiz")
    all_results.extend(res)
    run_shap(lgbm_3, Xte, FEATURE_COLS, "3class")

    # ── TASK C: Depression only — DEPRESJON subset ──
    dep_only = dep_df.copy()
    dep_only["label_bin"] = dep_only["label"]
    X_d  = dep_only[FEATURE_COLS].fillna(dep_only[FEATURE_COLS].median())
    Xs_d = StandardScaler().fit_transform(X_d)
    y_d  = dep_only["label_bin"].values
    Xtr,Xte,ytr,yte = train_test_split(Xs_d, y_d, test_size=0.2,
                                        random_state=42, stratify=y_d)
    res, _, _ = run_models(Xtr, Xte, ytr, yte, "depression_binary_DEPRESJON")
    all_results.extend(res)

    # ── SUMMARY ──
    rdf = pd.DataFrame(all_results)
    print("\n\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(rdf.to_string(index=False))
    rdf.to_csv("outputs/results/actigraphy_results.csv", index=False)

    # Bar chart
    tasks = rdf["Task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(6*len(tasks), 5))
    if len(tasks) == 1: axes = [axes]
    clrs = ["#5C6BC0","#42A5F5","#66BB6A","#AB47BC"]
    for i, task in enumerate(tasks):
        sub = rdf[rdf["Task"]==task].sort_values("Accuracy", ascending=False)
        axes[i].bar(range(len(sub)), sub["Accuracy"].values,
                    color=clrs[:len(sub)], edgecolor="white")
        axes[i].set_xticks(range(len(sub)))
        axes[i].set_xticklabels(sub["Model"].values, rotation=30, ha="right")
        axes[i].set_title(task, fontweight="bold", fontsize=9)
        axes[i].set_ylim(40, 100)
        axes[i].set_ylabel("Accuracy (%)")
        for j,v in enumerate(sub["Accuracy"].values):
            axes[i].text(j, v+0.4, f"{v:.1f}", ha="center", fontsize=8)
    plt.suptitle("Actigraphy Model Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/results/actigraphy_accuracy.png", dpi=150)
    plt.show()
    print("\nAll done. Check outputs/ folder.")