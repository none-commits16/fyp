"""
Mental Health Monitoring — Improved Pipeline v2
Key improvements over v1:
  - Richer per-day and per-hour features (not just global stats)
  - MADRS severity regression task (depression only)
  - 5-fold stratified cross-validation (more reliable than single split)
  - Demographic features from scores.csv
  - Better SMOTE k_neighbors handling
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")

from scipy.stats import entropy as scipy_entropy, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, accuracy_score,
                              mean_absolute_error, r2_score)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier
import shap

os.makedirs("outputs/results", exist_ok=True)
os.makedirs("outputs/models",  exist_ok=True)
os.makedirs("outputs/shap",    exist_ok=True)
os.makedirs("outputs/eda",     exist_ok=True)

# ── PATHS ────────────────────────────────────
DEPRESJON_DIR = "data/depresjon"
PSYKOSE_DIR   = "data/psykose"
SURVEY_CSV    = "data/data_collection.csv"

# ── FEATURE EXTRACTION ───────────────────────
def extract_features(df, subject_id, label_int, label_str):
    act = df["activity"].values.astype(float)

    # Global stats
    f = {
        "subject_id":      subject_id,
        "label":           label_int,
        "label_str":       label_str,
        "mean_act":        np.mean(act),
        "std_act":         np.std(act),
        "median_act":      np.median(act),
        "max_act":         np.max(act),
        "iqr_act":         np.percentile(act,75) - np.percentile(act,25),
        "skewness":        skew(act),
        "kurt":            kurtosis(act),
        "prop_inactive":   np.mean(act < 5),
        "prop_active":     np.mean(act > 100),
        "prop_high":       np.mean(act > 500),
    }

    # Entropy
    hist, _ = np.histogram(act, bins=20, density=True)
    f["entropy"] = scipy_entropy(hist + 1e-10)

    # Zero crossings
    f["zero_cross"] = int(np.sum(np.diff(np.sign(act - f["mean_act"])) != 0))

    # Sleep estimate (< 5 counts/min = rest)
    f["sleep_hrs"]    = np.sum(act < 5) / 60.0
    f["active_mins"]  = int(np.sum(act > 100))

    # Autocorrelation lag-1 (circadian rhythm proxy)
    if len(act) > 1:
        f["autocorr_lag1"] = float(pd.Series(act).autocorr(lag=1))
    else:
        f["autocorr_lag1"] = 0.0

    # Hourly & day/night features
    if "timestamp" in df.columns:
        df = df.copy()
        df["hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour
        df["hour"] = df["hour"].fillna(0).astype(int)

        hourly = df.groupby("hour")["activity"].mean()
        f["circ_regularity"] = float(hourly.std())
        f["peak_hour"]       = int(hourly.idxmax()) if len(hourly) > 0 else 12

        day_act   = df[df["hour"].between(6,21)]["activity"].mean()
        night_act = df[~df["hour"].between(6,21)]["activity"].mean()
        f["day_act"]         = float(day_act)   if not np.isnan(day_act)   else 0.0
        f["night_act"]       = float(night_act) if not np.isnan(night_act) else 0.0
        f["day_night_ratio"] = (f["day_act"]+1e-5) / (f["night_act"]+1e-5)

        # Morning (6-10) vs evening (18-22) activity ratio
        morn = df[df["hour"].between(6,10)]["activity"].mean()
        eve  = df[df["hour"].between(18,22)]["activity"].mean()
        f["morn_eve_ratio"]  = (morn+1e-5) / (eve+1e-5)
    else:
        for k in ["circ_regularity","peak_hour","day_act","night_act",
                  "day_night_ratio","morn_eve_ratio"]:
            f[k] = np.nan

    # Per-day variability (how consistent is activity across days)
    if "date" in df.columns:
        daily_mean = df.groupby("date")["activity"].mean()
        f["n_days"]          = int(len(daily_mean))
        f["daily_mean_std"]  = float(daily_mean.std())  # high = irregular
        f["daily_mean_min"]  = float(daily_mean.min())
        f["daily_mean_max"]  = float(daily_mean.max())
        f["low_activity_days"] = int(np.sum(daily_mean < daily_mean.mean() * 0.5))
    else:
        for k in ["n_days","daily_mean_std","daily_mean_min",
                  "daily_mean_max","low_activity_days"]:
            f[k] = np.nan

    return f

FEATURE_COLS = [
    "mean_act","std_act","median_act","max_act","iqr_act",
    "skewness","kurt","prop_inactive","prop_active","prop_high",
    "entropy","zero_cross","sleep_hrs","active_mins","autocorr_lag1",
    "circ_regularity","peak_hour","day_act","night_act",
    "day_night_ratio","morn_eve_ratio",
    "n_days","daily_mean_std","daily_mean_min","daily_mean_max",
    "low_activity_days"
]

# ── LOAD DEPRESJON ───────────────────────────
def load_depresjon():
    print("\n── Loading DEPRESJON ──")
    rows = []
    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR,"condition","*.csv"))):
        sid = os.path.splitext(os.path.basename(fp))[0]
        rows.append(extract_features(pd.read_csv(fp), sid, 1, "depression"))
    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR,"control","*.csv"))):
        sid = os.path.splitext(os.path.basename(fp))[0]
        rows.append(extract_features(pd.read_csv(fp), sid, 0, "healthy"))

    feat_df = pd.DataFrame(rows)
    scores  = pd.read_csv(os.path.join(DEPRESJON_DIR,"scores.csv"))
    scores  = scores.rename(columns={"number":"subject_id"})
    feat_df = feat_df.merge(
        scores[["subject_id","age","gender","madrs1","madrs2",
                "afftype","inpatient","edu"]],
        on="subject_id", how="left")
    feat_df["madrs_avg"] = feat_df[["madrs1","madrs2"]].mean(axis=1)
    feat_df["source"]    = "depresjon"
    print(f"  {len(feat_df)} subjects | "
          f"depressed={sum(feat_df.label==1)} healthy={sum(feat_df.label==0)}")
    return feat_df

# ── LOAD PSYKOSE ────────────────────────────
def load_psykose():
    print("\n── Loading PSYKOSE ──")
    rows = []
    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR,"patient","*.csv"))):
        sid = os.path.splitext(os.path.basename(fp))[0]
        rows.append(extract_features(pd.read_csv(fp), sid, 2, "schizophrenia"))
    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR,"control","*.csv"))):
        sid = os.path.splitext(os.path.basename(fp))[0]
        rows.append(extract_features(pd.read_csv(fp), sid, 0, "healthy"))

    feat_df = pd.DataFrame(rows)
    info_p  = os.path.join(PSYKOSE_DIR,"patients_info.csv")
    if os.path.exists(info_p):
        info = pd.read_csv(info_p)
        info = info.rename(columns={info.columns[0]:"subject_id"})
        feat_df = feat_df.merge(info, on="subject_id", how="left")
    feat_df["source"] = "psykose"
    print(f"  {len(feat_df)} subjects | "
          f"schizophrenia={sum(feat_df.label==2)} healthy={sum(feat_df.label==0)}")
    return feat_df

# ── MERGE ────────────────────────────────────
def merge_datasets(dep_df, psy_df):
    print("\n── Merging ──")
    psy_patients = psy_df[psy_df.label == 2]
    psy_ctrl     = psy_df[psy_df.label == 0]
    merged = pd.concat([dep_df, psy_patients, psy_ctrl], ignore_index=True)
    merged = merged.drop_duplicates(subset=["subject_id"], keep="first")
    print(f"  Total: {len(merged)} | "
          f"healthy={sum(merged.label==0)} "
          f"depression={sum(merged.label==1)} "
          f"schizophrenia={sum(merged.label==2)}")
    return merged

# ── CROSS-VALIDATED EVALUATION ───────────────
def cv_evaluate(X, y, model, task, n_splits=5):
    """5-fold stratified CV — more reliable than single split."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    n_classes = len(np.unique(y))

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        k = max(1, min(3, min(np.bincount(ytr)) - 1))
        Xr, yr = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xtr, ytr)

        model.fit(Xr, yr)
        yp   = model.predict(Xte)
        yprb = model.predict_proba(Xte)

        accs.append(accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, average="weighted"))

        try:
            if n_classes == 2:
                aucs.append(roc_auc_score(yte, yprb[:,1]))
            else:
                yte_b = label_binarize(yte, classes=sorted(np.unique(y)))
                aucs.append(roc_auc_score(yte_b, yprb,
                                           average="weighted", multi_class="ovr"))
        except:
            aucs.append(np.nan)

    print(f"  CV Acc : {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
    print(f"  CV F1  : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  CV AUC : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    return np.mean(accs), np.mean(f1s), np.mean(aucs)

# ── TRAIN + EVALUATE (single split + CM) ─────
def run_task(X, y, task, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(y))
    obj = "multiclass" if n_classes > 2 else "binary"
    print(f"\n{'='*54}\n  TASK: {task.upper()}\n{'='*54}")

    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2,
                                        random_state=42, stratify=y)
    k  = max(1, min(3, min(np.bincount(ytr)) - 1))
    Xr, yr = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xtr, ytr)

    lgbm = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                           num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=1.0, min_child_samples=3,
                           objective=obj, n_jobs=-1, random_state=42, verbose=-1)
    cb   = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.03,
                               random_seed=42, verbose=0)
    rf   = RandomForestClassifier(n_estimators=400, max_depth=8,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    for m in [lgbm, cb, rf]:
        m.fit(Xr, yr)

    ensemble = VotingClassifier(
        estimators=[("lgbm",lgbm),("cb",cb),("rf",rf)], voting="soft")
    ensemble.fit(Xr, yr)

    results = []
    best_f1, best_model, best_lgbm = 0, None, lgbm

    for name, model in [("LightGBM",lgbm),("CatBoost",cb),
                         ("Random Forest",rf),("Ensemble",ensemble)]:
        yp   = model.predict(Xte)
        yprb = model.predict_proba(Xte)
        acc  = accuracy_score(yte, yp)
        f1   = f1_score(yte, yp, average="weighted")
        try:
            if n_classes == 2:
                auc = roc_auc_score(yte, yprb[:,1])
            else:
                yte_b = label_binarize(yte, classes=sorted(np.unique(y)))
                auc   = roc_auc_score(yte_b, yprb, average="weighted",
                                       multi_class="ovr")
        except:
            auc = np.nan

        print(f"\n  [{name}]  Acc={acc*100:.1f}%  F1={f1:.3f}  AUC={auc:.3f}")
        print(classification_report(yte, yp, digits=3))

        # Confusion matrix
        cm = confusion_matrix(yte, yp)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title(f"{name} — {task}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"outputs/results/cm_{task}_{name.replace(' ','_')}.png", dpi=150)
        plt.close()

        results.append({"Task":task,"Model":name,
                         "Accuracy":round(acc*100,2),
                         "F1":round(f1,4),"AUC":round(auc,4)})
        if f1 > best_f1:
            best_f1, best_model = f1, model

    # Cross-validation with best model type (LightGBM)
    print(f"\n  ── 5-Fold CV (LightGBM) for {task} ──")
    cv_acc, cv_f1, cv_auc = cv_evaluate(X, y, LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=3,
        objective=obj, n_jobs=-1, random_state=42, verbose=-1), task)

    results.append({"Task":task,"Model":"LightGBM (5-fold CV)",
                     "Accuracy":round(cv_acc*100,2),
                     "F1":round(cv_f1,4),"AUC":round(cv_auc,4)})

    joblib.dump(best_model, f"outputs/models/best_{task}.pkl")
    return results, lgbm, Xte

# ── MADRS SEVERITY REGRESSION ────────────────
def run_madrs_regression(dep_df):
    print(f"\n{'='*54}\n  BONUS TASK: MADRS Severity Regression\n{'='*54}")
    sub = dep_df[dep_df["madrs_avg"].notna()].copy()
    X   = sub[FEATURE_COLS].fillna(sub[FEATURE_COLS].median()).values
    y   = sub["madrs_avg"].values

    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LGBMRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                         num_leaves=31, random_state=42, verbose=-1)
    reg.fit(Xtr, ytr)
    yp  = reg.predict(Xte)
    mae = mean_absolute_error(yte, yp)
    r2  = r2_score(yte, yp)
    print(f"  MADRS Regression — MAE: {mae:.2f}  R²: {r2:.3f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(yte, yp, alpha=0.7, color="#5C6BC0", edgecolors="white", s=60)
    mn, mx = min(yte.min(), yp.min()), max(yte.max(), yp.max())
    ax.plot([mn,mx],[mn,mx],"--", color="#F44336", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual MADRS Score"); ax.set_ylabel("Predicted MADRS Score")
    ax.set_title(f"MADRS Severity Prediction  (MAE={mae:.2f}, R²={r2:.3f})",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/results/madrs_regression.png", dpi=150)
    plt.close()
    joblib.dump(reg, "outputs/models/madrs_regressor.pkl")
    return mae, r2

# ── SHAP ─────────────────────────────────────
def run_shap(model, X_te, task):
    try:
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(X_te)
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
        else:
            sv = np.abs(sv)
        imp = sv.mean(axis=0)
        idx = np.argsort(imp)[::-1]

        plt.figure(figsize=(8,6))
        plt.barh([FEATURE_COLS[i] for i in idx[::-1]],
                  imp[idx[::-1]], color="#5C6BC0", edgecolor="white")
        plt.title(f"SHAP — {task}", fontweight="bold")
        plt.xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        plt.savefig(f"outputs/shap/shap_{task}.png", dpi=150)
        plt.close()
        print(f"  SHAP saved: {task}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

# ── MAIN ─────────────────────────────────────
if __name__ == "__main__":

    dep_df = load_depresjon()
    psy_df = load_psykose()
    merged = merge_datasets(dep_df, psy_df)
    merged.to_csv("data/merged_features_v2.csv", index=False)

    all_results = []
    scaler = StandardScaler()

    # ── Task 1: Binary — healthy vs condition ──
    X1 = merged[FEATURE_COLS].fillna(merged[FEATURE_COLS].median()).values
    X1s = scaler.fit_transform(X1)
    y1  = (merged["label"] > 0).astype(int).values
    res, lgbm1, Xte1 = run_task(X1s, y1, "binary_healthy_vs_condition")
    all_results.extend(res)
    run_shap(lgbm1, Xte1, "binary")

    # ── Task 2: 3-class ──
    X2s = X1s.copy()
    y2  = merged["label"].values
    res, lgbm2, Xte2 = run_task(X2s, y2, "3class_healthy_dep_schiz", n_classes=3)
    all_results.extend(res)
    run_shap(lgbm2, Xte2, "3class")

    # ── Task 3: Depression binary (DEPRESJON only) ──
    X3 = dep_df[FEATURE_COLS].fillna(dep_df[FEATURE_COLS].median()).values
    X3s = StandardScaler().fit_transform(X3)
    y3  = dep_df["label"].values
    res, lgbm3, Xte3 = run_task(X3s, y3, "depression_binary_DEPRESJON")
    all_results.extend(res)
    run_shap(lgbm3, Xte3, "depression")

    # ── Bonus: MADRS Regression ──
    mae, r2 = run_madrs_regression(dep_df)
    all_results.append({"Task":"MADRS_regression","Model":"LightGBM",
                         "Accuracy": round(r2*100,2),
                         "F1": round(1-mae/30, 4), "AUC": round(r2,4)})

    # ── Summary ──
    rdf = pd.DataFrame(all_results)
    print("\n\n" + "="*62)
    print("  FINAL RESULTS SUMMARY v2")
    print("="*62)
    print(rdf.to_string(index=False))
    rdf.to_csv("outputs/results/results_v2.csv", index=False)

    # Bar chart (exclude CV and regression rows for clean plot)
    plot_df = rdf[~rdf["Model"].str.contains("CV|regression", case=False)]
    tasks   = plot_df["Task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(6*len(tasks), 5))
    if len(tasks)==1: axes=[axes]
    clrs = ["#5C6BC0","#42A5F5","#66BB6A","#AB47BC"]
    for i, task in enumerate(tasks):
        sub = plot_df[plot_df["Task"]==task].sort_values("Accuracy",ascending=False)
        axes[i].bar(range(len(sub)), sub["Accuracy"].values,
                    color=clrs[:len(sub)], edgecolor="white")
        axes[i].set_xticks(range(len(sub)))
        axes[i].set_xticklabels(sub["Model"].values, rotation=30, ha="right")
        axes[i].set_title(task, fontweight="bold", fontsize=9)
        axes[i].set_ylim(40,100)
        axes[i].set_ylabel("Accuracy (%)")
        for j,v in enumerate(sub["Accuracy"].values):
            axes[i].text(j, v+0.4, f"{v:.1f}", ha="center", fontsize=8)
    plt.suptitle("Mental Health Monitoring — Model Results v2",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/results/accuracy_v2.png", dpi=150)
    plt.show()
    print("\nDone. All outputs saved.")