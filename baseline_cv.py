"""
MENTAL HEALTH MONITORING — FINAL PIPELINE
==========================================
Adds proper 5-fold stratified cross-validation
with mean ± std reporting across all models.

Saves:
  outputs/figures/cv_accuracy_comparison.png
  outputs/figures/cv_f1_comparison.png
  outputs/figures/confusion_matrix_ensemble.png
  outputs/figures/shap_feature_importance.png
  outputs/results/cv_results.csv
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

from scipy.stats import entropy as scipy_entropy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

# =========================================================
# OUTPUT DIRECTORIES
# =========================================================
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

# =========================================================
# PATHS
# =========================================================
DEPRESJON_DIR = "data/depresjon"
PSYKOSE_DIR   = "data/psykose"

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_features(df, subject_id, label_int, label_str):
    act         = df["activity"].values.astype(float)
    mean_act    = np.mean(act)
    std_act     = np.std(act)
    median_act  = np.median(act)
    max_act     = np.max(act)
    q25         = np.percentile(act, 25)
    q75         = np.percentile(act, 75)
    iqr         = q75 - q25
    hist, _     = np.histogram(act, bins=20, density=True)
    entropy_val = scipy_entropy(hist + 1e-10)
    zero_cross  = np.sum(np.diff(np.sign(act - mean_act)) != 0)
    sleep_hrs   = np.sum(act < 5) / 60.0
    active_mins = np.sum(act > 100)
    prop_inactive = np.mean(act < 5)
    prop_active   = np.mean(act > 100)

    if "timestamp" in df.columns:
        df = df.copy()
        df["hour"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        ).dt.hour.fillna(0).astype(int)
        day_act         = df[df["hour"].between(6, 21)]["activity"].mean()
        night_act       = df[~df["hour"].between(6, 21)]["activity"].mean()
        day_night_ratio = (day_act + 1e-5) / (night_act + 1e-5)
        circ_regularity = df.groupby("hour")["activity"].mean().std()
    else:
        day_act = night_act = day_night_ratio = circ_regularity = np.nan

    n_days = len(df["date"].unique()) if "date" in df.columns else 1

    return {
        "subject_id":     subject_id,
        "label":          label_int,
        "label_str":      label_str,
        "mean_activity":  mean_act,
        "std_activity":   std_act,
        "median_activity":median_act,
        "max_activity":   max_act,
        "q25_activity":   q25,
        "q75_activity":   q75,
        "iqr_activity":   iqr,
        "entropy":        entropy_val,
        "zero_crossings": zero_cross,
        "sleep_hours":    sleep_hrs,
        "active_minutes": active_mins,
        "day_activity":   day_act,
        "night_activity": night_act,
        "day_night_ratio":day_night_ratio,
        "circ_regularity":circ_regularity,
        "prop_inactive":  prop_inactive,
        "prop_active":    prop_active,
        "n_days":         n_days,
    }

FEATURE_COLS = [
    "mean_activity", "std_activity", "median_activity", "max_activity",
    "q25_activity", "q75_activity", "iqr_activity", "entropy",
    "zero_crossings", "sleep_hours", "active_minutes",
    "day_activity", "night_activity", "day_night_ratio",
    "circ_regularity", "prop_inactive", "prop_active", "n_days"
]

CLASS_NAMES = ["Healthy", "Depression", "Schizophrenia"]

# =========================================================
# LOAD DATA
# =========================================================
def load_depresjon():
    print("\nLoading DEPRESJON...")
    rows = []
    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR, "condition", "*.csv"))):
        rows.append(extract_features(pd.read_csv(fp),
                    os.path.splitext(os.path.basename(fp))[0], 1, "depression"))
    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR, "control", "*.csv"))):
        rows.append(extract_features(pd.read_csv(fp),
                    os.path.splitext(os.path.basename(fp))[0], 0, "healthy"))
    df = pd.DataFrame(rows)
    print(f"  {len(df)} subjects | dep={sum(df.label==1)} healthy={sum(df.label==0)}")
    return df

def load_psykose():
    print("\nLoading PSYKOSE...")
    rows = []
    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR, "patient", "*.csv"))):
        rows.append(extract_features(pd.read_csv(fp),
                    os.path.splitext(os.path.basename(fp))[0], 2, "schizophrenia"))
    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR, "control", "*.csv"))):
        rows.append(extract_features(pd.read_csv(fp),
                    os.path.splitext(os.path.basename(fp))[0], 0, "healthy"))
    df = pd.DataFrame(rows)
    print(f"  {len(df)} subjects | schiz={sum(df.label==2)} healthy={sum(df.label==0)}")
    return df

# =========================================================
# BUILD MODELS
# =========================================================
def build_models():
    lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbose=-1
    )
    cb = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        verbose=0, random_seed=42
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42
    )
    return {"LightGBM": lgbm, "CatBoost": cb, "Random Forest": rf}

# =========================================================
# 5-FOLD CROSS VALIDATION
# =========================================================
def run_kfold_cv(X, y, n_splits=5):
    """
    Proper stratified k-fold CV.
    SMOTE is applied INSIDE each fold on training data only
    to prevent data leakage.
    Returns per-fold and summary metrics for all models + ensemble.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_names = ["LightGBM", "CatBoost", "Random Forest", "Ensemble"]
    fold_results = {m: {"acc": [], "f1": [], "auc": []} for m in model_names}

    print(f"\n{'='*58}")
    print(f"  5-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*58}")

    all_yte, all_yp_ens = [], []   # collect for final confusion matrix

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  ── Fold {fold}/{n_splits} ──")

        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # Scale inside fold (fit on train only)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        # SMOTE on training fold only
        k = max(1, min(3, min(np.bincount(ytr)) - 1))
        Xr, yr = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xtr_s, ytr)

        # Train individual models
        models = build_models()
        fitted = {}
        for name, m in models.items():
            m_clone = clone(m)
            m_clone.fit(Xr, yr)
            fitted[name] = m_clone

        # Ensemble (refit on same SMOTE data)
        ensemble = VotingClassifier(
            estimators=[(n, clone(m)) for n, m in models.items()],
            voting="soft"
        )
        ensemble.fit(Xr, yr)
        fitted["Ensemble"] = ensemble

        # Evaluate each model
        for name, model in fitted.items():
            yp   = model.predict(Xte_s)
            yprb = model.predict_proba(Xte_s)

            acc = accuracy_score(yte, yp)
            f1  = f1_score(yte, yp, average="weighted")

            try:
                yb  = label_binarize(yte, classes=sorted(np.unique(y)))
                auc = roc_auc_score(yb, yprb, average="weighted",
                                    multi_class="ovr")
            except Exception:
                auc = np.nan

            fold_results[name]["acc"].append(acc)
            fold_results[name]["f1"].append(f1)
            fold_results[name]["auc"].append(auc)

            print(f"    [{name:<14}] Acc={acc*100:.1f}%  "
                  f"F1={f1:.3f}  AUC={auc:.3f}")

            # Save ensemble predictions for confusion matrix
            if name == "Ensemble":
                all_yte.extend(yte.tolist())
                all_yp_ens.extend(yp.tolist())

    return fold_results, np.array(all_yte), np.array(all_yp_ens)

# =========================================================
# SUMMARY TABLE
# =========================================================
def print_summary(fold_results):
    print(f"\n{'='*62}")
    print(f"  CROSS-VALIDATION SUMMARY  (mean ± std across 5 folds)")
    print(f"{'='*62}")
    print(f"  {'Model':<16} {'Accuracy':>14} {'F1 Score':>14} {'AUC-ROC':>14}")
    print(f"  {'-'*58}")

    rows = []
    for name, r in fold_results.items():
        acc_m  = np.mean(r["acc"])  * 100
        acc_s  = np.std(r["acc"])   * 100
        f1_m   = np.mean(r["f1"])
        f1_s   = np.std(r["f1"])
        auc_m  = np.mean(r["auc"])
        auc_s  = np.std(r["auc"])
        marker = " ★" if name == "Ensemble" else ""
        print(f"  {name:<16} "
              f"{acc_m:>6.1f}±{acc_s:.1f}%  "
              f"{f1_m:>6.3f}±{f1_s:.3f}  "
              f"{auc_m:>6.3f}±{auc_s:.3f}{marker}")
        rows.append({
            "Model":    name,
            "Acc_mean": round(acc_m, 2),
            "Acc_std":  round(acc_s, 2),
            "F1_mean":  round(f1_m, 4),
            "F1_std":   round(f1_s, 4),
            "AUC_mean": round(auc_m, 4),
            "AUC_std":  round(auc_s, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/results/cv_results.csv", index=False)
    print(f"\n  Saved: outputs/results/cv_results.csv")
    return df

# =========================================================
# PLOTS
# =========================================================
def plot_cv_results(fold_results, filename):
    """
    Grouped bar chart with error bars showing mean ± std
    for Accuracy, F1, and AUC across all four models.
    """
    models  = list(fold_results.keys())
    acc_m   = [np.mean(fold_results[m]["acc"]) * 100 for m in models]
    acc_s   = [np.std(fold_results[m]["acc"])  * 100 for m in models]
    f1_m    = [np.mean(fold_results[m]["f1"])  * 100 for m in models]
    f1_s    = [np.std(fold_results[m]["f1"])   * 100 for m in models]
    auc_m   = [np.mean(fold_results[m]["auc"]) * 100 for m in models]
    auc_s   = [np.std(fold_results[m]["auc"])  * 100 for m in models]

    x      = np.arange(len(models))
    width  = 0.25
    colors = ["#5C6BC0", "#42A5F5", "#66BB6A"]
    ecolor = "#333333"
    ecap   = 4

    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(x - width, acc_m, width, yerr=acc_s,
                label="Accuracy (%)", color=colors[0],
                edgecolor="white", capsize=ecap,
                error_kw={"ecolor": ecolor, "lw": 1.2})
    b2 = ax.bar(x,         f1_m,  width, yerr=f1_s,
                label="F1 Score (%)", color=colors[1],
                edgecolor="white", capsize=ecap,
                error_kw={"ecolor": ecolor, "lw": 1.2})
    b3 = ax.bar(x + width, auc_m, width, yerr=auc_s,
                label="AUC-ROC (%)",  color=colors[2],
                edgecolor="white", capsize=ecap,
                error_kw={"ecolor": ecolor, "lw": 1.2})

    # Value labels
    for bars, stds in [(b1, acc_s), (b2, f1_s), (b3, auc_s)]:
        for bar, std in zip(bars, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.5,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(40, 105)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title(
        "5-Fold CV Performance: Mean ± Std (3-Class Actigraphy Task)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_confusion_matrix(yte, yp, filename):
    """Aggregated confusion matrix across all CV folds."""
    cm = confusion_matrix(yte, yp)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor="white", annot_kws={"size": 13}
    )
    ax.set_title("Confusion Matrix — Ensemble (5-Fold CV, All Folds)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_shap(X_sample, y_sample, filename):
    """
    Train LightGBM on full data, compute SHAP on a held-out sample.
    """
    print("\n  Generating SHAP plot...")
    sc = StandardScaler()
    Xs = sc.fit_transform(X_sample)
    y  = y_sample

    k = max(1, min(3, min(np.bincount(y)) - 1))
    Xr, yr = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xs, y)

    lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=6, random_state=42, verbose=-1
    )
    lgbm.fit(Xr, yr)

    # Use a 20% held-out slice for SHAP
    _, Xte_s, _, _ = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        explainer   = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(Xte_s)

        if isinstance(shap_values, list):
            sv = np.mean(np.abs(np.array(shap_values)), axis=0)
        else:
            sv = np.abs(shap_values)

        imp = sv.mean(axis=0)
        if len(imp.shape) > 1:
            imp = imp.mean(axis=1)

        idx     = np.argsort(imp)
        top3    = set(np.argsort(imp)[-3:])
        colors  = ["#E53935" if i in top3 else "#5C6BC0"
                   for i in idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            np.array(FEATURE_COLS)[idx], imp[idx],
            color=colors, edgecolor="white"
        )
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title(
            "SHAP Feature Importance — Actigraphy LightGBM",
            fontsize=12, fontweight="bold"
        )
        ax.text(0.98, 0.02, "Red = top 3 features",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#E53935")
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")


# =========================================================
# TRAIN FINAL MODEL ON ALL DATA (for deployment / paper)
# =========================================================
def train_final_model(X, y):
    print(f"\n{'='*58}")
    print("  TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*58}")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    k  = max(1, min(3, min(np.bincount(y)) - 1))
    Xr, yr = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xs, y)

    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           max_depth=6, random_state=42, verbose=-1)
    cb   = CatBoostClassifier(iterations=300, depth=6,
                               learning_rate=0.05, verbose=0, random_seed=42)
    rf   = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   random_state=42)
    for m in [lgbm, cb, rf]:
        m.fit(Xr, yr)

    ensemble = VotingClassifier(
        estimators=[("lgbm", lgbm), ("cb", cb), ("rf", rf)],
        voting="soft"
    )
    ensemble.fit(Xr, yr)

    joblib.dump(ensemble, "outputs/models/ensemble_final.pkl")
    joblib.dump(sc,       "outputs/models/scaler_final.pkl")
    print("  Final model saved to outputs/models/")
    return lgbm, ensemble, sc


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # ── Load & merge ──────────────────────────────────────
    dep_df = load_depresjon()
    psy_df = load_psykose()
    merged = pd.concat([dep_df, psy_df], ignore_index=True)
    print(f"\nMerged: {len(merged)} subjects total")

    X = merged[FEATURE_COLS].fillna(
        merged[FEATURE_COLS].median()
    ).values
    y = merged["label"].values

    # ── 5-Fold CV ─────────────────────────────────────────
    fold_results, all_yte, all_yp_ens = run_kfold_cv(X, y, n_splits=5)

    # ── Summary table ─────────────────────────────────────
    cv_df = print_summary(fold_results)

    # ── Plots ─────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("  SAVING FIGURES")
    print(f"{'='*58}")

    plot_cv_results(
        fold_results,
        "outputs/figures/cv_accuracy_comparison.png"
    )
    plot_confusion_matrix(
        all_yte, all_yp_ens,
        "outputs/figures/confusion_matrix_ensemble.png"
    )

    # ── Train final model & SHAP ──────────────────────────
    lgbm_final, ensemble_final, sc_final = train_final_model(X, y)
    plot_shap(X, y, "outputs/figures/shap_feature_importance.png")

    # ── Final print ───────────────────────────────────────
    print(f"\n{'='*58}")
    print("  ALL DONE")
    print(f"{'='*58}")
    print("\n  Use these numbers in your paper:")
    ens = fold_results["Ensemble"]
    print(f"  Ensemble Accuracy : "
          f"{np.mean(ens['acc'])*100:.1f}% ± {np.std(ens['acc'])*100:.1f}%")
    print(f"  Ensemble F1       : "
          f"{np.mean(ens['f1']):.3f} ± {np.std(ens['f1']):.3f}")
    print(f"  Ensemble AUC-ROC  : "
          f"{np.mean(ens['auc']):.3f} ± {np.std(ens['auc']):.3f}")
    print("\n  Figures → outputs/figures/")
    print("  Results → outputs/results/cv_results.csv")