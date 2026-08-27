"""
STAGE 1 — RIGOROUS VALIDATION
DEPRESJON + PSYKOSE

Implements the guide's first three requirements:
1. 5-fold stratified cross-validation
2. Re-run RF, CatBoost and LightGBM with:
   accuracy, precision, recall, F1 and AUC
3. Nested CV for hyperparameter tuning

Important:
- No SMOTE is used in this stage. This is the clean baseline.
- Imputation and scaling are fitted inside each fold to prevent leakage.
- The outer CV estimates generalization.
- The inner CV is used only for hyperparameter tuning.
"""

import os
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import entropy as scipy_entropy

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


SEED = 42
N_SPLITS = 5

DEPRESJON_DIR = "data/depresjon"
PSYKOSE_DIR = "data/psykose"

RESULTS_DIR = Path("outputs/stage1_validation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FEATURE EXTRACTION — SAME FEATURES AS CURRENT PIPELINE
# =========================================================

def extract_features(df, subject_id, label_int, label_str):
    act = df["activity"].values.astype(float)

    mean_act = np.mean(act)
    std_act = np.std(act)
    median_act = np.median(act)
    max_act = np.max(act)

    q25 = np.percentile(act, 25)
    q75 = np.percentile(act, 75)
    iqr = q75 - q25

    hist, _ = np.histogram(act, bins=20, density=True)
    entropy_val = scipy_entropy(hist + 1e-10)

    zero_cross = np.sum(np.diff(np.sign(act - mean_act)) != 0)

    sleep_hrs = np.sum(act < 5) / 60.0
    active_mins = np.sum(act > 100)
    prop_inactive = np.mean(act < 5)
    prop_active = np.mean(act > 100)

    if "timestamp" in df.columns:
        temp = df.copy()
        temp["hour"] = pd.to_datetime(temp["timestamp"]).dt.hour

        day_act = temp[temp["hour"].between(6, 21)]["activity"].mean()
        night_act = temp[~temp["hour"].between(6, 21)]["activity"].mean()

        day_night_ratio = (day_act + 1e-5) / (night_act + 1e-5)

        hourly_mean = temp.groupby("hour")["activity"].mean()
        circ_regularity = hourly_mean.std()
    else:
        day_act = np.nan
        night_act = np.nan
        day_night_ratio = np.nan
        circ_regularity = np.nan

    n_days = len(df["date"].unique()) if "date" in df.columns else 1

    return {
        "subject_id": subject_id,
        "label": label_int,
        "label_str": label_str,
        "mean_activity": mean_act,
        "std_activity": std_act,
        "median_activity": median_act,
        "max_activity": max_act,
        "q25_activity": q25,
        "q75_activity": q75,
        "iqr_activity": iqr,
        "entropy": entropy_val,
        "zero_crossings": zero_cross,
        "sleep_hours": sleep_hrs,
        "active_minutes": active_mins,
        "day_activity": day_act,
        "night_activity": night_act,
        "day_night_ratio": day_night_ratio,
        "circ_regularity": circ_regularity,
        "prop_inactive": prop_inactive,
        "prop_active": prop_active,
        "n_days": n_days,
    }


FEATURE_COLS = [
    "mean_activity",
    "std_activity",
    "median_activity",
    "max_activity",
    "q25_activity",
    "q75_activity",
    "iqr_activity",
    "entropy",
    "zero_crossings",
    "sleep_hours",
    "active_minutes",
    "day_activity",
    "night_activity",
    "day_night_ratio",
    "circ_regularity",
    "prop_inactive",
    "prop_active",
    "n_days",
]


# =========================================================
# DATA LOADING
# =========================================================

def load_depresjon():
    rows = []

    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR, "condition", "*.csv"))):
        sid = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        rows.append(extract_features(df, sid, 1, "depression"))

    for fp in sorted(glob.glob(os.path.join(DEPRESJON_DIR, "control", "*.csv"))):
        sid = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        rows.append(extract_features(df, sid, 0, "healthy"))

    return pd.DataFrame(rows)


def load_psykose():
    rows = []

    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR, "patient", "*.csv"))):
        sid = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        rows.append(extract_features(df, sid, 2, "schizophrenia"))

    for fp in sorted(glob.glob(os.path.join(PSYKOSE_DIR, "control", "*.csv"))):
        sid = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        rows.append(extract_features(df, sid, 0, "healthy"))

    return pd.DataFrame(rows)


def load_dataset():
    dep = load_depresjon()
    psy = load_psykose()

    data = pd.concat([dep, psy], ignore_index=True)

  
    X = data[FEATURE_COLS].copy()
    y = data["label"].astype(int).copy()

    print("\nDataset shape:", data.shape)
    print("\nClass distribution:")
    print(data["label_str"].value_counts())
    print("\nClass counts:")
    print(y.value_counts().sort_index())

    return X, y, data


# =========================================================
# PIPELINES
# =========================================================

def make_baseline_models():
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),

        "CatBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                random_seed=SEED,
                verbose=False,
                allow_writing_files=False,
            )),
        ]),

        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=SEED,
                verbosity=-1,
                n_jobs=-1,
            )),
        ]),
    }


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }

    # For this project the task is multiclass (healthy/depression/schizophrenia).
    result["auc_weighted_ovr"] = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="weighted",
        labels=model.classes_,
    )

    return result


# =========================================================
# 1 + 2. 5-FOLD BASELINE VALIDATION
# =========================================================

def run_baseline_cv(X, y):
    print("\n" + "=" * 70)
    print("5-FOLD BASELINE CROSS-VALIDATION")
    print("=" * 70)

    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    rows = []

    for model_name, pipeline in make_baseline_models().items():
        print(f"\n>>> {model_name}")

        for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y), start=1
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = clone(pipeline)
            model.fit(X_train, y_train)

            metrics = calculate_metrics(model, X_test, y_test)

            row = {
                "model": model_name,
                "fold": fold,
                **metrics,
            }
            rows.append(row)

            print(
                f"Fold {fold}: "
                f"Acc={metrics['accuracy']:.4f} | "
                f"Prec={metrics['precision_weighted']:.4f} | "
                f"Recall={metrics['recall_weighted']:.4f} | "
                f"F1={metrics['f1_weighted']:.4f} | "
                f"AUC={metrics['auc_weighted_ovr']:.4f}"
            )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(RESULTS_DIR / "baseline_5fold_by_fold.csv", index=False)

    summary = (
        fold_df
        .groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision_weighted", "mean"),
            precision_std=("precision_weighted", "std"),
            recall_mean=("recall_weighted", "mean"),
            recall_std=("recall_weighted", "std"),
            f1_mean=("f1_weighted", "mean"),
            f1_std=("f1_weighted", "std"),
            auc_mean=("auc_weighted_ovr", "mean"),
            auc_std=("auc_weighted_ovr", "std"),
        )
        .reset_index()
    )

    summary.to_csv(RESULTS_DIR / "baseline_5fold_summary.csv", index=False)

    print("\nBASELINE SUMMARY")
    print(summary.to_string(index=False))

    return fold_df, summary


# =========================================================
# 3. NESTED CV FOR HYPERPARAMETER TUNING
# =========================================================

def make_nested_search_spaces():
    common = {
        "Random Forest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(
                    random_state=SEED,
                    n_jobs=-1,
                )),
            ]),
            {
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 6, 8],
                "model__min_samples_leaf": [1, 2],
            },
        ),

        "CatBoost": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", CatBoostClassifier(
                    random_seed=SEED,
                    verbose=False,
                    allow_writing_files=False,
                )),
            ]),
            {
                "model__iterations": [200, 300],
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
            },
        ),

        "LightGBM": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LGBMClassifier(
                    random_state=SEED,
                    verbosity=-1,
                    n_jobs=-1,
                )),
            ]),
            {
                "model__n_estimators": [200, 300],
                "model__max_depth": [3, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
            },
        ),
    }
    return common


def run_nested_cv(X, y):
    print("\n" + "=" * 70)
    print("NESTED CROSS-VALIDATION")
    print("=" * 70)

    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    # F1 is used for tuning because the guide specifically asks us
    # to prioritize F1/recall rather than raw accuracy in this screening context.
    scoring = "f1_weighted"

    all_rows = []
    search_spaces = make_nested_search_spaces()

    for model_name, (pipeline, param_grid) in search_spaces.items():
        print(f"\n>>> Nested CV: {model_name}")

        for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y), start=1
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=1,  # safer with CatBoost/LightGBM nested parallelism
                refit=True,
            )

            # Hyperparameter tuning sees ONLY the outer training fold.
            search.fit(X_train, y_train)

            # The outer test fold is untouched until this point.
            metrics = calculate_metrics(
                search.best_estimator_,
                X_test,
                y_test,
            )

            row = {
                "model": model_name,
                "outer_fold": fold,
                "best_inner_cv_f1": search.best_score_,
                "best_params": str(search.best_params_),
                **metrics,
            }
            all_rows.append(row)

            print(
                f"Fold {fold}: "
                f"inner-F1={search.best_score_:.4f} | "
                f"outer Acc={metrics['accuracy']:.4f} | "
                f"outer Prec={metrics['precision_weighted']:.4f} | "
                f"outer Recall={metrics['recall_weighted']:.4f} | "
                f"outer F1={metrics['f1_weighted']:.4f} | "
                f"outer AUC={metrics['auc_weighted_ovr']:.4f}"
            )

    nested_df = pd.DataFrame(all_rows)
    nested_df.to_csv(RESULTS_DIR / "nested_cv_by_fold.csv", index=False)

    summary = (
        nested_df
        .groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision_weighted", "mean"),
            precision_std=("precision_weighted", "std"),
            recall_mean=("recall_weighted", "mean"),
            recall_std=("recall_weighted", "std"),
            f1_mean=("f1_weighted", "mean"),
            f1_std=("f1_weighted", "std"),
            auc_mean=("auc_weighted_ovr", "mean"),
            auc_std=("auc_weighted_ovr", "std"),
        )
        .reset_index()
    )

    summary.to_csv(RESULTS_DIR / "nested_cv_summary.csv", index=False)

    print("\nNESTED CV SUMMARY")
    print(summary.to_string(index=False))

    return nested_df, summary


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 1 — RIGOROUS BASELINE VALIDATION")
    print("=" * 70)

    X, y, data = load_dataset()

    # Requirement 1 + 2
    run_baseline_cv(X, y)

    # Requirement 3
    run_nested_cv(X, y)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Results saved to: {RESULTS_DIR.resolve()}")
    print("=" * 70)
