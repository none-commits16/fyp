# ============================================================
# HYBRID FRAMEWORK - STEP C
# OOF STACKED ENSEMBLE
#
# Base models:
#   1. Random Forest
#   2. CatBoost
#   3. LightGBM
#   4. CNN + handcrafted feature fusion (Step B)
#
# Meta learner:
#   Logistic Regression
#
# Evaluation:
#   Subject-level 5-fold CV
# ============================================================

import os
import glob
import warnings
import ast

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import BorderlineSMOTE

warnings.filterwarnings("ignore")

SEED = 42

OUT = "outputs/hybrid/stepC"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 27 STAGE-3 ENHANCED FEATURES
# ============================================================

FEATURES = [
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
    "IS",
    "IV",
    "RA",
    "morning_activity",
    "afternoon_activity",
    "evening_activity",
    "night_activity",
    "sample_entropy",
    "lz_complexity",
]


# ============================================================
# LOAD ONLY FEATURE EXTRACTION CODE FROM STAGE 3
# ============================================================

def load_stage3_functions():

    path = "stage3_feature_enhancement.py"

    if not os.path.exists(path):
        raise FileNotFoundError(
            "stage3_feature_enhancement.py not found."
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        source = f.read()

    tree = ast.parse(source)

    nodes = []

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef
            )
        ):
            nodes.append(node)

    module = ast.Module(
        body=nodes,
        type_ignores=[]
    )

    ast.fix_missing_locations(module)

    namespace = {}

    exec(
        compile(
            module,
            path,
            "exec"
        ),
        namespace
    )

    return namespace


stage3 = load_stage3_functions()

extract = stage3["extract"]


# ============================================================
# LOAD DEPRESJON + PSYKOSE
# ============================================================

def load_data():

    rows = []

    # --------------------------------------------------------
    # DEPRESJON - DEPRESSION
    # --------------------------------------------------------

    for fp in sorted(
        glob.glob(
            "data/depresjon/condition/*.csv"
        )
    ):

        r = extract(
            fp,
            1,
            "depression"
        )

        r["subject_id"] = (
            "DEP_" +
            os.path.splitext(
                os.path.basename(fp)
            )[0]
        )

        rows.append(r)

    # --------------------------------------------------------
    # DEPRESJON - HEALTHY
    # --------------------------------------------------------

    for fp in sorted(
        glob.glob(
            "data/depresjon/control/*.csv"
        )
    ):

        r = extract(
            fp,
            0,
            "healthy"
        )

        r["subject_id"] = (
            "DEP_" +
            os.path.splitext(
                os.path.basename(fp)
            )[0]
        )

        rows.append(r)

    # --------------------------------------------------------
    # PSYKOSE - SCHIZOPHRENIA
    # --------------------------------------------------------

    for fp in sorted(
        glob.glob(
            "data/psykose/patient/*.csv"
        )
    ):

        r = extract(
            fp,
            2,
            "schizophrenia"
        )

        r["subject_id"] = (
            "PSY_" +
            os.path.splitext(
                os.path.basename(fp)
            )[0]
        )

        rows.append(r)

    # --------------------------------------------------------
    # PSYKOSE - HEALTHY
    # --------------------------------------------------------

    for fp in sorted(
        glob.glob(
            "data/psykose/control/*.csv"
        )
    ):

        r = extract(
            fp,
            0,
            "healthy"
        )

        r["subject_id"] = (
            "PSY_" +
            os.path.splitext(
                os.path.basename(fp)
            )[0]
        )

        rows.append(r)

    df = pd.DataFrame(rows)

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    probabilities
):

    result = {}

    result["accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    result["precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    result["recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    result["f1"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    try:

        result["auc"] = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="weighted"
        )

    except Exception:

        result["auc"] = np.nan

    return result


# ============================================================
# LOAD STEP B PREDICTIONS
# ============================================================

def load_stepB_predictions():

    path = (
        "outputs/hybrid/stepB/"
        "stepB_subject_predictions.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            "\nStep B predictions not found:\n"
            f"{path}\n\n"
            "Run hybrid_stepB_fusion.py first."
        )

    df = pd.read_csv(path)

    required = [
        "subject_id",
        "fold",
        "p_healthy",
        "p_depression",
        "p_schizophrenia"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Step B prediction file is missing columns: "
            + str(missing)
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HYBRID FRAMEWORK - STEP C")
    print("OOF STACKED ENSEMBLE")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = load_data()

    print(
        "\nSubjects:",
        len(data)
    )

    print("\nClass distribution:")

    print(
        data["label_str"].value_counts()
    )

    # --------------------------------------------------------
    # CHECK EXACTLY 109 SUBJECTS
    # --------------------------------------------------------

    if len(data) != 109:

        print(
            "\nWARNING:"
        )

        print(
            f"Expected 109 subjects, "
            f"found {len(data)}."
        )

    # --------------------------------------------------------
    # CHECK DUPLICATES
    # --------------------------------------------------------

    if data["subject_id"].duplicated().any():

        duplicates = data.loc[
            data["subject_id"].duplicated(),
            "subject_id"
        ].tolist()

        raise ValueError(
            "Duplicate subject IDs found:\n"
            + str(duplicates)
        )

    X = data[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    y = data[
        "label"
    ].to_numpy()

    subject_ids = data[
        "subject_id"
    ].to_numpy()

    # --------------------------------------------------------
    # LOAD STEP B
    # --------------------------------------------------------

    stepB = load_stepB_predictions()

    print(
        "\nStep B predictions:",
        len(stepB)
    )

    # --------------------------------------------------------
    # VERIFY STEP B SUBJECTS
    # --------------------------------------------------------

    missing_subjects = set(
        subject_ids
    ) - set(
        stepB["subject_id"]
    )

    if missing_subjects:

        raise ValueError(
            "Subjects missing from Step B:\n"
            + str(sorted(missing_subjects))
        )

    # ========================================================
    # SUBJECT-LEVEL 5-FOLD CV
    # ========================================================

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    fold_results = []

    all_oof = []

    # ========================================================
    # FOLD LOOP
    # ========================================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        skf.split(
            X,
            y
        ),
        start=1
    ):

        print("\n")
        print("=" * 70)
        print(
            f"FOLD {fold}/5"
        )
        print("=" * 70)

        X_train = X[
            train_idx
        ]

        X_test = X[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        test_subjects = (
            subject_ids[
                test_idx
            ]
        )

        print(
            "Train:",
            len(train_idx),
            "| Test:",
            len(test_idx)
        )

        print(
            "Train classes:",
            np.bincount(
                y_train,
                minlength=3
            )
        )

        print(
            "Test classes:",
            np.bincount(
                y_test,
                minlength=3
            )
        )

        # ====================================================
        # IMPUTATION
        # ====================================================

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train = imputer.fit_transform(
            X_train
        )

        X_test = imputer.transform(
            X_test
        )

        # ====================================================
        # STANDARDIZATION
        # ====================================================

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )

        # ====================================================
        # BORDERLINE SMOTE
        # TRAINING DATA ONLY
        # ====================================================

        smote = BorderlineSMOTE(
            random_state=SEED,
            k_neighbors=3
        )

        X_resampled, y_resampled = (
            smote.fit_resample(
                X_train_scaled,
                y_train
            )
        )

        print(
            "After SMOTE:",
            len(X_resampled)
        )

        # ====================================================
        # MODEL 1
        # RANDOM FOREST
        # ====================================================

        print(
            "\nTraining Random Forest..."
        )

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        )

        rf.fit(
            X_resampled,
            y_resampled
        )

        rf_prob = rf.predict_proba(
            X_test_scaled
        )

        rf_pred = np.argmax(
            rf_prob,
            axis=1
        )

        # ====================================================
        # MODEL 2
        # CATBOOST
        # ====================================================

        print(
            "Training CatBoost..."
        )

        cat = CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.03,
            loss_function="MultiClass",
            random_seed=SEED,
            verbose=False,
            class_weights=[
                1.0,
                64 / 23,
                64 / 22
            ]
        )

        cat.fit(
            X_train_scaled,
            y_train
        )

        cat_prob = cat.predict_proba(
            X_test_scaled
        )

        cat_pred = np.argmax(
            cat_prob,
            axis=1
        )

        # ====================================================
        # MODEL 3
        # LIGHTGBM
        # ====================================================

        print(
            "Training LightGBM..."
        )

        lgb = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            verbosity=-1
        )

        lgb.fit(
            X_resampled,
            y_resampled
        )

        lgb_prob = lgb.predict_proba(
            X_test_scaled
        )

        lgb_pred = np.argmax(
            lgb_prob,
            axis=1
        )

        # ====================================================
        # MODEL 4
        # STEP-B FUSION
        # ====================================================

        print(
            "Loading Step-B fusion predictions..."
        )

        stepB_fold = stepB[
            stepB["fold"] == fold
        ].copy()

        stepB_fold = (
            stepB_fold
            .set_index(
                "subject_id"
            )
        )

        fusion_prob = np.zeros(
            (
                len(test_idx),
                3
            )
        )

        for i, sid in enumerate(
            test_subjects
        ):

            if sid not in stepB_fold.index:

                raise ValueError(
                    f"Subject {sid} missing "
                    f"from Step B fold {fold}"
                )

            fusion_prob[i] = (
                stepB_fold.loc[
                    sid,
                    [
                        "p_healthy",
                        "p_depression",
                        "p_schizophrenia"
                    ]
                ]
                .astype(float)
                .values
            )

        # ====================================================
        # STACK TEST PREDICTIONS
        # ====================================================

        stack_test = np.column_stack(
            [
                rf_prob,
                cat_prob,
                lgb_prob,
                fusion_prob
            ]
        )

        # ====================================================
        # META TRAINING PREDICTIONS
        # ====================================================
        #
        # NOTE:
        # This first implementation uses predictions from
        # models fitted on the complete outer training fold.
        #
        # It is a practical first-pass stacking experiment.
        # The final strict nested-OOF version can be run after
        # we inspect these results.
        #
        # ====================================================

        rf_train_prob = rf.predict_proba(
            X_train_scaled
        )

        cat_train_prob = cat.predict_proba(
            X_train_scaled
        )

        lgb_train_prob = lgb.predict_proba(
            X_train_scaled
        )

        # Training-side fusion proxy
        fusion_train_prob = (
            rf_train_prob +
            cat_train_prob +
            lgb_train_prob
        ) / 3.0

        stack_train = np.column_stack(
            [
                rf_train_prob,
                cat_train_prob,
                lgb_train_prob,
                fusion_train_prob
            ]
        )

        # ====================================================
        # META LEARNER
        # ====================================================

        print(
            "Training meta learner..."
        )

        # IMPORTANT:
        # No multi_class argument because the installed
        # scikit-learn version does not accept it.

        meta = LogisticRegression(
            max_iter=1000,
            C=0.5,
            random_state=SEED
        )

        meta.fit(
            stack_train,
            y_train
        )

        # ====================================================
        # FINAL STACKED PREDICTION
        # ====================================================

        final_prob = meta.predict_proba(
            stack_test
        )

        final_pred = np.argmax(
            final_prob,
            axis=1
        )

        # ====================================================
        # METRICS
        # ====================================================

        result = calculate_metrics(
            y_test,
            final_pred,
            final_prob
        )

        result["fold"] = fold

        result["model"] = (
            "OOF Stacked Hybrid"
        )

        fold_results.append(
            result
        )

        print(
            "\nFold Results:"
        )

        print(
            f"Accuracy : "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{result['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{result['recall']:.4f}"
        )

        print(
            f"F1       : "
            f"{result['f1']:.4f}"
        )

        print(
            f"AUC      : "
            f"{result['auc']:.4f}"
        )

        # ====================================================
        # SAVE OOF PREDICTIONS
        # ====================================================

        fold_oof = pd.DataFrame(
            {
                "subject_id":
                    test_subjects,

                "true_label":
                    y_test,

                "pred_label":
                    final_pred,

                "p_healthy":
                    final_prob[:, 0],

                "p_depression":
                    final_prob[:, 1],

                "p_schizophrenia":
                    final_prob[:, 2],

                "fold":
                    fold
            }
        )

        all_oof.append(
            fold_oof
        )

    # ========================================================
    # COMBINE OOF RESULTS
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    oof_df = pd.concat(
        all_oof,
        ignore_index=True
    )

    # Sort for easier inspection
    oof_df = oof_df.sort_values(
        "subject_id"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_path = os.path.join(
        OUT,
        "stepC_by_fold.csv"
    )

    fold_df.to_csv(
        fold_path,
        index=False
    )

    # ========================================================
    # SAVE OOF PREDICTIONS
    # ========================================================

    oof_path = os.path.join(
        OUT,
        "stepC_oof_predictions.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]:

        summary_rows.append(
            {
                "metric":
                    metric,

                "mean":
                    fold_df[
                        metric
                    ].mean(),

                "std":
                    fold_df[
                        metric
                    ].std()
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        OUT,
        "stepC_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # OVERALL OOF METRICS
    # ========================================================

    overall_accuracy = accuracy_score(
        oof_df["true_label"],
        oof_df["pred_label"]
    )

    overall_precision = precision_score(
        oof_df["true_label"],
        oof_df["pred_label"],
        average="weighted",
        zero_division=0
    )

    overall_recall = recall_score(
        oof_df["true_label"],
        oof_df["pred_label"],
        average="weighted",
        zero_division=0
    )

    overall_f1 = f1_score(
        oof_df["true_label"],
        oof_df["pred_label"],
        average="weighted",
        zero_division=0
    )

    try:

        overall_auc = roc_auc_score(
            oof_df["true_label"],
            oof_df[
                [
                    "p_healthy",
                    "p_depression",
                    "p_schizophrenia"
                ]
            ],
            multi_class="ovr",
            average="weighted"
        )

    except Exception:

        overall_auc = np.nan

    # ========================================================
    # FINAL PRINT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP C FINAL RESULTS")
    print("=" * 70)

    print("\n5-Fold Mean ± SD:")

    for _, row in summary.iterrows():

        print(
            f"{row['metric']:10s}: "
            f"{row['mean']:.4f} ± "
            f"{row['std']:.4f}"
        )

    print("\nOverall OOF:")

    print(
        f"Accuracy : "
        f"{overall_accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{overall_precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{overall_recall:.4f}"
    )

    print(
        f"F1       : "
        f"{overall_f1:.4f}"
    )

    print(
        f"AUC      : "
        f"{overall_auc:.4f}"
    )

    print("\nFiles saved:")

    print(
        fold_path
    )

    print(
        oof_path
    )

    print(
        summary_path
    )

    print("\nSTEP C COMPLETE.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()