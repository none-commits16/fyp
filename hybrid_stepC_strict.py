# ============================================================
# HYBRID FRAMEWORK - STEP C
# STRICT NESTED OOF STACKING
#
# Outer CV:
#   5-fold subject-level CV
#
# Base models:
#   1. Random Forest
#   2. CatBoost
#   3. LightGBM
#   4. Step-B CNN + handcrafted fusion
#
# Meta learner:
#   Logistic Regression
#
# IMPORTANT:
#   Meta-training predictions are generated using INNER CV.
#   Therefore, the meta learner never sees in-sample
#   predictions from the base models.
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

OUT = "outputs/hybrid/stepC_strict"

os.makedirs(
    OUT,
    exist_ok=True
)


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
# LOAD STAGE-3 FEATURE EXTRACTION
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

    ast.fix_missing_locations(
        module
    )

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
# LOAD DATA
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

    df = pd.DataFrame(
        rows
    )

    return df


# ============================================================
# METRICS
# ============================================================

def metrics(
    y_true,
    y_pred,
    probabilities
):

    result = {}

    result["accuracy"] = (
        accuracy_score(
            y_true,
            y_pred
        )
    )

    result["precision"] = (
        precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    result["recall"] = (
        recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    result["f1"] = (
        f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    try:

        result["auc"] = (
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="weighted"
            )
        )

    except Exception:

        result["auc"] = np.nan

    return result


# ============================================================
# CREATE BASE MODEL
# ============================================================

def train_base_models(
    X_train,
    y_train
):

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    smote = BorderlineSMOTE(
        random_state=SEED,
        k_neighbors=3
    )

    X_resampled, y_resampled = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CATBOOST
    # --------------------------------------------------------

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
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # LIGHTGBM
    # --------------------------------------------------------

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

    return (
        rf,
        cat,
        lgb
    )


# ============================================================
# GET PREDICTIONS
# ============================================================

def get_predictions(
    models,
    X
):

    rf, cat, lgb = models

    rf_prob = rf.predict_proba(
        X
    )

    cat_prob = cat.predict_proba(
        X
    )

    lgb_prob = lgb.predict_proba(
        X
    )

    return (
        rf_prob,
        cat_prob,
        lgb_prob
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "HYBRID FRAMEWORK - STEP C"
    )
    print(
        "STRICT NESTED OOF STACKING"
    )
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    data = load_data()

    print(
        "\nSubjects:",
        len(data)
    )

    print(
        "\nClass distribution:"
    )

    print(
        data["label_str"].value_counts()
    )

    if len(data) != 109:

        print(
            "\nWARNING:"
        )

        print(
            "Expected 109 subjects, "
            f"found {len(data)}"
        )

    # ========================================================
    # CHECK DUPLICATES
    # ========================================================

    if data[
        "subject_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate subject IDs found."
        )

    # ========================================================
    # DATA ARRAYS
    # ========================================================

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

    # ========================================================
    # LOAD STEP B PREDICTIONS
    # ========================================================

    stepB_path = (
        "outputs/hybrid/stepB/"
        "stepB_subject_predictions.csv"
    )

    if not os.path.exists(
        stepB_path
    ):

        raise FileNotFoundError(
            "\nStep B predictions not found:\n"
            + stepB_path
        )

    stepB = pd.read_csv(
        stepB_path
    )

    required = [
        "subject_id",
        "fold",
        "p_healthy",
        "p_depression",
        "p_schizophrenia"
    ]

    for col in required:

        if col not in stepB.columns:

            raise ValueError(
                f"Step B missing column: {col}"
            )

    # ========================================================
    # OUTER CV
    # ========================================================

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    outer_results = []

    final_oof = []

    # ========================================================
    # OUTER FOLD LOOP
    # ========================================================

    for outer_fold, (
        outer_train_idx,
        outer_test_idx
    ) in enumerate(
        outer_cv.split(
            X,
            y
        ),
        start=1
    ):

        print("\n")
        print("=" * 70)

        print(
            f"OUTER FOLD "
            f"{outer_fold}/5"
        )

        print("=" * 70)

        X_outer_train = X[
            outer_train_idx
        ]

        y_outer_train = y[
            outer_train_idx
        ]

        X_outer_test = X[
            outer_test_idx
        ]

        y_outer_test = y[
            outer_test_idx
        ]

        test_subjects = (
            subject_ids[
                outer_test_idx
            ]
        )

        print(
            "Outer train:",
            len(outer_train_idx)
        )

        print(
            "Outer test:",
            len(outer_test_idx)
        )

        # ====================================================
        # INNER OOF PREDICTIONS
        # ====================================================

        inner_oof = np.zeros(
            (
                len(
                    outer_train_idx
                ),
                12
            )
        )

        inner_cv = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=SEED
        )

        print(
            "\nGenerating inner OOF "
            "predictions..."
        )

        for inner_fold, (
            inner_train_local,
            inner_val_local
        ) in enumerate(
            inner_cv.split(
                X_outer_train,
                y_outer_train
            ),
            start=1
        ):

            print(
                f"  Inner fold "
                f"{inner_fold}/4"
            )

            # ------------------------------------------------
            # Local split
            # ------------------------------------------------

            X_inner_train = (
                X_outer_train[
                    inner_train_local
                ]
            )

            y_inner_train = (
                y_outer_train[
                    inner_train_local
                ]
            )

            X_inner_val = (
                X_outer_train[
                    inner_val_local
                ]
            )

            # ------------------------------------------------
            # IMPUTATION
            # ------------------------------------------------

            imputer = SimpleImputer(
                strategy="median"
            )

            X_inner_train = (
                imputer.fit_transform(
                    X_inner_train
                )
            )

            X_inner_val = (
                imputer.transform(
                    X_inner_val
                )
            )

            # ------------------------------------------------
            # SCALING
            # ------------------------------------------------

            scaler = StandardScaler()

            X_inner_train = (
                scaler.fit_transform(
                    X_inner_train
                )
            )

            X_inner_val = (
                scaler.transform(
                    X_inner_val
                )
            )

            # ------------------------------------------------
            # TRAIN BASE MODELS
            # ------------------------------------------------

            models = train_base_models(
                X_inner_train,
                y_inner_train
            )

            # ------------------------------------------------
            # PREDICT INNER VALIDATION
            # ------------------------------------------------

            (
                rf_prob,
                cat_prob,
                lgb_prob
            ) = get_predictions(
                models,
                X_inner_val
            )

            # ------------------------------------------------
            # FUSION TRAINING SIGNAL
            #
            # For the meta training matrix, the fusion
            # component is represented using the average
            # of the three independently OOF tree models.
            #
            # This remains independent of the inner
            # validation labels.
            # ------------------------------------------------

            fusion_prob = (
                rf_prob +
                cat_prob +
                lgb_prob
            ) / 3.0

            inner_oof[
                inner_val_local,
                :
            ] = np.column_stack(
                [
                    rf_prob,
                    cat_prob,
                    lgb_prob,
                    fusion_prob
                ]
            )

        # ====================================================
        # TRAIN FINAL BASE MODELS ON COMPLETE OUTER TRAIN
        # ====================================================

        print(
            "\nTraining final outer-fold "
            "base models..."
        )

        imputer_outer = SimpleImputer(
            strategy="median"
        )

        X_train_imp = (
            imputer_outer.fit_transform(
                X_outer_train
            )
        )

        X_test_imp = (
            imputer_outer.transform(
                X_outer_test
            )
        )

        scaler_outer = StandardScaler()

        X_train_scaled = (
            scaler_outer.fit_transform(
                X_train_imp
            )
        )

        X_test_scaled = (
            scaler_outer.transform(
                X_test_imp
            )
        )

        models_outer = train_base_models(
            X_train_scaled,
            y_outer_train
        )

        # ====================================================
        # OUTER TEST BASE PREDICTIONS
        # ====================================================

        print(
            "Generating outer test "
            "predictions..."
        )

        (
            rf_test,
            cat_test,
            lgb_test
        ) = get_predictions(
            models_outer,
            X_test_scaled
        )

        # ====================================================
        # GET STEP-B FUSION PREDICTIONS
        # ====================================================

        stepB_fold = stepB[
            stepB["fold"]
            == outer_fold
        ].copy()

        stepB_fold = (
            stepB_fold
            .set_index(
                "subject_id"
            )
        )

        fusion_test = np.zeros(
            (
                len(
                    outer_test_idx
                ),
                3
            )
        )

        for i, sid in enumerate(
            test_subjects
        ):

            if sid not in stepB_fold.index:

                raise ValueError(
                    f"Subject {sid} "
                    f"not found in Step B "
                    f"fold {outer_fold}"
                )

            fusion_test[i] = (
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
        # OUTER TEST META FEATURES
        # ====================================================

        outer_test_meta = np.column_stack(
            [
                rf_test,
                cat_test,
                lgb_test,
                fusion_test
            ]
        )

        # ====================================================
        # TRAIN META LEARNER
        #
        # ONLY INNER OOF PREDICTIONS
        # ====================================================

        print(
            "Training strict OOF "
            "meta learner..."
        )

        meta = LogisticRegression(
            max_iter=1000,
            C=0.5,
            random_state=SEED
        )

        meta.fit(
            inner_oof,
            y_outer_train
        )

        # ====================================================
        # FINAL OUTER TEST PREDICTION
        # ====================================================

        final_prob = meta.predict_proba(
            outer_test_meta
        )

        final_pred = np.argmax(
            final_prob,
            axis=1
        )

        # ====================================================
        # METRICS
        # ====================================================

        result = metrics(
            y_outer_test,
            final_pred,
            final_prob
        )

        result[
            "fold"
        ] = outer_fold

        outer_results.append(
            result
        )

        print(
            "\nOuter Fold Results:"
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
        # SAVE OUTER OOF PREDICTIONS
        # ====================================================

        fold_predictions = pd.DataFrame(
            {
                "subject_id":
                    test_subjects,

                "true_label":
                    y_outer_test,

                "pred_label":
                    final_pred,

                "p_healthy":
                    final_prob[:, 0],

                "p_depression":
                    final_prob[:, 1],

                "p_schizophrenia":
                    final_prob[:, 2],

                "fold":
                    outer_fold
            }
        )

        final_oof.append(
            fold_predictions
        )

    # ========================================================
    # COMBINE OUTER OOF
    # ========================================================

    results_df = pd.DataFrame(
        outer_results
    )

    oof_df = pd.concat(
        final_oof,
        ignore_index=True
    )

    oof_df = (
        oof_df
        .sort_values(
            "subject_id"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    results_df.to_csv(
        os.path.join(
            OUT,
            "strict_stepC_by_fold.csv"
        ),
        index=False
    )

    # ========================================================
    # SAVE OOF PREDICTIONS
    # ========================================================

    oof_df.to_csv(
        os.path.join(
            OUT,
            "strict_stepC_oof_predictions.csv"
        ),
        index=False
    )

    # ========================================================
    # 5-FOLD SUMMARY
    # ========================================================

    summary_rows = []

    for metric_name in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]:

        summary_rows.append(
            {
                "metric":
                    metric_name,

                "mean":
                    results_df[
                        metric_name
                    ].mean(),

                "std":
                    results_df[
                        metric_name
                    ].std()
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        os.path.join(
            OUT,
            "strict_stepC_summary.csv"
        ),
        index=False
    )

    # ========================================================
    # OVERALL OOF
    # ========================================================

    overall_accuracy = (
        accuracy_score(
            oof_df["true_label"],
            oof_df["pred_label"]
        )
    )

    overall_precision = (
        precision_score(
            oof_df["true_label"],
            oof_df["pred_label"],
            average="weighted",
            zero_division=0
        )
    )

    overall_recall = (
        recall_score(
            oof_df["true_label"],
            oof_df["pred_label"],
            average="weighted",
            zero_division=0
        )
    )

    overall_f1 = (
        f1_score(
            oof_df["true_label"],
            oof_df["pred_label"],
            average="weighted",
            zero_division=0
        )
    )

    try:

        overall_auc = (
            roc_auc_score(
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
        )

    except Exception:

        overall_auc = np.nan

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "STRICT STEP C FINAL RESULTS"
    )
    print("=" * 70)

    print(
        "\n5-Fold Mean ± SD:"
    )

    for _, row in (
        summary_df.iterrows()
    ):

        print(
            f"{row['metric']:10s}: "
            f"{row['mean']:.4f} ± "
            f"{row['std']:.4f}"
        )

    print(
        "\nOverall Strict OOF:"
    )

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

    print(
        "\nSaved to:"
    )

    print(
        OUT
    )

    print(
        "\nSTRICT STEP C COMPLETE."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()