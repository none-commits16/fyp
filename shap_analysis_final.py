# ============================================================
# FINAL SHAP ANALYSIS
#
# Uses the EXACT 27 enhanced features from:
#     stage3_feature_enhancement.py
#
# No pre-generated stage3_features.csv is required.
#
# Dataset:
#     DEPRESJON + PSYKOSE
#
# Model:
#     LightGBM
#
# Outputs:
#     outputs/shap/
# ============================================================

import os
import glob
import ast
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.impute import SimpleImputer


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

OUT_DIR = "outputs/shap"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# DATA PATHS
# ============================================================

DEP_CONDITION = "data/depresjon/condition/*.csv"

DEP_CONTROL = "data/depresjon/control/*.csv"

PSY_PATIENT = "data/psykose/patient/*.csv"

PSY_CONTROL = "data/psykose/control/*.csv"


# ============================================================
# EXACT 27 FEATURES USED IN STAGE 3
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
    "lz_complexity"
]


# ============================================================
# LOAD EXACT EXTRACT() FUNCTION FROM STAGE 3
# ============================================================

def load_stage3_extract():

    path = "stage3_feature_enhancement.py"

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Cannot find {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        source = f.read()

    tree = ast.parse(
        source
    )

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

    if "extract" not in namespace:

        raise ValueError(
            "extract() was not found in "
            "stage3_feature_enhancement.py"
        )

    return namespace["extract"]


# ============================================================
# BUILD FEATURE DATASET
# ============================================================

def build_dataset():

    extract = load_stage3_extract()

    rows = []

    # --------------------------------------------------------
    # DEPRESJON - DEPRESSION
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(DEP_CONDITION)
    ):

        try:

            row = extract(
                filepath,
                1,
                "depression"
            )

            row["subject_id"] = (
                "DEP_" +
                os.path.splitext(
                    os.path.basename(filepath)
                )[0]
            )

            rows.append(row)

        except Exception as e:

            print(
                "Skipping:",
                filepath,
                "|",
                str(e)
            )

    # --------------------------------------------------------
    # DEPRESJON - HEALTHY
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(DEP_CONTROL)
    ):

        try:

            row = extract(
                filepath,
                0,
                "healthy"
            )

            row["subject_id"] = (
                "DEP_" +
                os.path.splitext(
                    os.path.basename(filepath)
                )[0]
            )

            rows.append(row)

        except Exception as e:

            print(
                "Skipping:",
                filepath,
                "|",
                str(e)
            )

    # --------------------------------------------------------
    # PSYKOSE - SCHIZOPHRENIA
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(PSY_PATIENT)
    ):

        try:

            row = extract(
                filepath,
                2,
                "schizophrenia"
            )

            row["subject_id"] = (
                "PSY_" +
                os.path.splitext(
                    os.path.basename(filepath)
                )[0]
            )

            rows.append(row)

        except Exception as e:

            print(
                "Skipping:",
                filepath,
                "|",
                str(e)
            )

    # --------------------------------------------------------
    # PSYKOSE - HEALTHY
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(PSY_CONTROL)
    ):

        try:

            row = extract(
                filepath,
                0,
                "healthy"
            )

            row["subject_id"] = (
                "PSY_" +
                os.path.splitext(
                    os.path.basename(filepath)
                )[0]
            )

            rows.append(row)

        except Exception as e:

            print(
                "Skipping:",
                filepath,
                "|",
                str(e)
            )

    df = pd.DataFrame(
        rows
    )

    if len(df) == 0:

        raise ValueError(
            "No feature rows were generated."
        )

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing = [
        f for f in FEATURES
        if f not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing Stage 3 features:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    X = df[
        FEATURES
    ].copy()

    y = df[
        "label"
    ].astype(int)

    subjects = df[
        "subject_id"
    ].astype(str)

    return (
        df,
        X,
        y,
        subjects
    )


# ============================================================
# SHAP VALUE HANDLING
# ============================================================

def get_mean_abs_shap(
    shap_values,
    n_features
):

    # --------------------------------------------------------
    # SHAP can return:
    #
    #   list of arrays
    # OR
    #
    #   3D numpy array
    #
    # depending on SHAP/model version.
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        stacked = np.stack(
            [
                np.asarray(v)
                for v in shap_values
            ],
            axis=0
        )

        return np.mean(
            np.abs(stacked),
            axis=(0, 1)
        )

    arr = np.asarray(
        shap_values
    )

    # Binary / ordinary 2D case

    if arr.ndim == 2:

        return np.mean(
            np.abs(arr),
            axis=0
        )

    # Multiclass 3D case

    if arr.ndim == 3:

        # Possible shapes:
        #
        # samples × features × classes
        #
        # OR
        #
        # samples × classes × features

        if arr.shape[1] == n_features:

            return np.mean(
                np.abs(arr),
                axis=(0, 2)
            )

        if arr.shape[2] == n_features:

            return np.mean(
                np.abs(arr),
                axis=(0, 1)
            )

    raise ValueError(
        f"Unexpected SHAP shape: {arr.shape}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "FINAL SHAP ANALYSIS"
    )

    print(
        "DEPRESJON + PSYKOSE"
    )

    print(
        "27 Stage-3 Enhanced Features"
    )

    print("=" * 70)

    # ========================================================
    # DATA
    # ========================================================

    print(
        "\nExtracting Stage 3 features..."
    )

    (
        df,
        X,
        y,
        subjects
    ) = build_dataset()

    print(
        "Subjects:",
        len(df)
    )

    print(
        "\nClass distribution:"
    )

    print(
        y.value_counts()
        .sort_index()
    )

    print(
        "\nFeature count:",
        len(FEATURES)
    )

    # ========================================================
    # 5-FOLD CV
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    fold_importance = []

    all_shap_values = []

    all_X_test = []

    fold = 1

    # ========================================================
    # CV LOOP
    # ========================================================

    for train_idx, test_idx in cv.split(
        X,
        y
    ):

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"FOLD {fold}/5"
        )

        print(
            "-" * 60
        )

        X_train = (
            X.iloc[
                train_idx
            ]
            .copy()
        )

        X_test = (
            X.iloc[
                test_idx
            ]
            .copy()
        )

        y_train = (
            y.iloc[
                train_idx
            ]
        )

        # ----------------------------------------------------
        # Imputation
        # ----------------------------------------------------

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train_imp = (
            imputer.fit_transform(
                X_train
            )
        )

        X_test_imp = (
            imputer.transform(
                X_test
            )
        )

        # ----------------------------------------------------
        # LightGBM
        #
        # Tree models do not require scaling.
        # ----------------------------------------------------

        model = LGBMClassifier(

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

        model.fit(
            X_train_imp,
            y_train
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = (
            explainer.shap_values(
                X_test_imp
            )
        )

        importance = (
            get_mean_abs_shap(
                shap_values,
                len(FEATURES)
            )
        )

        fold_importance.append(
            importance
        )

        # ----------------------------------------------------
        # Store values for later combined plot
        # ----------------------------------------------------

        all_shap_values.append(
            shap_values
        )

        all_X_test.append(
            X_test_imp
        )

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        fold_df = pd.DataFrame(
            {
                "feature": FEATURES,
                "mean_abs_shap": importance
            }
        ).sort_values(
            "mean_abs_shap",
            ascending=False
        )

        print(
            "\nTop features:"
        )

        print(
            fold_df.head(10)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Save fold importance
        # ----------------------------------------------------

        fold_df.to_csv(
            os.path.join(
                OUT_DIR,
                f"fold_{fold}_importance.csv"
            ),
            index=False
        )

        fold += 1

    # ========================================================
    # COMBINE FOLD IMPORTANCE
    # ========================================================

    importance_matrix = np.asarray(
        fold_importance
    )

    importance_df = pd.DataFrame(
        importance_matrix,
        columns=FEATURES
    )

    importance_df.index = [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "fold_5"
    ]

    importance_df.to_csv(
        os.path.join(
            OUT_DIR,
            "fold_feature_importance.csv"
        )
    )

    # ========================================================
    # MEAN IMPORTANCE
    # ========================================================

    mean_importance = (
        importance_df
        .mean(axis=0)
        .sort_values(
            ascending=False
        )
    )

    mean_importance_df = (
        pd.DataFrame(
            {
                "feature":
                    mean_importance.index,

                "mean_abs_shap":
                    mean_importance.values
            }
        )
    )

    mean_importance_df.to_csv(
        os.path.join(
            OUT_DIR,
            "mean_feature_importance.csv"
        ),
        index=False
    )

    # ========================================================
    # TOP 15 BAR PLOT
    # ========================================================

    top15 = (
        mean_importance
        .head(15)
        .sort_values()
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.barh(
        top15.index,
        top15.values
    )

    plt.xlabel(
        "Mean |SHAP value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 15 Features by Mean Absolute SHAP Value"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            "top15_feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # TOP 10 PRINT
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "OVERALL SHAP FEATURE IMPORTANCE"
    )

    print("=" * 70)

    print(
        mean_importance_df
        .head(15)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # COMBINED SHAP SUMMARY
    #
    # We need to normalize SHAP representation across
    # SHAP versions.
    # ========================================================

    print(
        "\nGenerating combined SHAP summary..."
    )

    combined_X = np.vstack(
        all_X_test
    )

    # --------------------------------------------------------
    # Convert SHAP outputs to a single importance-compatible
    # representation.
    #
    # For multiclass:
    # average absolute contribution across classes for the
    # summary representation.
    # --------------------------------------------------------

    combined_abs = []

    for sv in all_shap_values:

        if isinstance(
            sv,
            list
        ):

            arr = np.stack(
                sv,
                axis=-1
            )

            # samples × features × classes

            arr = np.mean(
                np.abs(arr),
                axis=2
            )

        else:

            arr = np.asarray(
                sv
            )

            if arr.ndim == 2:

                arr = np.abs(arr)

            elif arr.ndim == 3:

                if arr.shape[1] == len(FEATURES):

                    arr = np.mean(
                        np.abs(arr),
                        axis=2
                    )

                elif arr.shape[2] == len(FEATURES):

                    arr = np.mean(
                        np.abs(arr),
                        axis=1
                    )

                else:

                    raise ValueError(
                        f"Unexpected SHAP shape: "
                        f"{arr.shape}"
                    )

            else:

                raise ValueError(
                    f"Unexpected SHAP dimensions: "
                    f"{arr.ndim}"
                )

        combined_abs.append(
            arr
        )

    combined_abs = np.vstack(
        combined_abs
    )

    # --------------------------------------------------------
    # Beeswarm-compatible summary
    #
    # We use mean absolute SHAP values as the final
    # multiclass importance representation.
    # --------------------------------------------------------

    mean_abs_combined = (
        np.mean(
            combined_abs,
            axis=0
        )
    )

    summary_order = np.argsort(
        mean_abs_combined
    )[::-1]

    top_features = [
        FEATURES[i]
        for i in summary_order[:15]
    ]

    top_values = (
        mean_abs_combined[
            summary_order[:15]
        ]
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.barh(
        top_features[::-1],
        top_values[::-1]
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SHAP Feature Importance Across 5 Folds"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            "shap_summary_final.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # SAVE SUMMARY DATA
    # ========================================================

    summary = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_abs_shap":
                mean_abs_combined
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False
    )

    summary.to_csv(
        os.path.join(
            OUT_DIR,
            "shap_summary_values.csv"
        ),
        index=False
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "SHAP ANALYSIS COMPLETE"
    )

    print("=" * 70)

    print(
        "\nOutput directory:"
    )

    print(
        OUT_DIR
    )

    print(
        "\nGenerated:"
    )

    print(
        "  fold_1_importance.csv"
    )

    print(
        "  fold_2_importance.csv"
    )

    print(
        "  fold_3_importance.csv"
    )

    print(
        "  fold_4_importance.csv"
    )

    print(
        "  fold_5_importance.csv"
    )

    print(
        "  fold_feature_importance.csv"
    )

    print(
        "  mean_feature_importance.csv"
    )

    print(
        "  top15_feature_importance.png"
    )

    print(
        "  shap_summary_final.png"
    )

    print(
        "  shap_summary_values.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()