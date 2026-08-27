# ============================================================
# 09 — EXPLAINABILITY DEEP-DIVE
# Final Hybrid / Stacked Ensemble
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

warnings.filterwarnings("ignore")

# Create output folders
SHAP_DIR = "outputs/explainability/shap"
IG_DIR = "outputs/explainability/integrated_gradients"
RESULTS_DIR = "outputs/explainability/results"

os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(IG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("EXPLAINABILITY DEEP-DIVE")
print("=" * 70)
print("Output folders created successfully.")

# ============================================================
# STEP 1 — LOAD FEATURE FUSION DATA
# ============================================================

FUSION_DIR = "outputs/stage4_feature_fusion"

X_train_fusion = np.load(
    os.path.join(FUSION_DIR, "X_train_fusion.npy")
)

X_test_fusion = np.load(
    os.path.join(FUSION_DIR, "X_test_fusion.npy")
)

y_train = np.load(
    os.path.join(FUSION_DIR, "y_train.npy")
)

y_test = np.load(
    os.path.join(FUSION_DIR, "y_test.npy")
)

print("\nFeature Fusion Data Loaded")
print("-" * 50)
print("X_train_fusion:", X_train_fusion.shape)
print("X_test_fusion:", X_test_fusion.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# ============================================================
# STEP 2 — RECREATE FEATURE-FUSION CATBOOST MODEL
# ============================================================

from catboost import CatBoostClassifier

print("\n" + "=" * 70)
print("RECREATING FEATURE-FUSION CATBOOST MODEL")
print("=" * 70)

fusion_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="MultiClass",
    random_seed=42,
    verbose=100
)

fusion_model.fit(
    X_train_fusion,
    y_train
)

print("✓ Feature-Fusion CatBoost model trained.")

# ============================================================
# STEP 3 — SHAP ANALYSIS OF FEATURE-FUSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("GENERATING SHAP VALUES")
print("=" * 70)

# Use CatBoost's native SHAP implementation.
# This supports the multiclass CatBoost model directly.

from catboost import Pool, EFstrType

test_pool = Pool(X_test_fusion)

shap_values = fusion_model.get_feature_importance(
    data=test_pool,
    type=EFstrType.ShapValues
)

print("✓ Native CatBoost SHAP values generated.")
print("Raw SHAP shape:", np.array(shap_values).shape)

# ------------------------------------------------------------
# Feature names
# ------------------------------------------------------------

feature_names = (
    [f"CNN_Embedding_{i+1}" for i in range(64)]
    +
    [
        "Mean Activity",
        "Std Activity",
        "Min Activity",
        "Max Activity",
        "Median Activity",
        "Total Activity",
        "Mean Active Activity",
        "Std Active Activity",
        "25th Percentile",
        "75th Percentile",
        "90th Percentile",
        "Activity Proportion",
        "Night Activity",
        "Morning Activity",
        "Afternoon Activity",
        "Evening Activity",
        "Mean Activity Change",
        "Std Activity Change",
        "Peak Count"
    ]
)

print("Number of feature names:", len(feature_names))

#------------------------------------------------------------
# Prepare SHAP values for plotting
# ------------------------------------------------------------

# CatBoost multiclass SHAP shape:
# (samples, classes, features + 1)
#
# The last column is the expected value, so we remove it.

shap_values = np.asarray(shap_values)

shap_features = shap_values[:, :, :-1]

print("SHAP feature matrix shape:", shap_features.shape)

# ------------------------------------------------------------
# SHAP summary plot for each class
# ------------------------------------------------------------

class_names = [
    "Healthy",
    "Depression",
    "Schizophrenia"
]

for class_index, class_name in enumerate(class_names):

    class_shap = shap_features[:, class_index, :]

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        class_shap,
        X_test_fusion,
        feature_names=feature_names,
        show=False
    )

    plt.title(
        f"SHAP Summary — {class_name}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = os.path.join(
        SHAP_DIR,
        f"shap_summary_{class_name.lower()}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ SHAP summary saved: {output_path}"
    )

    # ============================================================
# STEP 4 — SHAP DEPENDENCE PLOTS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING SHAP DEPENDENCE PLOTS")
print("=" * 70)

# Generate dependence plots for the top 5 features
# of each class.

for class_index, class_name in enumerate(class_names):

    class_shap = shap_features[:, class_index, :]

    # Calculate mean absolute SHAP importance
    importance = np.mean(
        np.abs(class_shap),
        axis=0
    )

    # Get indices of top 5 features
    top_indices = np.argsort(importance)[-5:][::-1]

    print(f"\nTop features for {class_name}:")

    for rank, feature_index in enumerate(top_indices, start=1):

        print(
            f"{rank}. {feature_names[feature_index]} "
            f"(mean |SHAP| = {importance[feature_index]:.6f})"
        )

        plt.figure(figsize=(8, 6))

        shap.dependence_plot(
            feature_index,
            class_shap,
            X_test_fusion,
            feature_names=feature_names,
            interaction_index="auto",
            show=False
        )

        plt.title(
            f"SHAP Dependence — {class_name}\n"
            f"{feature_names[feature_index]}",
            fontsize=13,
            fontweight="bold"
        )

        plt.tight_layout()

        safe_name = (
            feature_names[feature_index]
            .replace(" ", "_")
            .replace("/", "_")
        )

        output_path = os.path.join(
            SHAP_DIR,
            f"dependence_{class_name.lower()}_{rank}_{safe_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

print("\n✓ All SHAP dependence plots generated.")

# ============================================================
# STEP 5 — SAVE TOP-5 SHAP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("SAVING TOP-5 SHAP FEATURES")
print("=" * 70)

top_features_records = []

for class_index, class_name in enumerate(class_names):

    class_shap = shap_features[:, class_index, :]

    # Mean absolute SHAP importance
    importance = np.mean(
        np.abs(class_shap),
        axis=0
    )

    # Top 5 features
    top_indices = np.argsort(importance)[-5:][::-1]

    for rank, feature_index in enumerate(top_indices, start=1):

        top_features_records.append({
            "Class": class_name,
            "Rank": rank,
            "Feature": feature_names[feature_index],
            "Mean_Absolute_SHAP": importance[feature_index]
        })

top_features_df = pd.DataFrame(top_features_records)

csv_path = os.path.join(
    RESULTS_DIR,
    "top5_shap_features.csv"
)

top_features_df.to_csv(
    csv_path,
    index=False
)

print("\n✓ Top-5 SHAP feature results saved:")
print(csv_path)

print("\nTop-5 SHAP Features:")
print(top_features_df.to_string(index=False))

# ============================================================
# STEP 6 — 5-FOLD SHAP CONSISTENCY ANALYSIS
# ============================================================

from sklearn.model_selection import StratifiedKFold

print("\n" + "=" * 70)
print("5-FOLD SHAP CONSISTENCY ANALYSIS")
print("=" * 70)

# Five-fold stratified cross-validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_records = []

for fold_number, (train_idx, val_idx) in enumerate(
    skf.split(X_train_fusion, y_train),
    start=1
):

    print(f"\nProcessing Fold {fold_number}/5...")

    X_fold_train = X_train_fusion[train_idx]
    X_fold_val = X_train_fusion[val_idx]

    y_fold_train = y_train[train_idx]

    # Train a fresh CatBoost model for this fold
    fold_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False
    )

    fold_model.fit(
        X_fold_train,
        y_fold_train
    )

    # Calculate SHAP values for validation fold
    fold_pool = Pool(X_fold_val)

    fold_shap = fold_model.get_feature_importance(
        data=fold_pool,
        type=EFstrType.ShapValues
    )

    fold_shap = np.asarray(fold_shap)

    # Remove expected-value column
    fold_shap_features = fold_shap[:, :, :-1]

    # Mean absolute SHAP across samples AND classes
    fold_importance = np.mean(
        np.abs(fold_shap_features),
        axis=(0, 1)
    )

    # Top 5 features
    top_indices = np.argsort(
        fold_importance
    )[-5:][::-1]

    print(f"Top features — Fold {fold_number}:")

    for rank, feature_index in enumerate(
        top_indices,
        start=1
    ):

        feature_name = feature_names[feature_index]
        importance_value = fold_importance[feature_index]

        print(
            f"{rank}. {feature_name} "
            f"(mean |SHAP| = {importance_value:.6f})"
        )

        fold_records.append({
            "Fold": fold_number,
            "Rank": rank,
            "Feature": feature_name,
            "Mean_Absolute_SHAP": importance_value
        })


# ------------------------------------------------------------
# Save fold-level results
# ------------------------------------------------------------

fold_results_df = pd.DataFrame(
    fold_records
)

fold_csv_path = os.path.join(
    RESULTS_DIR,
    "shap_fold_top5.csv"
)

fold_results_df.to_csv(
    fold_csv_path,
    index=False
)

print(
    f"\n✓ Fold-level SHAP results saved:\n"
    f"{fold_csv_path}"
)


# ------------------------------------------------------------
# Calculate feature stability
# ------------------------------------------------------------

stability_df = (
    fold_results_df
    .groupby("Feature")
    .agg(
        Folds_in_Top5=("Fold", "nunique"),
        Average_Rank=("Rank", "mean"),
        Average_SHAP=("Mean_Absolute_SHAP", "mean")
    )
    .reset_index()
    .sort_values(
        ["Folds_in_Top5", "Average_SHAP"],
        ascending=[False, False]
    )
)

stability_csv_path = os.path.join(
    RESULTS_DIR,
    "shap_feature_stability.csv"
)

stability_df.to_csv(
    stability_csv_path,
    index=False
)

print(
    f"✓ Feature stability results saved:\n"
    f"{stability_csv_path}"
)

print("\nFeature Stability:")
print(
    stability_df.to_string(index=False)
)