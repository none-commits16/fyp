import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SHAP VS INTEGRATED GRADIENTS
# EXPLAINABILITY COMPARISON
# ============================================================

OUTPUT_DIR = "outputs/explainability"

IG_DIR = os.path.join(
    OUTPUT_DIR,
    "integrated_gradients_by_class"
)

COMPARISON_DIR = os.path.join(
    OUTPUT_DIR,
    "shap_vs_ig"
)

os.makedirs(
    COMPARISON_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD IG RESULTS
# ============================================================

ig_file = os.path.join(
    IG_DIR,
    "ig_class_specific_mean.csv"
)

ig = pd.read_csv(
    ig_file
)

print(
    "\nLoaded Integrated Gradients results."
)

print(
    "Rows:",
    len(ig)
)


# ============================================================
# 2. FIND TOP IG MINUTES FOR EACH CLASS
# ============================================================

top_rows = []

for class_id, class_name in [
    (0, "Healthy"),
    (1, "Depression"),
    (2, "Schizophrenia")
]:

    class_data = ig[
        ig["true_class"] == class_id
    ].copy()

    class_data = class_data.sort_values(
        "absolute_importance",
        ascending=False
    )

    top = class_data.head(10)

    for _, row in top.iterrows():

        top_rows.append({

            "class": class_name,

            "minute":
                int(row["minute"]),

            "ig_importance":
                float(
                    row["absolute_importance"]
                )

        })


top_ig = pd.DataFrame(
    top_rows
)


# ============================================================
# 3. SAVE TOP IG RESULTS
# ============================================================

top_ig_path = os.path.join(
    COMPARISON_DIR,
    "top_ig_minutes.csv"
)

top_ig.to_csv(
    top_ig_path,
    index=False
)

print(
    "\nSaved:",
    top_ig_path
)


# ============================================================
# 4. CREATE EXPLAINABILITY METHOD TABLE
# ============================================================

comparison = pd.DataFrame({

    "Aspect": [

        "Explainability method",

        "Primary input",

        "Model type",

        "Explanation unit",

        "Main purpose",

        "Output interpretation",

        "Strength",

        "Limitation"

    ],

    "SHAP": [

        "SHAP",

        "Tabular features",

        "Tree-based models",

        "Feature",

        "Identify influential survey/activity features",

        "Contribution of each feature to prediction",

        "Global and local feature-level interpretation",

        "Does not directly explain minute-level CNN patterns"

    ],

    "Integrated Gradients": [

        "Integrated Gradients",

        "1440-minute activity sequence",

        "1D CNN",

        "Individual time point",

        "Identify influential portions of daily activity",

        "Attribution of activity observations to target class",

        "Captures deep-learning sequence behaviour",

        "Sensitive to baseline and model/input representation"

    ]

})


# ============================================================
# 5. SAVE COMPARISON TABLE
# ============================================================

comparison_path = os.path.join(
    COMPARISON_DIR,
    "shap_vs_integrated_gradients.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print(
    "Saved:",
    comparison_path
)


# ============================================================
# 6. CREATE CLASS SUMMARY
# ============================================================

class_summary = []

for class_name in [
    "Healthy",
    "Depression",
    "Schizophrenia"
]:

    data = top_ig[
        top_ig["class"] == class_name
    ]

    if len(data) == 0:
        continue

    class_summary.append({

        "class":
            class_name,

        "strongest_minute":
            int(
                data.iloc[0]["minute"]
            ),

        "strongest_ig":
            float(
                data.iloc[0]["ig_importance"]
            ),

        "top_10_mean_ig":
            float(
                data["ig_importance"].mean()
            )

    })


class_summary_df = pd.DataFrame(
    class_summary
)


class_summary_path = os.path.join(
    COMPARISON_DIR,
    "class_ig_summary.csv"
)

class_summary_df.to_csv(
    class_summary_path,
    index=False
)

print(
    "Saved:",
    class_summary_path
)


# ============================================================
# 7. PLOT CLASS COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    class_summary_df["class"],
    class_summary_df["top_10_mean_ig"]
)

plt.xlabel(
    "Target class"
)

plt.ylabel(
    "Mean attribution of top 10 minutes"
)

plt.title(
    "Integrated Gradients Attribution by Class"
)

plt.tight_layout()


plot_path = os.path.join(
    COMPARISON_DIR,
    "ig_class_comparison.png"
)

plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    plot_path
)


# ============================================================
# 8. PRINT FINAL SUMMARY
# ============================================================

print(
    "\n=============================================="
)

print(
    "SHAP VS INTEGRATED GRADIENTS"
)

print(
    "=============================================="
)

print(
    "\nMethod comparison:"
)

print(
    comparison.to_string(
        index=False
    )
)

print(
    "\nClass-specific IG summary:"
)

print(
    class_summary_df.to_string(
        index=False
    )
)

print(
    "\n=============================================="
)

print(
    "EXPLAINABILITY COMPARISON COMPLETE"
)

print(
    "=============================================="
)