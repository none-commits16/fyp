# ============================================================
# FINAL RESULTS SUMMARY
# ============================================================
#
# Creates:
#   outputs/final_results/
#       final_results_table.csv
#       final_results_table.xlsx
#       final_progression.csv
#       final_results_comparison.png
#
# Uses ONLY verified project results.
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl.styles import Font


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "outputs/final_results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# VERIFIED RESULTS
# ============================================================

results = [

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    {
        "Stage": "Baseline",
        "Method": "Random Forest",
        "Accuracy": 74.3290,
        "Accuracy_SD": None,
        "F1": 70.4332,
        "F1_SD": None,
        "AUC": 86.0570,
        "AUC_SD": None,
        "Notes": "18 handcrafted baseline features"
    },

    {
        "Stage": "Baseline",
        "Method": "CatBoost",
        "Accuracy": 75.2381,
        "Accuracy_SD": None,
        "F1": 71.3531,
        "F1_SD": None,
        "AUC": 84.6935,
        "AUC_SD": None,
        "Notes": "18 handcrafted baseline features"
    },

    {
        "Stage": "Baseline",
        "Method": "LightGBM",
        "Accuracy": 77.0996,
        "Accuracy_SD": None,
        "F1": 73.6784,
        "F1_SD": None,
        "AUC": 88.4367,
        "AUC_SD": None,
        "Notes": "18 handcrafted baseline features"
    },

    # --------------------------------------------------------
    # IMBALANCE HANDLING
    # --------------------------------------------------------

    {
        "Stage": "Imbalance Handling",
        "Method": "Random Forest + ADASYN",
        "Accuracy": 76.1472,
        "Accuracy_SD": 13.0063,
        "F1": 73.9429,
        "F1_SD": 14.0359,
        "AUC": 86.8269,
        "AUC_SD": 11.7401,
        "Notes": "Class imbalance correction"
    },

    {
        "Stage": "Imbalance Handling",
        "Method": "CatBoost + ADASYN",
        "Accuracy": 74.2424,
        "Accuracy_SD": 14.0509,
        "F1": 72.6926,
        "F1_SD": 15.9486,
        "AUC": 85.7740,
        "AUC_SD": 12.0814,
        "Notes": "Class imbalance correction"
    },

    {
        "Stage": "Imbalance Handling",
        "Method": "LightGBM + ADASYN",
        "Accuracy": 78.9177,
        "Accuracy_SD": 12.2391,
        "F1": 76.5522,
        "F1_SD": 14.9720,
        "AUC": 89.0820,
        "AUC_SD": 10.2524,
        "Notes": "Class imbalance correction"
    },

    {
        "Stage": "Imbalance Handling",
        "Method": "Random Forest + Borderline-SMOTE",
        "Accuracy": 73.3766,
        "Accuracy_SD": 13.0518,
        "F1": 68.7015,
        "F1_SD": 15.1998,
        "AUC": 85.0678,
        "AUC_SD": 12.2682,
        "Notes": "Class imbalance correction"
    },

    {
        "Stage": "Imbalance Handling",
        "Method": "CatBoost + Borderline-SMOTE",
        "Accuracy": 76.1039,
        "Accuracy_SD": 9.9903,
        "F1": 73.6632,
        "F1_SD": 11.5060,
        "AUC": 83.5999,
        "AUC_SD": 12.2622,
        "Notes": "Class imbalance correction"
    },

    {
        "Stage": "Imbalance Handling",
        "Method": "LightGBM + Borderline-SMOTE",
        "Accuracy": 79.8268,
        "Accuracy_SD": 8.2186,
        "F1": 77.8625,
        "F1_SD": 10.2451,
        "AUC": 91.1158,
        "AUC_SD": 9.3908,
        "Notes": "Class imbalance correction"
    },

    {
        "Stage": "Imbalance Handling",
        "Method": "Random Forest + Class Weight",
        "Accuracy": 74.3290,
        "Accuracy_SD": 6.8029,
        "F1": 69.6527,
        "F1_SD": 9.8961,
        "AUC": 85.5324,
        "AUC_SD": 11.4279,
        "Notes": "Class imbalance correction"
    },

    {
        "Stage": "Imbalance Handling",
        "Method": "CatBoost + Class Weight",
        "Accuracy": 76.1472,
        "Accuracy_SD": 8.1153,
        "F1": 73.8251,
        "F1_SD": 9.1089,
        "AUC": 83.6237,
        "AUC_SD": 12.7058,
        "Notes": "Class imbalance correction"
    },

    {
        "Stage": "Imbalance Handling",
        "Method": "LightGBM + Class Weight",
        "Accuracy": 76.1039,
        "Accuracy_SD": 10.4946,
        "F1": 73.0149,
        "F1_SD": 13.6338,
        "AUC": 87.3622,
        "AUC_SD": 8.0491,
        "Notes": "Class imbalance correction"
    },

    # --------------------------------------------------------
    # FEATURE ENHANCEMENT
    # --------------------------------------------------------

    {
        "Stage": "Feature Enhancement",
        "Method": "Random Forest + Enhanced 27 Features",
        "Accuracy": 75.2381,
        "Accuracy_SD": None,
        "F1": 71.0411,
        "F1_SD": None,
        "AUC": 89.8118,
        "AUC_SD": None,
        "Notes": "IS, IV, RA, time segments, Sample Entropy, LZ"
    },

    {
        "Stage": "Feature Enhancement",
        "Method": "CatBoost + Enhanced 27 Features",
        "Accuracy": 76.1472,
        "Accuracy_SD": None,
        "F1": 72.7222,
        "F1_SD": None,
        "AUC": 89.3523,
        "AUC_SD": None,
        "Notes": "IS, IV, RA, time segments, Sample Entropy, LZ"
    },

    {
        "Stage": "Feature Enhancement",
        "Method": "LightGBM + Enhanced 27 Features",
        "Accuracy": 80.6926,
        "Accuracy_SD": None,
        "F1": 77.7791,
        "F1_SD": None,
        "AUC": 92.6513,
        "AUC_SD": None,
        "Notes": "IS, IV, RA, time segments, Sample Entropy, LZ"
    },

    {
        "Stage": "Feature Enhancement",
        "Method": "Random Forest + Enhanced 27 + Borderline-SMOTE",
        "Accuracy": 73.4199,
        "Accuracy_SD": None,
        "F1": 70.9160,
        "F1_SD": None,
        "AUC": 87.9927,
        "AUC_SD": None,
        "Notes": "Enhanced features + imbalance correction"
    },

    {
        "Stage": "Feature Enhancement",
        "Method": "CatBoost + Enhanced 27 + Borderline-SMOTE",
        "Accuracy": 73.4199,
        "Accuracy_SD": None,
        "F1": 70.7316,
        "F1_SD": None,
        "AUC": 87.4554,
        "AUC_SD": None,
        "Notes": "Enhanced features + imbalance correction"
    },

    {
        "Stage": "Feature Enhancement",
        "Method": "LightGBM + Enhanced 27 + Borderline-SMOTE",
        "Accuracy": 74.2857,
        "Accuracy_SD": None,
        "F1": 72.8429,
        "F1_SD": None,
        "AUC": 90.1841,
        "AUC_SD": None,
        "Notes": "Enhanced features + imbalance correction"
    },

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    {
        "Stage": "Hybrid",
        "Method": "1D-CNN",
        "Accuracy": 78.87,
        "Accuracy_SD": 9.49,
        "F1": 75.28,
        "F1_SD": 12.93,
        "AUC": 90.93,
        "AUC_SD": 5.98,
        "Notes": "Raw 15-minute activity sequences"
    },

    {
        "Stage": "Hybrid",
        "Method": "CNN + 27 Features + LightGBM",
        "Accuracy": 79.83,
        "Accuracy_SD": 10.43,
        "F1": 77.11,
        "F1_SD": 9.93,
        "AUC": 89.56,
        "AUC_SD": 7.43,
        "Notes": "CNN embedding + handcrafted features"
    },

    {
        "Stage": "Hybrid - Final",
        "Method": "Full Nested Stacking",
        "Accuracy": 80.69,
        "Accuracy_SD": 7.61,
        "F1": 78.49,
        "F1_SD": 8.99,
        "AUC": 92.54,
        "AUC_SD": 7.51,
        "Notes": (
            "RF + CatBoost + LightGBM + CNN-fusion "
            "+ Logistic Regression meta-learner"
        )
    }
]


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# FORMAT RESULTS
# ============================================================

def format_mean_sd(mean, sd):

    if pd.isna(sd):
        return f"{mean:.2f}%"

    return f"{mean:.2f}% ± {sd:.2f}%"


df["Accuracy_Result"] = df.apply(
    lambda row: format_mean_sd(
        row["Accuracy"],
        row["Accuracy_SD"]
    ),
    axis=1
)

df["F1_Result"] = df.apply(
    lambda row: format_mean_sd(
        row["F1"],
        row["F1_SD"]
    ),
    axis=1
)

df["AUC_Result"] = df.apply(
    lambda row: format_mean_sd(
        row["AUC"],
        row["AUC_SD"]
    ),
    axis=1
)


# ============================================================
# SAVE DETAILED CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "final_results_table.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# REPORT-FRIENDLY DATAFRAME
# ============================================================

excel_df = df[
    [
        "Stage",
        "Method",
        "Accuracy_Result",
        "F1_Result",
        "AUC_Result",
        "Notes"
    ]
].copy()

excel_df.columns = [
    "Stage",
    "Method",
    "Accuracy",
    "F1",
    "AUC",
    "Notes"
]


# ============================================================
# SAVE EXCEL
# ============================================================

xlsx_path = os.path.join(
    OUTPUT_DIR,
    "final_results_table.xlsx"
)

with pd.ExcelWriter(
    xlsx_path,
    engine="openpyxl"
) as writer:

    excel_df.to_excel(
        writer,
        sheet_name="Results",
        index=False
    )

    # IMPORTANT:
    # Access worksheet through writer.book
    # rather than writer["Results"].

    worksheet = writer.book["Results"]

    worksheet.freeze_panes = "A2"

    worksheet.auto_filter.ref = (
        worksheet.dimensions
    )

    # Column widths

    widths = {
        "A": 24,
        "B": 50,
        "C": 23,
        "D": 23,
        "E": 23,
        "F": 65
    }

    for column, width in widths.items():

        worksheet.column_dimensions[
            column
        ].width = width

    # Bold header

    for cell in worksheet[1]:

        cell.font = Font(
            bold=True
        )


# ============================================================
# FINAL MODEL PROGRESSION
# ============================================================

progression = pd.DataFrame(
    [
        {
            "Stage": "Baseline",
            "Method": "LightGBM",
            "Accuracy": 77.0996,
            "F1": 73.6784,
            "AUC": 88.4367
        },

        {
            "Stage": "Imbalance Handling",
            "Method": "LightGBM + Borderline-SMOTE",
            "Accuracy": 79.8268,
            "F1": 77.8625,
            "AUC": 91.1158
        },

        {
            "Stage": "Feature Enhancement",
            "Method": "LightGBM + Enhanced 27 Features",
            "Accuracy": 80.6926,
            "F1": 77.7791,
            "AUC": 92.6513
        },

        {
            "Stage": "Hybrid A",
            "Method": "1D-CNN",
            "Accuracy": 78.87,
            "F1": 75.28,
            "AUC": 90.93
        },

        {
            "Stage": "Hybrid B",
            "Method": "CNN + Features + LightGBM",
            "Accuracy": 79.83,
            "F1": 77.11,
            "AUC": 89.56
        },

        {
            "Stage": "Hybrid C - Final",
            "Method": "Full Nested Stacking",
            "Accuracy": 80.69,
            "F1": 78.49,
            "AUC": 92.54
        }
    ]
)


# ============================================================
# SAVE PROGRESSION CSV
# ============================================================

progression_path = os.path.join(
    OUTPUT_DIR,
    "final_progression.csv"
)

progression.to_csv(
    progression_path,
    index=False
)


# ============================================================
# CREATE COMPARISON PLOT
# ============================================================

stages = progression["Stage"].values

accuracy = progression["Accuracy"].values
f1 = progression["F1"].values
auc = progression["AUC"].values

x = list(range(len(stages)))

width = 0.25

x_accuracy = [
    i - width
    for i in x
]

x_f1 = x

x_auc = [
    i + width
    for i in x
]


fig, ax = plt.subplots(
    figsize=(14, 8)
)


ax.bar(
    x_accuracy,
    accuracy,
    width=width,
    label="Accuracy"
)

ax.bar(
    x_f1,
    f1,
    width=width,
    label="F1"
)

ax.bar(
    x_auc,
    auc,
    width=width,
    label="AUC"
)


# ============================================================
# PLOT LABELS
# ============================================================

ax.set_ylabel(
    "Score (%)",
    fontsize=12
)

ax.set_xlabel(
    "Development Stage",
    fontsize=12
)

ax.set_title(
    "Model Performance Progression",
    fontsize=15
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    stages,
    rotation=20,
    ha="right"
)

ax.set_ylim(
    60,
    100
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


# ============================================================
# ADD VALUES
# ============================================================

for positions, values in [
    (x_accuracy, accuracy),
    (x_f1, f1),
    (x_auc, auc)
]:

    for position, value in zip(
        positions,
        values
    ):

        ax.text(
            position,
            value + 0.7,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8
        )


plt.tight_layout()


plot_path = os.path.join(
    OUTPUT_DIR,
    "final_results_comparison.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRINT DETAILED RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("FINAL RESULTS PACKAGE CREATED")
print("=" * 80)

print("\nDETAILED RESULTS")
print("-" * 80)

print(
    excel_df.to_string(
        index=False
    )
)


# ============================================================
# PRINT PROGRESSION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL PROGRESSION")
print("=" * 80)

print(
    progression.to_string(
        index=False
    )
)


# ============================================================
# PRINT FILE LOCATIONS
# ============================================================

print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"\nCSV:"
    f"\n{csv_path}"
)

print(
    f"\nExcel:"
    f"\n{xlsx_path}"
)

print(
    f"\nProgression CSV:"
    f"\n{progression_path}"
)

print(
    f"\nComparison plot:"
    f"\n{plot_path}"
)


# ============================================================
# FINAL MODEL
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL")
print("=" * 80)

print(
    "\nFull Nested Stacking"
)

print(
    "Accuracy: 80.69% ± 7.61%"
)

print(
    "F1:       78.49% ± 8.99%"
)

print(
    "AUC:      92.54% ± 7.51%"
)


# ============================================================
# OVERALL OOF RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("OVERALL STRICT OOF RESULTS")
print("=" * 80)

print(
    "\nAccuracy:  80.73%"
)

print(
    "Precision: 79.69%"
)

print(
    "Recall:    80.73%"
)

print(
    "F1:        78.68%"
)

print(
    "AUC:       89.50%"
)


# ============================================================
# END
# ============================================================

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)