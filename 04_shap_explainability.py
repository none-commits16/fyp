import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import warnings
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")

os.makedirs("outputs/shap", exist_ok=True)

splits  = joblib.load("outputs/models/splits.pkl")
FEATURES = joblib.load("outputs/models/features.pkl")

FEATURE_LABELS = [
    "Age", "Smartphone Hours", "Social Media Hours",
    "Study Proportion", "Sleep Hours", "Living Arrangement",
    "Meditation Frequency", "Academic Pressure"
]

for task_name in ["severity", "anxiety", "depression"]:
    print(f"\nGenerating SHAP for task: {task_name}")
    X_tr, X_te, y_tr, y_te = splits[task_name]
    imputer = SimpleImputer(strategy='mean')

    X_tr = imputer.fit_transform(X_tr)
    X_te = imputer.transform(X_te)
    model = joblib.load(f"outputs/models/best_{task_name}.pkl")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_te)

    # For multiclass, take mean abs SHAP across classes
    shap_mean = np.abs(shap_values.values)

# Handle multiclass SHAP output
    if len(shap_mean.shape) == 3:
     shap_mean = shap_mean.mean(axis=2)

    # --- Summary bar plot ---
    plt.figure(figsize=(8, 5))
    mean_importance = np.mean(shap_mean, axis=0)
    sorted_idx = np.argsort(mean_importance)[::-1]
    plt.barh(
        [FEATURE_LABELS[i] for i in sorted_idx[::-1]],
        mean_importance[sorted_idx[::-1]],
        color="#5C6BC0", edgecolor="white"
    )
    plt.title(f"SHAP Feature Importance — {task_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(f"outputs/shap/shap_bar_{task_name}.png", dpi=150)
    plt.show()

    # --- SHAP beeswarm (summary plot) ---
    plt.figure(figsize=(9, 6))

    shap.summary_plot(
    shap_values.values,
    X_te,
    feature_names=FEATURE_LABELS,
    show=False
)
    plt.title(f"SHAP Summary Plot — {task_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/shap/shap_summary_{task_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  SHAP plots saved for {task_name}")

print("\nAll SHAP outputs saved to outputs/shap/")