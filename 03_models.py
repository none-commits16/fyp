import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, accuracy_score)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

os.makedirs("outputs/results", exist_ok=True)

splits   = joblib.load("outputs/models/splits.pkl")
FEATURES = joblib.load("outputs/models/features.pkl")

all_results = []

for task, (Xtr, Xte, ytr, yte) in splits.items():
    n_classes = len(np.unique(ytr))
    obj = "multiclass" if n_classes > 2 else "binary"
    print(f"\n{'='*50}\n  TASK: {task.upper()}\n{'='*50}")

    # SMOTE on training only
    Xr, yr = SMOTE(random_state=42).fit_resample(Xtr, ytr)

    models = {
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            objective=obj, n_jobs=-1, random_state=42, verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            random_seed=42, verbose=0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    # Fit individual models first, then ensemble
    fitted = {}
    for name, m in models.items():
        m.fit(Xr, yr)
        fitted[name] = m

    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=list(fitted.items()), voting="soft"
    )
    ensemble.fit(Xr, yr)
    fitted["Ensemble"] = ensemble

    best_f1, best_name, best_model = 0, "", None

    for name, model in fitted.items():
        yp   = model.predict(Xte)
        yprb = model.predict_proba(Xte)
        acc  = accuracy_score(yte, yp)
        f1   = f1_score(yte, yp, average="weighted")

        if n_classes == 2:
            auc = roc_auc_score(yte, yprb[:, 1])
        else:
            yte_b = label_binarize(yte, classes=sorted(np.unique(yte)))
            auc   = roc_auc_score(yte_b, yprb, average="weighted",
                                   multi_class="ovr")

        print(f"  [{name}]  Acc={acc*100:.1f}%  F1={f1:.3f}  AUC={auc:.3f}")
        print(classification_report(yte, yp, digits=3))

        all_results.append({"Task": task, "Model": name,
                             "Accuracy": round(acc*100, 2),
                             "F1": round(f1, 4), "AUC": round(auc, 4)})

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

        # Confusion matrix
        cm = confusion_matrix(yte, yp)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_title(f"{name} — {task}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"outputs/results/cm_{task}_{name.replace(' ','_')}.png", dpi=150)
        plt.close()

    joblib.dump(best_model, f"outputs/models/best_{task}.pkl")
    print(f"\n  ★ Best: {best_name}  F1={best_f1:.3f}")

# Summary table
rdf = pd.DataFrame(all_results)
print("\n" + "="*58)
print("  SUMMARY")
print("="*58)
print(rdf.to_string(index=False))
rdf.to_csv("outputs/results/all_results.csv", index=False)

# Plot
tasks = rdf["Task"].unique()
fig, axes = plt.subplots(1, len(tasks), figsize=(5*len(tasks), 5))
colors = ["#5C6BC0","#42A5F5","#66BB6A","#AB47BC"]
for i, task in enumerate(tasks):
    sub = rdf[rdf["Task"]==task].sort_values("Accuracy", ascending=False)
    axes[i].bar(range(len(sub)), sub["Accuracy"].values,
                color=colors[:len(sub)], edgecolor="white")
    axes[i].set_xticks(range(len(sub)))
    axes[i].set_xticklabels(sub["Model"].values, rotation=30, ha="right")
    axes[i].set_title(task, fontweight="bold")
    axes[i].set_ylim(40, 100)
    axes[i].set_ylabel("Accuracy (%)")
    for j, v in enumerate(sub["Accuracy"].values):
        axes[i].text(j, v+0.5, f"{v:.1f}", ha="center", fontsize=8)
plt.suptitle("Model Accuracy Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/results/accuracy_comparison.png", dpi=150)
plt.show()
print("\nAll done! Results in outputs/results/")