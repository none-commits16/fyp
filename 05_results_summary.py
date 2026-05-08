import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

results = pd.read_csv("outputs/results/all_results.csv")

print("\n" + "="*65)
print("  FINAL RESULTS TABLE (for your paper)")
print("="*65)

for task in results["Task"].unique():
    print(f"\n── Task: {task} ──")
    sub = results[results["Task"] == task][["Model", "Accuracy", "F1 (weighted)", "AUC-ROC"]]
    sub = sub.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    print(sub.to_string(index=False))

# --- Publication-style comparison table plot ---
fig, ax = plt.subplots(figsize=(13, 5))
ax.axis("off")

best_per_task = results.loc[results.groupby("Task")["Accuracy"].idxmax()]
best_per_task = best_per_task.reset_index(drop=True)

table_data = best_per_task[["Task", "Model", "Accuracy", "F1 (weighted)", "AUC-ROC"]].values
col_labels = ["Task", "Best Model", "Accuracy (%)", "F1 Score", "AUC-ROC"]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3, 2.0)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#3F51B5")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(table_data) + 1):
    color = "#E8EAF6" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(color)

ax.set_title("Best Model per Task — Mental Health Monitoring",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("outputs/results/paper_results_table.png", dpi=200, bbox_inches="tight")
plt.show()
print("\nPaper-ready table saved to outputs/results/paper_results_table.png")