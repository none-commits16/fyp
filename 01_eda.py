import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/eda", exist_ok=True)

df = pd.read_csv("data/data_collection.csv")

# Rename columns to short names
df.columns = [
    "timestamp", "age", "smartphone_hours", "social_media_hours",
    "study_proportion", "sleep_hours", "living_arrangement",
    "meditation_freq", "academic_pressure",
    "phq4_q1_anxious", "phq4_q2_worrying",
    "phq4_q3_interest", "phq4_q4_hopeless"
]

# Compute PHQ-4 subscores
df["GAD2"] = df["phq4_q1_anxious"] + df["phq4_q2_worrying"]
df["PHQ2"] = df["phq4_q3_interest"] + df["phq4_q4_hopeless"]
df["PHQ4_total"] = df["GAD2"] + df["PHQ2"]

# Severity label (standard PHQ-4 cutoffs)
def get_severity(score):
    if score <= 3: return "Normal"
    elif score <= 5: return "Mild"
    elif score <= 8: return "Moderate"
    else: return "Severe"

df["severity"] = df["PHQ4_total"].apply(get_severity)
severity_order = ["Normal", "Mild", "Moderate", "Severe"]

print("Dataset shape:", df.shape)
print("\nSeverity distribution:")
print(df["severity"].value_counts())
print("\nMissing values:", df.isnull().sum().sum())

# --- Plot 1: PHQ-4 severity distribution ---
fig, ax = plt.subplots(figsize=(7, 4))
counts = df["severity"].value_counts()[severity_order]
colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
ax.bar(severity_order, counts.values, color=colors, edgecolor="white", linewidth=0.8)
ax.set_title("PHQ-4 Severity Distribution (n=1823)", fontsize=13, fontweight="bold")
ax.set_xlabel("Severity Level")
ax.set_ylabel("Number of Respondents")
for i, v in enumerate(counts.values):
    ax.text(i, v + 10, str(v), ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("outputs/eda/severity_distribution.png", dpi=150)
plt.show()
print("Saved: severity_distribution.png")

# --- Plot 2: PHQ-4 total score histogram ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["PHQ4_total"], bins=13, color="#5C6BC0", edgecolor="white", linewidth=0.8)
ax.axvline(3.5, color="#4CAF50", linestyle="--", label="Normal/Mild (3)")
ax.axvline(5.5, color="#FFC107", linestyle="--", label="Mild/Moderate (5)")
ax.axvline(8.5, color="#F44336", linestyle="--", label="Moderate/Severe (8)")
ax.set_title("PHQ-4 Total Score Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("PHQ-4 Total Score (0–12)")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda/phq4_histogram.png", dpi=150)
plt.show()

# --- Plot 3: Smartphone hours vs Severity ---
fig, ax = plt.subplots(figsize=(8, 4))
phone_order = ["2", "3", "4", "5", "6", "7", "8", "9", "10+"]
df["smartphone_hours"] = df["smartphone_hours"].astype(str)
mean_phq4 = df.groupby("smartphone_hours")["PHQ4_total"].mean().reindex(phone_order)
ax.bar(phone_order, mean_phq4.values, color="#42A5F5", edgecolor="white")
ax.set_title("Avg PHQ-4 Score by Daily Smartphone Usage", fontsize=13, fontweight="bold")
ax.set_xlabel("Smartphone Hours per Day")
ax.set_ylabel("Mean PHQ-4 Total Score")
plt.tight_layout()
plt.savefig("outputs/eda/smartphone_vs_phq4.png", dpi=150)
plt.show()

# --- Plot 4: Sleep vs Severity ---
fig, ax = plt.subplots(figsize=(8, 4))
sleep_order = ["2", "3", "4", "5", "6", "7", "8", "10+"]
df["sleep_hours"] = df["sleep_hours"].astype(str).str.strip()
mean_sleep = df.groupby("sleep_hours")["PHQ4_total"].mean().reindex(sleep_order)
ax.bar(sleep_order, mean_sleep.values, color="#66BB6A", edgecolor="white")
ax.set_title("Avg PHQ-4 Score by Sleep Hours", fontsize=13, fontweight="bold")
ax.set_xlabel("Sleep Hours per Day")
ax.set_ylabel("Mean PHQ-4 Total Score")
plt.tight_layout()
plt.savefig("outputs/eda/sleep_vs_phq4.png", dpi=150)
plt.show()

# --- Plot 5: Correlation heatmap (numeric features) ---
numeric_cols = ["age", "academic_pressure", "GAD2", "PHQ2", "PHQ4_total",
                "phq4_q1_anxious", "phq4_q2_worrying", "phq4_q3_interest", "phq4_q4_hopeless"]
df_num = df[numeric_cols]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda/correlation_heatmap.png", dpi=150)
plt.show()

#data/eda.py --- Plot 6: Living arrangement vs PHQ4 ---
fig, ax = plt.subplots(figsize=(7, 4))
df["living_arrangement"] = df["living_arrangement"].str.strip()
living_means = df.groupby("living_arrangement")["PHQ4_total"].mean().sort_values(ascending=False)
ax.barh(living_means.index, living_means.values, color="#AB47BC", edgecolor="white")
ax.set_title("Avg PHQ-4 Score by Living Arrangement", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean PHQ-4 Score")
plt.tight_layout()
plt.savefig("outputs/eda/living_vs_phq4.png", dpi=150)
plt.show()

print("\nAll EDA plots saved to outputs/eda/")
df.to_csv("data/df_with_labels.csv", index=False)
print("Labelled dataframe saved to data/df_with_labels.csv")