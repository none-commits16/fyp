import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# INTEGRATED GRADIENTS — MULTIPLE REAL ACTIGRAPHY SEQUENCES
# ============================================================

MODEL_PATH = "outputs/deep_learning/models/cnn_fold_1.keras"

OUTPUT_DIR = "outputs/explainability/integrated_gradients_multi"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 1. DATASET PATHS
# ============================================================

DEPRESJON_DIR = "data/depresjon"
PSYKOSE_DIR = "data/psykose"


# ============================================================
# 2. FIND ALL ACTIGRAPHY FILES
# ============================================================

def get_all_files():

    files = []

    files.extend(
        glob.glob(
            os.path.join(
                DEPRESJON_DIR,
                "condition",
                "*.csv"
            )
        )
    )

    files.extend(
        glob.glob(
            os.path.join(
                DEPRESJON_DIR,
                "control",
                "*.csv"
            )
        )
    )

    files.extend(
        glob.glob(
            os.path.join(
                PSYKOSE_DIR,
                "patient",
                "*.csv"
            )
        )
    )

    files.extend(
        glob.glob(
            os.path.join(
                PSYKOSE_DIR,
                "control",
                "*.csv"
            )
        )
    )

    return files


# ============================================================
# 3. EXTRACT 1440-MINUTE SEQUENCES
# ============================================================

def extract_sequences(file_path):

    df = pd.read_csv(
        file_path
    )

    if "timestamp" not in df.columns:
        return []

    if "activity" not in df.columns:
        return []

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    df["date"] = df["timestamp"].dt.date

    sequences = []

    for date, group in df.groupby("date"):

        activity = group[
            "activity"
        ].values.astype(
            np.float32
        )

        if len(activity) == 1440:

            sequences.append(
                {
                    "activity": activity,
                    "date": date,
                    "file": file_path
                }
            )

    return sequences


# ============================================================
# 4. NORMALIZATION
# ============================================================

def normalize_sequence(data):

    data = np.asarray(
        data,
        dtype=np.float32
    )

    mean = np.mean(
        data,
        axis=0,
        keepdims=True
    )

    std = np.std(
        data,
        axis=0,
        keepdims=True
    )

    return (
        data - mean
    ) / (
        std + 1e-8
    )


# ============================================================
# 5. INTEGRATED GRADIENTS
# ============================================================

def integrated_gradients(
    model,
    x,
    target_class,
    baseline=None,
    steps=50
):

    x = tf.cast(
        x,
        tf.float32
    )

    if baseline is None:

        baseline = tf.zeros_like(
            x
        )

    baseline = tf.cast(
        baseline,
        tf.float32
    )

    alphas = tf.linspace(
        0.0,
        1.0,
        steps + 1
    )

    interpolated = (
        baseline
        +
        alphas[:, None, None]
        *
        (x - baseline)
    )

    with tf.GradientTape() as tape:

        tape.watch(
            interpolated
        )

        predictions = model(
            interpolated,
            training=False
        )

        target_output = predictions[
            :,
            target_class
        ]

    gradients = tape.gradient(
        target_output,
        interpolated
    )

    average_gradients = tf.reduce_mean(
        (
            gradients[:-1]
            +
            gradients[1:]
        ) / 2.0,
        axis=0
    )

    attributions = (
        x - baseline
    ) * average_gradients

    return attributions.numpy()


# ============================================================
# 6. LOAD MODEL
# ============================================================

print(
    "\nLoading CNN model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "CNN loaded successfully."
)

print(
    "Input shape:",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)


# ============================================================
# 7. LOAD REAL SEQUENCES
# ============================================================

files = get_all_files()

print(
    f"\nFound {len(files)} actigraphy files."
)

all_sequences = []

for file_path in files:

    sequences = extract_sequences(
        file_path
    )

    all_sequences.extend(
        sequences
    )

print(
    f"Found {len(all_sequences)} complete 1440-minute sequences."
)


# ============================================================
# 8. LIMIT NUMBER OF SEQUENCES
# ============================================================

# Start with 10 real sequences.
# We can increase this later after confirming everything works.

MAX_SEQUENCES = 1000

selected_sequences = (
    all_sequences[:MAX_SEQUENCES]
)

print(
    f"Using {len(selected_sequences)} sequences for IG."
)


# ============================================================
# 9. CALCULATE IG FOR EACH SEQUENCE
# ============================================================

all_results = []

for sequence_number, item in enumerate(
    selected_sequences,
    start=1
):

    print(
        f"\nProcessing sequence "
        f"{sequence_number}/{len(selected_sequences)}"
    )

    activity = item[
        "activity"
    ]

    normalized_activity = (
        normalize_sequence(
            activity
        )
    )

    sample = normalized_activity.reshape(
        1,
        1440,
        1
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        sample,
        verbose=0
    )

    predicted_class = int(
        np.argmax(
            prediction[0]
        )
    )

    print(
        "Predicted class:",
        predicted_class
    )

    # --------------------------------------------------------
    # Integrated Gradients
    # --------------------------------------------------------

    ig = integrated_gradients(
        model=model,
        x=sample,
        target_class=predicted_class,
        steps=50
    )

    importance = np.abs(
        ig[0, :, 0]
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    result = pd.DataFrame({

        "sequence": sequence_number,

        "file": item["file"],

        "date": str(item["date"]),

        "minute": np.arange(
            1440
        ),

        "activity": activity,

        "normalized_activity":
            normalized_activity,

        "integrated_gradient":
            ig[0, :, 0],

        "absolute_importance":
            importance,

        "predicted_class":
            predicted_class

    })

    all_results.append(
        result
    )


# ============================================================
# 10. COMBINE ALL RESULTS
# ============================================================

combined = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# 11. SAVE INDIVIDUAL RESULTS
# ============================================================

combined_path = os.path.join(
    OUTPUT_DIR,
    "ig_all_sequences.csv"
)

combined.to_csv(
    combined_path,
    index=False
)

print(
    "\nSaved:",
    combined_path
)


# ============================================================
# 12. AGGREGATE IMPORTANCE BY MINUTE
# ============================================================

minute_summary = (
    combined
    .groupby("minute")
    ["absolute_importance"]
    .mean()
    .reset_index()
)

minute_summary = (
    minute_summary
    .sort_values(
        "absolute_importance",
        ascending=False
    )
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "ig_mean_importance_by_minute.csv"
)

minute_summary.to_csv(
    summary_path,
    index=False
)

print(
    "Saved:",
    summary_path
)


# ============================================================
# 13. TOP 20 MINUTES ACROSS SEQUENCES
# ============================================================

top20 = (
    minute_summary
    .head(20)
)

top20_path = os.path.join(
    OUTPUT_DIR,
    "ig_top20_minutes_aggregate.csv"
)

top20.to_csv(
    top20_path,
    index=False
)

print(
    "\nTop 20 important minutes across sequences:"
)

print(
    top20.to_string(
        index=False
    )
)


# ============================================================
# 14. PLOT AGGREGATED IMPORTANCE
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    minute_summary["minute"],
    minute_summary["absolute_importance"]
)

plt.xlabel(
    "Minute of Day"
)

plt.ylabel(
    "Mean |Integrated Gradients|"
)

plt.title(
    "Aggregated Integrated Gradients Across Real Actigraphy Sequences"
)

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "ig_aggregate_importance.png"
)

plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved:",
    plot_path
)


# ============================================================
# 15. FINISHED
# ============================================================

print(
    "\n=============================================="
)

print(
    "MULTI-SEQUENCE INTEGRATED GRADIENTS COMPLETE"
)

print(
    "=============================================="
)