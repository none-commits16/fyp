import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# INTEGRATED GRADIENTS — REAL ACTIGRAPHY DATA
# ============================================================

MODEL_PATH = "outputs/deep_learning/models/cnn_fold_1.keras"

OUTPUT_DIR = "outputs/explainability/integrated_gradients"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. DATASET PATHS
# ============================================================

DEPRESJON_DIR = "data/depresjon"
PSYKOSE_DIR = "data/psykose"


# ============================================================
# 2. LOAD ONE REAL 1440-MINUTE ACTIVITY SEQUENCE
# ============================================================

def load_first_real_sequence():

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

    if len(files) == 0:
        raise FileNotFoundError(
            "No actigraphy CSV files found."
        )

    print(
        f"Found {len(files)} actigraphy files."
    )

    # --------------------------------------------------------
    # Find the first complete 1440-minute day
    # --------------------------------------------------------

    for file_path in files:

        print(
            f"Checking: {file_path}"
        )

        df = pd.read_csv(
            file_path
        )

        # Same basic columns used by the CNN pipeline
        if "timestamp" not in df.columns:
            continue

        if "activity" not in df.columns:
            continue

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        )

        df["date"] = df["timestamp"].dt.date

        for date, group in df.groupby("date"):

            activity = group[
                "activity"
            ].values.astype(
                np.float32
            )

            if len(activity) == 1440:

                print(
                    "\nFound complete 1440-minute sequence."
                )

                print(
                    "File:",
                    file_path
                )

                print(
                    "Date:",
                    date
                )

                return activity, file_path, date

    raise ValueError(
        "No complete 1440-minute sequence found."
    )


# ============================================================
# 3. NORMALIZE EXACTLY LIKE THE CNN PIPELINE
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
# 4. INTEGRATED GRADIENTS
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
# 5. LOAD CNN
# ============================================================

print(
    "\nLoading trained CNN..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully."
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
# 6. LOAD REAL DATA
# ============================================================

activity, source_file, source_date = (
    load_first_real_sequence()
)


print(
    "\nRaw sequence shape:",
    activity.shape
)


# ============================================================
# 7. NORMALIZE
# ============================================================

activity_normalized = normalize_sequence(
    activity
)

# CNN expects:
# (samples, 1440, 1)

sample = activity_normalized.reshape(
    1,
    1440,
    1
)


print(
    "CNN input shape:",
    sample.shape
)


# ============================================================
# 8. PREDICTION
# ============================================================

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
    "\nPrediction probabilities:"
)

print(
    prediction[0]
)

print(
    "Predicted class:",
    predicted_class
)


# ============================================================
# 9. CALCULATE IG
# ============================================================

print(
    "\nCalculating Integrated Gradients..."
)

ig = integrated_gradients(
    model=model,
    x=sample,
    target_class=predicted_class,
    steps=50
)

print(
    "Integrated Gradients calculated."
)

print(
    "IG shape:",
    ig.shape
)


# ============================================================
# 10. ABSOLUTE IMPORTANCE PER MINUTE
# ============================================================

importance = np.abs(
    ig[0, :, 0]
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "minute": np.arange(
        1440
    ),

    "activity": activity,

    "normalized_activity":
        activity_normalized,

    "integrated_gradient":
        ig[0, :, 0],

    "absolute_importance":
        importance

})


csv_path = os.path.join(
    OUTPUT_DIR,
    "ig_real_sequence_fold1.csv"
)

results.to_csv(
    csv_path,
    index=False
)

print(
    "\nSaved IG results:"
)

print(
    csv_path
)


# ============================================================
# 12. TOP 20 MOST IMPORTANT MINUTES
# ============================================================

top20 = (
    results
    .sort_values(
        "absolute_importance",
        ascending=False
    )
    .head(20)
)

top20_path = os.path.join(
    OUTPUT_DIR,
    "ig_top20_minutes_fold1.csv"
)

top20.to_csv(
    top20_path,
    index=False
)

print(
    "\nTop 20 important minutes:"
)

print(
    top20[
        [
            "minute",
            "activity",
            "integrated_gradient",
            "absolute_importance"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 13. PLOT
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    results["minute"],
    results["absolute_importance"]
)

plt.xlabel(
    "Minute of Day"
)

plt.ylabel(
    "|Integrated Gradients|"
)

plt.title(
    "Integrated Gradients — Real Actigraphy Sequence"
)

plt.tight_layout()


plot_path = os.path.join(
    OUTPUT_DIR,
    "ig_real_sequence_fold1.png"
)

plt.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved IG plot:"
)

print(
    plot_path
)


print(
    "\n=============================================="
)

print(
    "INTEGRATED GRADIENTS COMPLETE"
)

print(
    "=============================================="
)