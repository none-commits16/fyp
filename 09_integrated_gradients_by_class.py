import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# CLASS-SPECIFIC INTEGRATED GRADIENTS
# ============================================================

MODEL_PATH = "outputs/deep_learning/models/cnn_fold_1.keras"

OUTPUT_DIR = "outputs/explainability/integrated_gradients_by_class"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

HEALTHY = 0
DEPRESSION = 1
SCHIZOPHRENIA = 2

CLASS_NAMES = {
    0: "Healthy",
    1: "Depression",
    2: "Schizophrenia"
}


# ============================================================
# DATASET PATHS
# ============================================================

DEPRESJON_DIR = "data/depresjon"
PSYKOSE_DIR = "data/psykose"


# ============================================================
# FIND FILES AND ASSIGN LABELS
# ============================================================

def get_files_with_labels():

    files = []

    # Depression
    for file_path in glob.glob(
        os.path.join(
            DEPRESJON_DIR,
            "condition",
            "*.csv"
        )
    ):
        files.append(
            (file_path, DEPRESSION)
        )

    # Healthy from Depression dataset
    for file_path in glob.glob(
        os.path.join(
            DEPRESJON_DIR,
            "control",
            "*.csv"
        )
    ):
        files.append(
            (file_path, HEALTHY)
        )

    # Schizophrenia
    for file_path in glob.glob(
        os.path.join(
            PSYKOSE_DIR,
            "patient",
            "*.csv"
        )
    ):
        files.append(
            (file_path, SCHIZOPHRENIA)
        )

    # Healthy from Psychosis dataset
    for file_path in glob.glob(
        os.path.join(
            PSYKOSE_DIR,
            "control",
            "*.csv"
        )
    ):
        files.append(
            (file_path, HEALTHY)
        )

    return files


# ============================================================
# EXTRACT 1440-MINUTE DAILY SEQUENCES
# ============================================================

def extract_sequences(
    file_path,
    label
):

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
                    "label": label,
                    "date": date,
                    "file": file_path
                }
            )

    return sequences


# ============================================================
# NORMALIZATION
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
# INTEGRATED GRADIENTS
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
# LOAD MODEL
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
# LOAD ALL REAL SEQUENCES
# ============================================================

files = get_files_with_labels()

print(
    f"\nFound {len(files)} subject files."
)

all_sequences = []

for file_path, label in files:

    sequences = extract_sequences(
        file_path,
        label
    )

    all_sequences.extend(
        sequences
    )

print(
    f"Found {len(all_sequences)} complete "
    f"1440-minute sequences."
)


# ============================================================
# SHOW CLASS COUNTS
# ============================================================

print(
    "\nSequence counts by true class:"
)

for class_id in [
    HEALTHY,
    DEPRESSION,
    SCHIZOPHRENIA
]:

    count = sum(
        item["label"] == class_id
        for item in all_sequences
    )

    print(
        f"  {CLASS_NAMES[class_id]}: {count}"
    )


# ============================================================
# LIMIT SEQUENCES PER CLASS
# ============================================================

MAX_PER_CLASS = 25

selected_sequences = []

for class_id in [
    HEALTHY,
    DEPRESSION,
    SCHIZOPHRENIA
]:

    class_sequences = [
        item
        for item in all_sequences
        if item["label"] == class_id
    ]

    selected_sequences.extend(
        class_sequences[
            :MAX_PER_CLASS
        ]
    )

print(
    f"\nUsing up to {MAX_PER_CLASS} "
    f"sequences per class."
)

print(
    f"Total selected: "
    f"{len(selected_sequences)}"
)


# ============================================================
# CALCULATE CLASS-SPECIFIC IG
# ============================================================

all_results = []


for sequence_number, item in enumerate(
    selected_sequences,
    start=1
):

    true_class = item["label"]

    print(
        f"\nProcessing "
        f"{sequence_number}/"
        f"{len(selected_sequences)}"
        f" — True class: "
        f"{CLASS_NAMES[true_class]}"
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


    # --------------------------------------------------------
    # Explain the TRUE class
    # --------------------------------------------------------

    ig = integrated_gradients(
        model=model,
        x=sample,
        target_class=true_class,
        steps=50
    )

    importance = np.abs(
        ig[0, :, 0]
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    result = pd.DataFrame({

        "sequence":
            sequence_number,

        "file":
            item["file"],

        "date":
            str(item["date"]),

        "true_class":
            true_class,

        "true_class_name":
            CLASS_NAMES[true_class],

        "predicted_class":
            predicted_class,

        "predicted_class_name":
            CLASS_NAMES[predicted_class],

        "minute":
            np.arange(1440),

        "activity":
            activity,

        "normalized_activity":
            normalized_activity,

        "integrated_gradient":
            ig[0, :, 0],

        "absolute_importance":
            importance

    })

    all_results.append(
        result
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

combined = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

all_path = os.path.join(
    OUTPUT_DIR,
    "ig_class_specific_all.csv"
)

combined.to_csv(
    all_path,
    index=False
)

print(
    "\nSaved:",
    all_path
)


# ============================================================
# AGGREGATE BY CLASS AND MINUTE
# ============================================================

class_summary = (
    combined
    .groupby(
        [
            "true_class",
            "true_class_name",
            "minute"
        ]
    )[
        "absolute_importance"
    ]
    .mean()
    .reset_index()
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "ig_class_specific_mean.csv"
)

class_summary.to_csv(
    summary_path,
    index=False
)

print(
    "Saved:",
    summary_path
)


# ============================================================
# TOP 20 MINUTES FOR EACH CLASS
# ============================================================

for class_id in [
    HEALTHY,
    DEPRESSION,
    SCHIZOPHRENIA
]:

    class_name = CLASS_NAMES[
        class_id
    ]

    class_data = class_summary[
        class_summary[
            "true_class"
        ] == class_id
    ]

    top20 = (
        class_data
        .sort_values(
            "absolute_importance",
            ascending=False
        )
        .head(20)
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"ig_top20_{class_name.lower()}.csv"
    )

    top20.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nTop 20 — {class_name}"
    )

    print(
        top20[
            [
                "minute",
                "absolute_importance"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# CLASS-SPECIFIC PLOTS
# ============================================================

for class_id in [
    HEALTHY,
    DEPRESSION,
    SCHIZOPHRENIA
]:

    class_name = CLASS_NAMES[
        class_id
    ]

    class_data = class_summary[
        class_summary[
            "true_class"
        ] == class_id
    ]

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        class_data["minute"],
        class_data[
            "absolute_importance"
        ]
    )

    plt.xlabel(
        "Minute of Day"
    )

    plt.ylabel(
        "Mean |Integrated Gradients|"
    )

    plt.title(
        f"Integrated Gradients — "
        f"{class_name}"
    )

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        f"ig_{class_name.lower()}.png"
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
# COMPLETE
# ============================================================

print(
    "\n=============================================="
)

print(
    "CLASS-SPECIFIC INTEGRATED GRADIENTS COMPLETE"
)

print(
    "=============================================="
)