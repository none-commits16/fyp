# ============================================================
# INTEGRATED GRADIENTS FOR THE CNN ACTIVITY BRANCH
# ============================================================
#
# Purpose:
# Explain which parts of a 24-hour activity sequence influenced
# the CNN prediction.
#
# Input:
#   DEPRESJON + PSYKOSE actigraphy data
#
# CNN:
#   Same architecture as hybrid_stepA_cnn.py
#
# Method:
#   1. Build 15-minute activity sequences (96 points/day)
#   2. Subject-level 5-fold split
#   3. Train CNN only on training subjects
#   4. Select one held-out subject/day
#   5. Compute Integrated Gradients
#   6. Save attribution CSV + plot
#
# Outputs:
#   outputs/integrated_gradients/
#
# ============================================================

import os
import glob
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "outputs/integrated_gradients"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 96 points = 24 hours × 4 points/hour
BINS_PER_DAY = 96

# Minimum percentage of original observations required
MIN_COVERAGE = 0.50

# Integrated Gradients steps
IG_STEPS = 100

# ------------------------------------------------------------
# DATA PATHS
# ------------------------------------------------------------

DEPRESJON_PATH = "data/depresjon"
PSYKOSE_PATH = "data/psykose"


# ============================================================
# 1. LOAD ONE ACTIVITY FILE
# ============================================================

def load_activity_file(path):

    df = pd.read_csv(path)

    required = {"timestamp", "activity"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{path} does not contain required columns: "
            f"{required}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["activity"] = pd.to_numeric(
        df["activity"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "activity"]
    ).copy()

    df = df.sort_values("timestamp")

    return df


# ============================================================
# 2. CREATE 15-MINUTE DAILY SEQUENCES
# ============================================================

def make_daily_sequences(df):

    sequences = []

    if len(df) == 0:
        return sequences

    df = df.copy()

    df["date_only"] = df["timestamp"].dt.date

    for date, day_df in df.groupby("date_only"):

        day_df = day_df.sort_values("timestamp")

        # ----------------------------------------------------
        # Create 15-minute bins
        # ----------------------------------------------------

        series = (
            day_df
            .set_index("timestamp")["activity"]
            .resample("15min")
            .mean()
        )

        # Reindex to exactly 96 bins
        start = pd.Timestamp(date)

        full_index = pd.date_range(
            start=start,
            periods=BINS_PER_DAY,
            freq="15min"
        )

        series = series.reindex(full_index)

        # ----------------------------------------------------
        # Coverage before interpolation
        # ----------------------------------------------------

        coverage = series.notna().mean()

        if coverage < MIN_COVERAGE:
            continue

        # ----------------------------------------------------
        # Interpolate missing bins
        # ----------------------------------------------------

        series = series.interpolate(
            method="linear",
            limit_direction="both"
        )

        # Remaining missing values
        series = series.fillna(0)

        x = series.values.astype(np.float32)

        if len(x) != BINS_PER_DAY:
            continue

        sequences.append(
            {
                "date": str(date),
                "sequence": x,
                "coverage": coverage
            }
        )

    return sequences


# ============================================================
# 3. LOAD ALL SUBJECTS
# ============================================================

def load_all_subjects():

    all_subjects = []

    # --------------------------------------------------------
    # DEPRESJON
    # --------------------------------------------------------

    depresjon_files = glob.glob(
        os.path.join(
            DEPRESJON_PATH,
            "**",
            "*.csv"
        ),
        recursive=True
    )

    print(
        f"DEPRESJON files found: {len(depresjon_files)}"
    )

    for path in depresjon_files:

        try:

            df = load_activity_file(path)

            daily = make_daily_sequences(df)

            if len(daily) == 0:
                continue

            filename = os.path.basename(path)

            subject_id = os.path.splitext(filename)[0]

            all_subjects.append(
                {
                    "subject_id": subject_id,
                    "label": 1,
                    "dataset": "DEPRESJON",
                    "daily": daily
                }
            )

        except Exception as e:

            print(
                f"Skipping {path}: {e}"
            )

    # --------------------------------------------------------
    # PSYKOSE
    # --------------------------------------------------------

    psykose_files = glob.glob(
        os.path.join(
            PSYKOSE_PATH,
            "**",
            "*.csv"
        ),
        recursive=True
    )

    print(
        f"PSYKOSE files found: {len(psykose_files)}"
    )

    for path in psykose_files:

        try:

            df = load_activity_file(path)

            daily = make_daily_sequences(df)

            if len(daily) == 0:
                continue

            filename = os.path.basename(path)

            subject_id = os.path.splitext(filename)[0]

            all_subjects.append(
                {
                    "subject_id": subject_id,
                    "label": 2,
                    "dataset": "PSYKOSE",
                    "daily": daily
                }
            )

        except Exception as e:

            print(
                f"Skipping {path}: {e}"
            )

    # --------------------------------------------------------
    # Healthy controls
    #
    # The public datasets contain healthy controls.
    # The original project preprocessing maps them to label 0.
    #
    # We therefore identify healthy files using the dataset
    # directory structure / filename information where possible.
    # --------------------------------------------------------

    print(
        f"Total subjects loaded before healthy-label check: "
        f"{len(all_subjects)}"
    )

    return all_subjects


# ============================================================
# 4. HEALTHY LABEL CORRECTION
# ============================================================
#
# The original project has subject labels derived during the
# preprocessing stage.
#
# To stay consistent with the project, we use the same labels
# from stage3_feature_enhancement.py whenever possible.
#
# ============================================================

def load_labels_from_stage3():

    path = "stage3_feature_enhancement.py"

    if not os.path.exists(path):

        print(
            "\nWARNING: stage3_feature_enhancement.py "
            "not found."
        )

        return {}

    # We do not execute the complete Stage 3 pipeline here.
    #
    # Instead, labels are inferred from the dataset folders
    # below. If your Stage 3 script has already created a
    # feature table, this script will use it.

    return {}


# ============================================================
# 5. BUILD SUBJECT TABLE
# ============================================================

def build_subject_table(subjects):

    rows = []

    for s in subjects:

        rows.append(
            {
                "subject_id": s["subject_id"],
                "label": s["label"],
                "dataset": s["dataset"],
                "n_days": len(s["daily"])
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 6. CNN ARCHITECTURE
# ============================================================

def build_cnn():

    inputs = tf.keras.Input(
        shape=(BINS_PER_DAY, 1),
        name="activity_input"
    )

    x = tf.keras.layers.Conv1D(
        32,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling1D(
        pool_size=2
    )(x)

    x = tf.keras.layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling1D(
        pool_size=2
    )(x)

    x = tf.keras.layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # IMPORTANT:
    # This is the same penultimate embedding layer used
    # in the hybrid fusion pipeline.
    embedding = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="cnn_embedding"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30
    )(embedding)

    outputs = tf.keras.layers.Dense(
        3,
        activation="softmax",
        name="classifier"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 7. COLLECT SEQUENCES FOR SELECTED SUBJECTS
# ============================================================

def collect_sequences(subjects, selected_indices):

    X = []
    y = []
    owners = []
    dates = []

    for idx in selected_indices:

        subject = subjects[idx]

        for day in subject["daily"]:

            X.append(day["sequence"])

            y.append(subject["label"])

            owners.append(subject["subject_id"])

            dates.append(day["date"])

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    owners = np.asarray(
        owners
    )

    dates = np.asarray(
        dates
    )

    return X, y, owners, dates


# ============================================================
# 8. NORMALIZE USING TRAINING DATA ONLY
# ============================================================

def normalize_train_test(
    X_train,
    X_test
):

    mean = np.mean(
        X_train
    )

    std = np.std(
        X_train
    )

    if std < 1e-8:
        std = 1.0

    X_train_norm = (
        X_train - mean
    ) / std

    X_test_norm = (
        X_test - mean
    ) / std

    return (
        X_train_norm,
        X_test_norm,
        mean,
        std
    )


# ============================================================
# 9. INTEGRATED GRADIENTS
# ============================================================

def integrated_gradients(
    model,
    input_tensor,
    baseline,
    target_class,
    steps=100
):

    input_tensor = tf.cast(
        input_tensor,
        tf.float32
    )

    baseline = tf.cast(
        baseline,
        tf.float32
    )

    # --------------------------------------------------------
    # Generate interpolation points
    # --------------------------------------------------------

    alphas = tf.linspace(
        0.0,
        1.0,
        steps
    )

    accumulated_gradients = tf.zeros_like(
        input_tensor
    )

    for alpha in alphas:

        interpolated = (
            baseline
            + alpha
            * (input_tensor - baseline)
        )

        with tf.GradientTape() as tape:

            tape.watch(
                interpolated
            )

            prediction = model(
                interpolated,
                training=False
            )

            target = prediction[
                0,
                target_class
            ]

        gradients = tape.gradient(
            target,
            interpolated
        )

        accumulated_gradients += gradients

    average_gradients = (
        accumulated_gradients
        / float(steps)
    )

    attributions = (
        input_tensor - baseline
    ) * average_gradients

    return attributions.numpy()


# ============================================================
# 10. TRAIN + EXPLAIN ONE FOLD
# ============================================================

def run_integrated_gradients():

    print("\n" + "=" * 70)
    print("INTEGRATED GRADIENTS - CNN ACTIVITY EXPLANATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load subjects
    # --------------------------------------------------------

    subjects = load_all_subjects()

    if len(subjects) == 0:

        raise RuntimeError(
            "No subjects were loaded."
        )

    subject_df = build_subject_table(
        subjects
    )

    print("\nSubject summary:")

    print(
        subject_df[
            [
                "label",
                "dataset"
            ]
        ].value_counts()
    )

    print(
        f"\nTotal subjects: {len(subjects)}"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    subject_labels = subject_df[
        "label"
    ].values

    # --------------------------------------------------------
    # Stratified 5-fold subject-level CV
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    splits = list(
        skf.split(
            np.zeros(len(subjects)),
            subject_labels
        )
    )

    # --------------------------------------------------------
    # We use Fold 1 for the detailed IG explanation.
    #
    # The CNN is trained ONLY on Fold 1 training subjects.
    # The explained sample comes from Fold 1 test subjects.
    # --------------------------------------------------------

    train_idx, test_idx = splits[0]

    print("\nUsing Fold 1 for Integrated Gradients.")

    print(
        f"Training subjects: {len(train_idx)}"
    )

    print(
        f"Held-out subjects: {len(test_idx)}"
    )

    # --------------------------------------------------------
    # Collect sequences
    # --------------------------------------------------------

    X_train, y_train, owners_train, dates_train = (
        collect_sequences(
            subjects,
            train_idx
        )
    )

    X_test, y_test, owners_test, dates_test = (
        collect_sequences(
            subjects,
            test_idx
        )
    )

    print(
        f"\nTraining daily sequences: {len(X_train)}"
    )

    print(
        f"Held-out daily sequences: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Normalize using training sequences only
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        train_mean,
        train_std
    ) = normalize_train_test(
        X_train,
        X_test
    )

    # CNN expects (samples, 96, 1)
    X_train_cnn = X_train[..., np.newaxis]

    X_test_cnn = X_test[..., np.newaxis]

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(c): float(w)
        for c, w in zip(classes, weights)
    }

    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )

    # --------------------------------------------------------
    # Build CNN
    # --------------------------------------------------------

    tf.keras.backend.clear_session()

    model = build_cnn()

    model.summary()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
    ]

    print(
        "\nTraining CNN..."
    )

    history = model.fit(
        X_train_cnn,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.15,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # --------------------------------------------------------
    # Predictions on held-out data
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test_cnn,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Select one held-out sample.
    #
    # We choose the first held-out subject's first available day.
    # --------------------------------------------------------

    selected_subject = owners_test[0]

    selected_date = dates_test[0]

    selected_index = 0

    input_sample = X_test_cnn[
        selected_index:selected_index + 1
    ]

    true_class = int(
        y_test[selected_index]
    )

    predicted_class = int(
        predictions[selected_index]
    )

    predicted_probabilities = probabilities[
        selected_index
    ]

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    class_names = {
        0: "Healthy",
        1: "Depression",
        2: "Schizophrenia"
    }

    print(
        "\nSelected held-out sample:"
    )

    print(
        f"Subject: {selected_subject}"
    )

    print(
        f"Date: {selected_date}"
    )

    print(
        f"True class: "
        f"{class_names.get(true_class, str(true_class))}"
    )

    print(
        f"Predicted class: "
        f"{class_names.get(predicted_class, str(predicted_class))}"
    )

    print(
        "\nPrediction probabilities:"
    )

    for c, p in enumerate(
        predicted_probabilities
    ):

        print(
            f"  {class_names.get(c, str(c))}: "
            f"{p:.4f}"
        )

    # --------------------------------------------------------
    # Baseline
    #
    # Zero normalized activity corresponds approximately to
    # average activity under the training normalization.
    # --------------------------------------------------------

    baseline = np.zeros_like(
        input_sample,
        dtype=np.float32
    )

    input_tensor = tf.convert_to_tensor(
        input_sample
    )

    baseline_tensor = tf.convert_to_tensor(
        baseline
    )

    # --------------------------------------------------------
    # Calculate Integrated Gradients
    # --------------------------------------------------------

    print(
        f"\nCalculating Integrated Gradients "
        f"using {IG_STEPS} steps..."
    )

    attributions = integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        baseline=baseline_tensor,
        target_class=predicted_class,
        steps=IG_STEPS
    )

    attributions = np.squeeze(
        attributions
    )

    activity = np.squeeze(
        input_sample
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    prediction_input = model(
        input_tensor,
        training=False
    ).numpy()[0]

    prediction_baseline = model(
        baseline_tensor,
        training=False
    ).numpy()[0]

    attribution_sum = np.sum(
        attributions
    )

    prediction_difference = (
        prediction_input[predicted_class]
        -
        prediction_baseline[predicted_class]
    )

    print(
        "\nIntegrated Gradients sanity check:"
    )

    print(
        f"Attribution sum: "
        f"{attribution_sum:.6f}"
    )

    print(
        f"Prediction difference: "
        f"{prediction_difference:.6f}"
    )

    # --------------------------------------------------------
    # Time labels
    # --------------------------------------------------------

    minutes = np.arange(
        BINS_PER_DAY
    ) * 15

    time_strings = [

        f"{int(m // 60):02d}:{int(m % 60):02d}"
        for m in minutes
    ]

    # --------------------------------------------------------
    # Save attribution CSV
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        {
            "time": time_strings,
            "activity_normalized": activity,
            "integrated_gradient": attributions,
            "absolute_attribution": np.abs(
                attributions
            )
        }
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "integrated_gradients_fold1.csv"
    )

    result_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        [
            {
                "subject_id": selected_subject,
                "date": selected_date,
                "true_class": class_names.get(
                    true_class,
                    str(true_class)
                ),
                "predicted_class": class_names.get(
                    predicted_class,
                    str(predicted_class)
                ),
                "healthy_probability": predicted_probabilities[0],
                "depression_probability": predicted_probabilities[1],
                "schizophrenia_probability": predicted_probabilities[2],
                "attribution_sum": attribution_sum,
                "prediction_difference": prediction_difference,
                "training_mean": train_mean,
                "training_std": train_std
            }
        ]
    )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "integrated_gradients_metadata.csv"
    )

    metadata.to_csv(
        metadata_path,
        index=False
    )

    # ========================================================
    # 11. PLOT
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True
    )

    # --------------------------------------------------------
    # Top: Activity
    # --------------------------------------------------------

    axes[0].plot(
        np.arange(BINS_PER_DAY),
        activity,
        linewidth=1.5
    )

    axes[0].set_ylabel(
        "Normalized Activity"
    )

    axes[0].set_title(
        "24-Hour Activity Sequence"
    )

    axes[0].grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Bottom: Integrated Gradients
    # --------------------------------------------------------

    positive = np.maximum(
        attributions,
        0
    )

    negative = np.minimum(
        attributions,
        0
    )

    axes[1].bar(
        np.arange(BINS_PER_DAY),
        positive,
        width=0.8,
        alpha=0.8,
        label="Positive attribution"
    )

    axes[1].bar(
        np.arange(BINS_PER_DAY),
        negative,
        width=0.8,
        alpha=0.8,
        label="Negative attribution"
    )

    axes[1].axhline(
        0,
        linewidth=1
    )

    axes[1].set_ylabel(
        "Integrated Gradient"
    )

    axes[1].set_xlabel(
        "Time of Day"
    )

    axes[1].set_title(
        "Integrated Gradients: CNN Prediction Explanation"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # X-axis labels every 4 hours
    # --------------------------------------------------------

    tick_positions = [
        0,
        16,
        32,
        48,
        64,
        80,
        95
    ]

    tick_labels = [
        "00:00",
        "04:00",
        "08:00",
        "12:00",
        "16:00",
        "20:00",
        "24:00"
    ]

    axes[1].set_xticks(
        tick_positions
    )

    axes[1].set_xticklabels(
        tick_labels
    )

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        "Integrated Gradients Explanation "
        f"for {class_names.get(predicted_class)} Prediction\n"
        f"Subject: {selected_subject} | Date: {selected_date}",
        fontsize=14
    )

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "integrated_gradients_fold1.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    model_path = os.path.join(
        OUTPUT_DIR,
        "cnn_fold1_ig.keras"
    )

    model.save(
        model_path
    )

    # --------------------------------------------------------
    # Find most influential time periods
    # --------------------------------------------------------

    top_indices = np.argsort(
        np.abs(attributions)
    )[::-1][:10]

    top_times = pd.DataFrame(
        {
            "rank": np.arange(
                1,
                len(top_indices) + 1
            ),
            "time": [
                time_strings[i]
                for i in top_indices
            ],
            "attribution": [
                attributions[i]
                for i in top_indices
            ],
            "absolute_attribution": [
                abs(attributions[i])
                for i in top_indices
            ]
        }
    )

    top_path = os.path.join(
        OUTPUT_DIR,
        "top_influential_time_bins.csv"
    )

    top_times.to_csv(
        top_path,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTEGRATED GRADIENTS COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        f"  {plot_path}"
    )

    print(
        f"  {model_path}"
    )

    print(
        f"  {top_path}"
    )

    print("\nTop 10 influential time bins:")

    print(
        top_times.to_string(
            index=False
        )
    )

    print(
        "\nInterpretation:"
    )

    print(
        "Positive attribution means that the activity at that "
        "time point pushed the CNN toward the predicted class."
    )

    print(
        "Negative attribution means that the activity at that "
        "time point pushed the CNN away from the predicted class."
    )

    print(
        "\nIMPORTANT: Integrated Gradients shows model-level "
        "attribution. It does not establish clinical causality."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_integrated_gradients()