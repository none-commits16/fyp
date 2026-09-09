# ============================================================
# HYBRID FRAMEWORK - STEP A
# Deep Sequence Branch: 1D CNN
#
# Dataset:
#   DEPRESJON + PSYKOSE
#   109 subjects
#
# Important:
#   - Subject-level 5-fold CV
#   - No subject leakage
#   - 15-minute activity representation
#   - Incomplete days allowed
#   - Training-fold normalization only
#   - 3-class classification:
#       0 = healthy
#       1 = depression
#       2 = schizophrenia
# ============================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

DEP = "data/depresjon"
PSY = "data/psykose"

OUT = "outputs/hybrid/stepA"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# 24 hours represented using 15-minute bins
BINS_PER_DAY = 96

# Minimum percentage of a day that must contain observations
MIN_COVERAGE = 0.50

N_SPLITS = 5
EPOCHS = 40
BATCH_SIZE = 16

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------


def load_subject_files():

    records = []

    # --------------------------------------------------------
    # DEPRESJON
    # --------------------------------------------------------

    folders = [
        ("condition", 1, "depression"),
        ("control", 0, "healthy"),
    ]

    for folder, label, class_name in folders:

        path = os.path.join(DEP, folder)

        for file in sorted(glob.glob(os.path.join(path, "*.csv"))):

            try:
                df = pd.read_csv(file)

                if "timestamp" not in df.columns or "activity" not in df.columns:
                    continue

                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], errors="coerce"
                )

                df["activity"] = pd.to_numeric(
                    df["activity"], errors="coerce"
                )

                df = df.dropna(subset=["timestamp", "activity"])

                if len(df) == 0:
                    continue

                subject_id = "DEP_" + os.path.splitext(
                    os.path.basename(file)
                )[0]

                records.append(
                    {
                        "subject_id": subject_id,
                        "label": label,
                        "class_name": class_name,
                        "dataset": "DEPRESJON",
                        "file": file,
                        "data": df[["timestamp", "activity"]].copy(),
                    }
                )

            except Exception as e:
                print("Skipping:", file, "|", e)

    # --------------------------------------------------------
    # PSYKOSE
    # --------------------------------------------------------

    folders = [
        ("patient", 2, "schizophrenia"),
        ("control", 0, "healthy"),
    ]

    for folder, label, class_name in folders:

        path = os.path.join(PSY, folder)

        for file in sorted(glob.glob(os.path.join(path, "*.csv"))):

            try:
                df = pd.read_csv(file)

                if "timestamp" not in df.columns or "activity" not in df.columns:
                    continue

                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], errors="coerce"
                )

                df["activity"] = pd.to_numeric(
                    df["activity"], errors="coerce"
                )

                df = df.dropna(subset=["timestamp", "activity"])

                if len(df) == 0:
                    continue

                subject_id = "PSY_" + os.path.splitext(
                    os.path.basename(file)
                )[0]

                records.append(
                    {
                        "subject_id": subject_id,
                        "label": label,
                        "class_name": class_name,
                        "dataset": "PSYKOSE",
                        "file": file,
                        "data": df[["timestamp", "activity"]].copy(),
                    }
                )

            except Exception as e:
                print("Skipping:", file, "|", e)

    return records


# ------------------------------------------------------------
# CONVERT RAW ACTIVITY INTO DAILY 15-MINUTE SEQUENCES
# ------------------------------------------------------------


def make_daily_sequences(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove duplicate timestamps
    df = df.drop_duplicates("timestamp")

    df = df.sort_values("timestamp")

    # --------------------------------------------------------
    # Resample to 15-minute intervals
    # --------------------------------------------------------

    df = df.set_index("timestamp")

    activity = df["activity"].resample("15min").mean()

    # --------------------------------------------------------
    # Process each calendar day independently
    # --------------------------------------------------------

    sequences = []

    for date, day in activity.groupby(activity.index.date):

        # Number of observed bins
        observed = day.notna().sum()

        coverage = observed / BINS_PER_DAY

        # Ignore extremely sparse days
        if coverage < MIN_COVERAGE:
            continue

        # Force exactly 96 bins
        day = day.reindex(
            pd.date_range(
                start=pd.Timestamp(date),
                periods=BINS_PER_DAY,
                freq="15min",
            )
        )

        # ----------------------------------------------------
        # Fill missing values inside the day
        # ----------------------------------------------------

        day = day.interpolate(
            method="linear",
            limit_direction="both",
        )

        # If still missing, skip
        if day.isna().any():
            continue

        seq = day.values.astype(np.float32)

        sequences.append(seq)

    return sequences


# ------------------------------------------------------------
# LOAD ALL SEQUENCES
# ------------------------------------------------------------


def build_dataset():

    subjects = load_subject_files()

    X = []
    y = []
    subject_ids = []
    metadata = []

    print("\nBuilding sequences...\n")

    for i, subject in enumerate(subjects):

        sequences = make_daily_sequences(subject["data"])

        if len(sequences) == 0:
            print(
                "WARNING: no usable days ->",
                subject["subject_id"]
            )
            continue

        for seq_idx, seq in enumerate(sequences):

            X.append(seq)
            y.append(subject["label"])
            subject_ids.append(subject["subject_id"])

            metadata.append(
                {
                    "subject_id": subject["subject_id"],
                    "label": subject["label"],
                    "class_name": subject["class_name"],
                    "dataset": subject["dataset"],
                    "sequence_id": seq_idx,
                }
            )

        print(
            f"{i+1:3d}/{len(subjects)} "
            f"{subject['subject_id']:15s} "
            f"days={len(sequences)}"
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    metadata = pd.DataFrame(metadata)

    return X, y, metadata


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------


def normalize_train_test(X_train, X_test):

    # Calculate normalization statistics ONLY from training data
    mean = np.mean(X_train)
    std = np.std(X_train)

    if std < 1e-8:
        std = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, mean, std


# ------------------------------------------------------------
# CNN MODEL
# ------------------------------------------------------------


def build_cnn():

    inputs = tf.keras.Input(
        shape=(BINS_PER_DAY, 1),
        name="activity_sequence",
    )

    # --------------------------------------------------------
    # CNN feature extraction
    # --------------------------------------------------------

    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # --------------------------------------------------------
    # PENULTIMATE EMBEDDING
    # --------------------------------------------------------

    embedding = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="cnn_embedding",
    )(x)

    embedding = tf.keras.layers.Dropout(0.30)(embedding)

    outputs = tf.keras.layers.Dense(
        3,
        activation="softmax",
        name="classification",
    )(embedding)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ------------------------------------------------------------
# SUBJECT-LEVEL PREDICTION
# ------------------------------------------------------------


def aggregate_subject_predictions(
    probabilities,
    metadata,
):

    temp = metadata.copy()

    temp["p_healthy"] = probabilities[:, 0]
    temp["p_depression"] = probabilities[:, 1]
    temp["p_schizophrenia"] = probabilities[:, 2]

    rows = []

    for subject_id, group in temp.groupby("subject_id"):

        probs = group[
            [
                "p_healthy",
                "p_depression",
                "p_schizophrenia",
            ]
        ].mean()

        true_label = int(group["label"].iloc[0])

        pred_label = int(np.argmax(probs.values))

        rows.append(
            {
                "subject_id": subject_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "p_healthy": probs["p_healthy"],
                "p_depression": probs["p_depression"],
                "p_schizophrenia": probs["p_schizophrenia"],
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------


def calculate_metrics(df):

    y_true = df["true_label"].values
    y_pred = df["pred_label"].values

    probabilities = df[
        [
            "p_healthy",
            "p_depression",
            "p_schizophrenia",
        ]
    ].values

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="weighted",
        )
    except Exception:
        auc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------


def main():

    print("=" * 70)
    print("HYBRID FRAMEWORK - STEP A: 1D CNN")
    print("=" * 70)

    print("\nTensorFlow:", tf.__version__)

    # --------------------------------------------------------
    # BUILD DATASET
    # --------------------------------------------------------

    X, y, metadata = build_dataset()

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print("Total sequences:", len(X))
    print("Unique subjects:", metadata["subject_id"].nunique())

    print("\nSubject class distribution:")

    subject_table = (
        metadata[
            ["subject_id", "label", "class_name"]
        ]
        .drop_duplicates()
        ["class_name"]
        .value_counts()
    )

    print(subject_table)

    print("\nSequence shape:", X.shape)

    # --------------------------------------------------------
    # SUBJECT-LEVEL DATA
    # --------------------------------------------------------

    subject_info = (
        metadata[
            [
                "subject_id",
                "label",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    subjects = subject_info["subject_id"].values
    subject_labels = subject_info["label"].values

    # --------------------------------------------------------
    # STRATIFIED SUBJECT-LEVEL 5-FOLD CV
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    fold_results = []
    all_predictions = []

    for fold, (train_subject_idx, test_subject_idx) in enumerate(
        skf.split(subjects, subject_labels),
        start=1,
    ):

        print("\n")
        print("=" * 70)
        print(f"FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        train_subjects = set(
            subjects[train_subject_idx]
        )

        test_subjects = set(
            subjects[test_subject_idx]
        )

        # ----------------------------------------------------
        # Select sequences belonging to subjects
        # ----------------------------------------------------

        train_mask = metadata["subject_id"].isin(
            train_subjects
        ).values

        test_mask = metadata["subject_id"].isin(
            test_subjects
        ).values

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        train_metadata = metadata.loc[
            train_mask
        ].copy()

        test_metadata = metadata.loc[
            test_mask
        ].copy()

        print(
            "Training subjects:",
            len(train_subjects)
        )

        print(
            "Testing subjects:",
            len(test_subjects)
        )

        print(
            "Training sequences:",
            len(X_train)
        )

        print(
            "Testing sequences:",
            len(X_test)
        )

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        X_train, X_test, mean, std = normalize_train_test(
            X_train,
            X_test,
        )

        # CNN expects:
        # samples x timesteps x channels

        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # ----------------------------------------------------
        # CLASS WEIGHTS
        # ----------------------------------------------------

        classes = np.unique(y_train)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )

        class_weights = {
            int(c): float(w)
            for c, w in zip(classes, weights)
        }

        print(
            "Class weights:",
            class_weights
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = build_cnn()

        # ----------------------------------------------------
        # CALLBACKS
        # ----------------------------------------------------

        callbacks = [

            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
            ),
        ]

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        # ----------------------------------------------------
        # TEST PREDICTIONS
        # ----------------------------------------------------

        probabilities = model.predict(
            X_test,
            verbose=0,
        )

        # ----------------------------------------------------
        # AGGREGATE ALL WINDOWS BACK TO SUBJECT LEVEL
        # ----------------------------------------------------

        subject_predictions = aggregate_subject_predictions(
            probabilities,
            test_metadata,
        )

        metrics = calculate_metrics(
            subject_predictions
        )

        metrics["fold"] = fold
        metrics["n_train_subjects"] = len(train_subjects)
        metrics["n_test_subjects"] = len(test_subjects)

        fold_results.append(metrics)

        # Add fold information
        subject_predictions["fold"] = fold

        all_predictions.append(
            subject_predictions
        )

        print("\nFold results:")
        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )
        print(
            f"Precision: {metrics['precision']:.4f}"
        )
        print(
            f"Recall   : {metrics['recall']:.4f}"
        )
        print(
            f"F1       : {metrics['f1']:.4f}"
        )
        print(
            f"AUC      : {metrics['auc']:.4f}"
        )

        # ----------------------------------------------------
        # SAVE FOLD MODEL
        # ----------------------------------------------------

        model.save(
            os.path.join(
                OUT,
                f"cnn_fold_{fold}.keras",
            )
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    fold_results_df = pd.DataFrame(
        fold_results
    )

    all_predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    fold_results_df.to_csv(
        os.path.join(
            OUT,
            "stepA_by_fold.csv",
        ),
        index=False,
    )

    all_predictions_df.to_csv(
        os.path.join(
            OUT,
            "stepA_subject_predictions.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    metrics_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    ]

    summary = []

    for metric in metrics_names:

        summary.append(
            {
                "metric": metric,
                "mean": fold_results_df[metric].mean(),
                "std": fold_results_df[metric].std(),
            }
        )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        os.path.join(
            OUT,
            "stepA_summary.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP A FINAL RESULTS")
    print("=" * 70)

    for _, row in summary_df.iterrows():

        print(
            f"{row['metric']:10s}: "
            f"{row['mean']:.4f} ± {row['std']:.4f}"
        )

    print("\nSaved files:")

    print(
        OUT + "/stepA_by_fold.csv"
    )

    print(
        OUT + "/stepA_summary.csv"
    )

    print(
        OUT + "/stepA_subject_predictions.csv"
    )

    print(
        OUT + "/cnn_fold_1.keras ... cnn_fold_5.keras"
    )

    print("\nStep A completed.")


if __name__ == "__main__":
    main()