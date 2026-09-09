# ============================================================
# HYBRID STEP C - FULL NESTED STACKING
#
# Outer 5-fold subject-level CV
# Inner 4-fold subject-level OOF
#
# Base models:
#   1. Random Forest
#   2. CatBoost
#   3. LightGBM
#   4. CNN + handcrafted feature fusion
#
# Meta learner:
#   Logistic Regression
#
# Classes:
#   0 = Healthy
#   1 = Depression
#   2 = Schizophrenia
# ============================================================

import os
import glob
import warnings
import random
import ast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Model, Input

from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import BorderlineSMOTE


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

SEQ_LEN = 96

CNN_BATCH = 16

INNER_EPOCHS = 8

OUTER_EPOCHS = 10

OUT_DIR = "outputs/hybrid/stepC_full_nested"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)

random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# DATA PATHS
# ============================================================

DEP_CONDITION = "data/depresjon/condition/*.csv"

DEP_CONTROL = "data/depresjon/control/*.csv"

PSY_PATIENT = "data/psykose/patient/*.csv"

PSY_CONTROL = "data/psykose/control/*.csv"


# ============================================================
# STAGE 3 FEATURES
# ============================================================

FEATURES = [

    "mean_activity",
    "std_activity",
    "median_activity",
    "max_activity",
    "q25_activity",
    "q75_activity",
    "iqr_activity",
    "entropy",
    "zero_crossings",
    "sleep_hours",
    "active_minutes",
    "day_activity",
    "night_activity",
    "day_night_ratio",
    "circ_regularity",
    "prop_inactive",
    "prop_active",
    "n_days",

    "IS",
    "IV",
    "RA",

    "morning_activity",
    "afternoon_activity",
    "evening_activity",
    "night_activity",

    "sample_entropy",
    "lz_complexity"
]


# ============================================================
# CNN
# ============================================================

def build_cnn():

    inputs = Input(
        shape=(SEQ_LEN, 1)
    )

    x = Conv1D(
        32,
        5,
        padding="same",
        activation="relu"
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling1D(
        2
    )(x)

    x = Conv1D(
        64,
        5,
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(
        2
    )(x)

    x = Conv1D(
        128,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    embedding = Dense(
        64,
        activation="relu",
        name="cnn_embedding"
    )(x)

    x = Dropout(
        0.30
    )(embedding)

    outputs = Dense(
        3,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    embedding_model = Model(
        inputs=model.input,
        outputs=embedding
    )

    return model, embedding_model


# ============================================================
# DAILY 15-MINUTE SEQUENCES
# ============================================================

def make_daily_sequences(filepath):

    df = pd.read_csv(filepath)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"'timestamp' missing in {filepath}"
        )

    if "activity" not in df.columns:
        raise ValueError(
            f"'activity' missing in {filepath}"
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
        subset=[
            "timestamp",
            "activity"
        ]
    )

    if len(df) == 0:
        return []

    df = df.sort_values(
        "timestamp"
    )

    df["day"] = (
        df["timestamp"]
        .dt.floor("D")
    )

    sequences = []

    for day, day_df in df.groupby("day"):

        series = (
            day_df
            .set_index("timestamp")["activity"]
            .resample("15min")
            .mean()
        )

        coverage = (
            series.notna().sum() /
            SEQ_LEN
        )

        if coverage < 0.50:
            continue

        series = series.interpolate(
            method="linear",
            limit_direction="both"
        )

        full_index = pd.date_range(
            start=pd.Timestamp(day),
            periods=SEQ_LEN,
            freq="15min"
        )

        series = series.reindex(
            full_index
        )

        series = series.interpolate(
            method="linear",
            limit_direction="both"
        )

        if series.isna().any():
            continue

        values = series.to_numpy(
            dtype=np.float32
        )

        if len(values) != SEQ_LEN:
            continue

        sequences.append(values)

    return sequences


# ============================================================
# LOAD ALL CNN SEQUENCES
# ============================================================

def load_all_sequences():

    records = []

    # --------------------------------------------------------
    # DEPRESJON - DEPRESSION
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(DEP_CONDITION)
    ):

        sid = (
            "DEP_" +
            os.path.splitext(
                os.path.basename(filepath)
            )[0]
        )

        sequences = make_daily_sequences(
            filepath
        )

        for seq in sequences:

            records.append(
                {
                    "subject_id": sid,
                    "label": 1,
                    "sequence": seq
                }
            )

    # --------------------------------------------------------
    # DEPRESJON - HEALTHY
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(DEP_CONTROL)
    ):

        sid = (
            "DEP_" +
            os.path.splitext(
                os.path.basename(filepath)
            )[0]
        )

        sequences = make_daily_sequences(
            filepath
        )

        for seq in sequences:

            records.append(
                {
                    "subject_id": sid,
                    "label": 0,
                    "sequence": seq
                }
            )

    # --------------------------------------------------------
    # PSYKOSE - SCHIZOPHRENIA
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(PSY_PATIENT)
    ):

        sid = (
            "PSY_" +
            os.path.splitext(
                os.path.basename(filepath)
            )[0]
        )

        sequences = make_daily_sequences(
            filepath
        )

        for seq in sequences:

            records.append(
                {
                    "subject_id": sid,
                    "label": 2,
                    "sequence": seq
                }
            )

    # --------------------------------------------------------
    # PSYKOSE - HEALTHY
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(PSY_CONTROL)
    ):

        sid = (
            "PSY_" +
            os.path.splitext(
                os.path.basename(filepath)
            )[0]
        )

        sequences = make_daily_sequences(
            filepath
        )

        for seq in sequences:

            records.append(
                {
                    "subject_id": sid,
                    "label": 0,
                    "sequence": seq
                }
            )

    return records


# ============================================================
# ORGANIZE SEQUENCES BY SUBJECT
# ============================================================

def prepare_sequence_data():

    records = load_all_sequences()

    subject_sequences = {}

    subject_labels = {}

    for record in records:

        sid = record["subject_id"]

        if sid not in subject_sequences:
            subject_sequences[sid] = []

        subject_sequences[sid].append(
            record["sequence"]
        )

        subject_labels[sid] = record["label"]

    print(
        "\nSubjects with usable CNN sequences:",
        len(subject_sequences)
    )

    return (
        subject_sequences,
        subject_labels
    )


# ============================================================
# GET SEQUENCES FOR SUBJECTS
# ============================================================

def sequences_for_subjects(
    subject_ids,
    subject_sequences,
    subject_labels
):

    X = []

    y = []

    owners = []

    for sid in subject_ids:

        if sid not in subject_sequences:
            continue

        for seq in subject_sequences[sid]:

            X.append(seq)

            y.append(
                subject_labels[sid]
            )

            owners.append(sid)

    if len(X) == 0:

        return (
            np.empty(
                (0, SEQ_LEN, 1),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.int32
            ),
            []
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    X = X.reshape(
        -1,
        SEQ_LEN,
        1
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    return X, y, owners


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sequences(
    X_train,
    X_other
):

    mean = np.mean(
        X_train
    )

    std = np.std(
        X_train
    )

    if std < 1e-8:
        std = 1.0

    return (
        (X_train - mean) / std,
        (X_other - mean) / std
    )


# ============================================================
# CNN CLASS WEIGHTS
# ============================================================

def cnn_class_weights(y):

    counts = np.bincount(
        y,
        minlength=3
    )

    total = len(y)

    weights = {}

    for cls in range(3):

        if counts[cls] > 0:

            weights[cls] = (
                total /
                (3 * counts[cls])
            )

    return weights


# ============================================================
# SUBJECT EMBEDDINGS
#
# IMPORTANT:
# Use direct model call instead of predict().
# This guarantees one embedding per input sequence.
# ============================================================

def get_subject_embeddings(
    embedding_model,
    X,
    owners
):

    if len(X) == 0:
        return {}

    if len(X) != len(owners):

        raise ValueError(
            "Sequence/owner mismatch: "
            f"X={len(X)}, owners={len(owners)}"
        )

    embeddings = (
        embedding_model(
            X,
            training=False
        )
        .numpy()
    )

    if len(embeddings) != len(owners):

        raise ValueError(
            "Embedding/owner mismatch: "
            f"embeddings={len(embeddings)}, "
            f"owners={len(owners)}"
        )

    subject_vectors = {}

    for embedding, sid in zip(
        embeddings,
        owners
    ):

        if sid not in subject_vectors:
            subject_vectors[sid] = []

        subject_vectors[sid].append(
            embedding
        )

    result = {}

    for sid, vectors in (
        subject_vectors.items()
    ):

        result[sid] = np.mean(
            np.asarray(vectors),
            axis=0
        )

    return result


# ============================================================
# TRAIN CNN
# ============================================================

def train_cnn(
    train_subjects,
    subject_sequences,
    subject_labels,
    epochs
):

    X_all, y_all, owners_all = (
        sequences_for_subjects(
            train_subjects,
            subject_sequences,
            subject_labels
        )
    )

    if len(X_all) == 0:

        raise ValueError(
            "No CNN training sequences."
        )

    # --------------------------------------------------------
    # Internal subject-level validation
    # --------------------------------------------------------

    subject_y = np.asarray(
        [
            subject_labels[s]
            for s in train_subjects
        ]
    )

    internal_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    fit_idx, val_idx = next(
        internal_cv.split(
            train_subjects,
            subject_y
        )
    )

    fit_subjects = (
        np.asarray(train_subjects)[
            fit_idx
        ]
    )

    val_subjects = (
        np.asarray(train_subjects)[
            val_idx
        ]
    )

    X_fit, y_fit, _ = (
        sequences_for_subjects(
            fit_subjects.tolist(),
            subject_sequences,
            subject_labels
        )
    )

    X_val, y_val, _ = (
        sequences_for_subjects(
            val_subjects.tolist(),
            subject_sequences,
            subject_labels
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    X_fit, X_val = (
        normalize_sequences(
            X_fit,
            X_val
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, embedding_model = build_cnn()

    weights = cnn_class_weights(
        y_fit
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        X_fit,
        y_fit,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=epochs,
        batch_size=CNN_BATCH,
        class_weight=weights,
        callbacks=[
            early_stop
        ],
        verbose=0
    )

    # --------------------------------------------------------
    # Get embeddings for ALL current training subjects
    # --------------------------------------------------------

    X_all_norm, _ = (
        normalize_sequences(
            X_all,
            X_all
        )
    )

    embeddings = (
        get_subject_embeddings(
            embedding_model,
            X_all_norm,
            owners_all
        )
    )

    return (
        model,
        embedding_model,
        embeddings
    )


# ============================================================
# LOAD STAGE 3 FEATURE EXTRACTION FUNCTION
# ============================================================

def load_stage3_extract():

    path = "stage3_feature_enhancement.py"

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Cannot find {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        source = f.read()

    tree = ast.parse(
        source
    )

    nodes = []

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef
            )
        ):

            nodes.append(node)

    module = ast.Module(
        body=nodes,
        type_ignores=[]
    )

    ast.fix_missing_locations(
        module
    )

    namespace = {}

    exec(
        compile(
            module,
            path,
            "exec"
        ),
        namespace
    )

    if "extract" not in namespace:

        raise ValueError(
            "Could not find extract() "
            "in stage3_feature_enhancement.py"
        )

    return namespace["extract"]


# ============================================================
# LOAD HANDCRAFTED FEATURES
# ============================================================

def load_handcrafted_features():

    extract = load_stage3_extract()

    rows = []

    # --------------------------------------------------------
    # DEPRESJON CONDITION
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(DEP_CONDITION)
    ):

        row = extract(
            filepath,
            1,
            "depression"
        )

        row["subject_id"] = (
            "DEP_" +
            os.path.splitext(
                os.path.basename(filepath)
            )[0]
        )

        rows.append(row)

    # --------------------------------------------------------
    # DEPRESJON CONTROL
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(DEP_CONTROL)
    ):

        row = extract(
            filepath,
            0,
            "healthy"
        )

        row["subject_id"] = (
            "DEP_" +
            os.path.splitext(
                os.path.basename(filepath)
            )[0]
        )

        rows.append(row)

    # --------------------------------------------------------
    # PSYKOSE PATIENT
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(PSY_PATIENT)
    ):

        row = extract(
            filepath,
            2,
            "schizophrenia"
        )

        row["subject_id"] = (
            "PSY_" +
            os.path.splitext(
                os.path.basename(filepath)
            )[0]
        )

        rows.append(row)

    # --------------------------------------------------------
    # PSYKOSE CONTROL
    # --------------------------------------------------------

    for filepath in sorted(
        glob.glob(PSY_CONTROL)
    ):

        row = extract(
            filepath,
            0,
            "healthy"
        )

        row["subject_id"] = (
            "PSY_" +
            os.path.splitext(
                os.path.basename(filepath)
            )[0]
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.set_index(
        "subject_id"
    )

    return df


# ============================================================
# BUILD FUSION FEATURES
#
# 64 CNN embedding
# +
# 27 handcrafted features
#
# = 91-dimensional vector
# ============================================================

def build_fusion_matrix(
    subject_ids,
    embeddings,
    handcrafted_df
):

    rows = []

    valid_subjects = []

    for sid in subject_ids:

        if sid not in embeddings:
            continue

        if sid not in handcrafted_df.index:
            continue

        handcrafted = (
            handcrafted_df
            .loc[
                sid,
                FEATURES
            ]
            .to_numpy(
                dtype=float
            )
        )

        embedding = np.asarray(
            embeddings[sid],
            dtype=float
        )

        combined = np.concatenate(
            [
                embedding,
                handcrafted
            ]
        )

        rows.append(combined)

        valid_subjects.append(sid)

    if len(rows) == 0:

        raise ValueError(
            "No valid subjects for fusion."
        )

    return (
        np.asarray(
            rows,
            dtype=float
        ),
        valid_subjects
    )


# ============================================================
# TRAIN TREE MODELS
# ============================================================

def train_tree_models(
    X_train,
    y_train
):

    # --------------------------------------------------------
    # Borderline-SMOTE
    # --------------------------------------------------------

    smote = BorderlineSMOTE(
        random_state=SEED,
        k_neighbors=3
    )

    X_smote, y_smote = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    rf.fit(
        X_smote,
        y_smote
    )

    # --------------------------------------------------------
    # CATBOOST
    # --------------------------------------------------------

    cat = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.03,
        loss_function="MultiClass",
        random_seed=SEED,
        verbose=False,
        class_weights=[
            1.0,
            64 / 23,
            64 / 22
        ]
    )

    cat.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # LIGHTGBM
    # --------------------------------------------------------

    lgb = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        verbosity=-1
    )

    lgb.fit(
        X_smote,
        y_smote
    )

    return rf, cat, lgb


# ============================================================
# FORCE PROBABILITIES INTO 3 CLASS ORDER
# ============================================================

def force_three_classes(
    probability,
    model
):

    output = np.zeros(
        (
            len(probability),
            3
        ),
        dtype=float
    )

    for i, cls in enumerate(
        model.classes_
    ):

        output[:, int(cls)] = (
            probability[:, i]
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "FULL NESTED HYBRID STEP C"
    )

    print(
        "CNN + 27 FEATURES + RF + CATBOOST + LIGHTGBM"
    )

    print(
        "STRICT OOF LOGISTIC META-LEARNER"
    )

    print("=" * 70)

    # ========================================================
    # HANDCRAFTED FEATURES
    # ========================================================

    print(
        "\nLoading handcrafted features..."
    )

    handcrafted_df = (
        load_handcrafted_features()
    )

    print(
        "Handcrafted subjects:",
        len(handcrafted_df)
    )

    # ========================================================
    # CNN DATA
    # ========================================================

    print(
        "\nLoading CNN sequences..."
    )

    (
        subject_sequences,
        subject_labels
    ) = prepare_sequence_data()

    # ========================================================
    # COMMON SUBJECTS
    # ========================================================

    common_subjects = sorted(
        set(handcrafted_df.index)
        &
        set(subject_sequences.keys())
    )

    print(
        "Common subjects:",
        len(common_subjects)
    )

    subjects = np.asarray(
        common_subjects
    )

    labels = np.asarray(
        [
            subject_labels[s]
            for s in subjects
        ]
    )

    print(
        "\nClass distribution:"
    )

    print(
        pd.Series(labels)
        .map(
            {
                0: "healthy",
                1: "depression",
                2: "schizophrenia"
            }
        )
        .value_counts()
    )

    # ========================================================
    # OUTER CV
    # ========================================================

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    outer_results = []

    outer_predictions = []

    # ========================================================
    # OUTER LOOP
    # ========================================================

    for outer_fold, (
        outer_train_idx,
        outer_test_idx
    ) in enumerate(
        outer_cv.split(
            subjects,
            labels
        ),
        start=1
    ):

        print("\n")
        print("=" * 70)

        print(
            f"OUTER FOLD {outer_fold}/5"
        )

        print("=" * 70)

        outer_train_subjects = (
            subjects[
                outer_train_idx
            ]
        )

        outer_test_subjects = (
            subjects[
                outer_test_idx
            ]
        )

        y_outer_train = (
            labels[
                outer_train_idx
            ]
        )

        # ====================================================
        # INNER OOF MATRIX
        #
        # RF       = 3
        # CatBoost = 3
        # LightGBM = 3
        # Fusion   = 3
        #
        # Total = 12
        # ====================================================

        inner_meta = np.full(
            (
                len(
                    outer_train_subjects
                ),
                12
            ),
            np.nan,
            dtype=float
        )

        # subject -> row in OOF matrix

        subject_to_row = {
            sid: i
            for i, sid in enumerate(
                outer_train_subjects
            )
        }

        # ====================================================
        # INNER CV
        # ====================================================

        inner_cv = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=SEED
        )

        for inner_fold, (
            inner_train_idx,
            inner_val_idx
        ) in enumerate(
            inner_cv.split(
                outer_train_subjects,
                y_outer_train
            ),
            start=1
        ):

            print(
                f"\n  INNER FOLD "
                f"{inner_fold}/4"
            )

            inner_train_subjects = (
                outer_train_subjects[
                    inner_train_idx
                ]
            )

            inner_val_subjects = (
                outer_train_subjects[
                    inner_val_idx
                ]
            )

            y_inner_train = (
                y_outer_train[
                    inner_train_idx
                ]
            )

            # =================================================
            # CNN
            # =================================================

            print(
                "    Training CNN..."
            )

            (
                cnn_model,
                embedding_model,
                train_embeddings
            ) = train_cnn(
                inner_train_subjects.tolist(),
                subject_sequences,
                subject_labels,
                INNER_EPOCHS
            )

            # =================================================
            # VALIDATION CNN EMBEDDINGS
            # =================================================

            X_val_seq, _, val_owners = (
                sequences_for_subjects(
                    inner_val_subjects.tolist(),
                    subject_sequences,
                    subject_labels
                )
            )

            X_train_seq, _, _ = (
                sequences_for_subjects(
                    inner_train_subjects.tolist(),
                    subject_sequences,
                    subject_labels
                )
            )

            X_train_seq, X_val_seq = (
                normalize_sequences(
                    X_train_seq,
                    X_val_seq
                )
            )

            val_embeddings = (
                get_subject_embeddings(
                    embedding_model,
                    X_val_seq,
                    val_owners
                )
            )

            # =================================================
            # FUSION
            # =================================================

            print(
                "    Training CNN + feature fusion..."
            )

            (
                X_fusion_train,
                valid_train_subjects
            ) = build_fusion_matrix(
                inner_train_subjects,
                train_embeddings,
                handcrafted_df
            )

            (
                X_fusion_val,
                valid_val_subjects
            ) = build_fusion_matrix(
                inner_val_subjects,
                val_embeddings,
                handcrafted_df
            )

            y_fusion_train = np.asarray(
                [
                    subject_labels[s]
                    for s in valid_train_subjects
                ]
            )

            y_fusion_val = np.asarray(
                [
                    subject_labels[s]
                    for s in valid_val_subjects
                ]
            )

            fusion_imputer = (
                SimpleImputer(
                    strategy="median"
                )
            )

            X_fusion_train = (
                fusion_imputer.fit_transform(
                    X_fusion_train
                )
            )

            X_fusion_val = (
                fusion_imputer.transform(
                    X_fusion_val
                )
            )

            fusion_model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=4,
                min_child_samples=6,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=SEED,
                verbosity=-1
            )

            fusion_model.fit(
                X_fusion_train,
                y_fusion_train
            )

            fusion_prob = (
                fusion_model.predict_proba(
                    X_fusion_val
                )
            )

            fusion_prob = (
                force_three_classes(
                    fusion_prob,
                    fusion_model
                )
            )

            # =================================================
            # TREE MODELS
            # =================================================

            print(
                "    Training RF/CatBoost/LightGBM..."
            )

            tree_train = (
                handcrafted_df
                .loc[
                    valid_train_subjects,
                    FEATURES
                ]
            )

            tree_val = (
                handcrafted_df
                .loc[
                    valid_val_subjects,
                    FEATURES
                ]
            )

            tree_imputer = (
                SimpleImputer(
                    strategy="median"
                )
            )

            X_tree_train = (
                tree_imputer.fit_transform(
                    tree_train
                )
            )

            X_tree_val = (
                tree_imputer.transform(
                    tree_val
                )
            )

            scaler = StandardScaler()

            X_tree_train = (
                scaler.fit_transform(
                    X_tree_train
                )
            )

            X_tree_val = (
                scaler.transform(
                    X_tree_val
                )
            )

            tree_models = (
                train_tree_models(
                    X_tree_train,
                    y_fusion_train
                )
            )

            rf_prob = (
                tree_models[0]
                .predict_proba(
                    X_tree_val
                )
            )

            cat_prob = (
                tree_models[1]
                .predict_proba(
                    X_tree_val
                )
            )

            lgb_prob = (
                tree_models[2]
                .predict_proba(
                    X_tree_val
                )
            )

            rf_prob = force_three_classes(
                rf_prob,
                tree_models[0]
            )

            cat_prob = force_three_classes(
                cat_prob,
                tree_models[1]
            )

            lgb_prob = force_three_classes(
                lgb_prob,
                tree_models[2]
            )

            # =================================================
            # STORE OOF BY SUBJECT ID
            # =================================================

            for i, sid in enumerate(
                valid_val_subjects
            ):

                if sid not in subject_to_row:

                    raise ValueError(
                        f"Unknown validation subject: {sid}"
                    )

                row = subject_to_row[sid]

                inner_meta[
                    row,
                    0:3
                ] = rf_prob[i]

                inner_meta[
                    row,
                    3:6
                ] = cat_prob[i]

                inner_meta[
                    row,
                    6:9
                ] = lgb_prob[i]

                inner_meta[
                    row,
                    9:12
                ] = fusion_prob[i]

            print(
                "    Genuine inner OOF predictions generated."
            )

            # -------------------------------------------------
            # Cleanup
            # -------------------------------------------------

            del cnn_model
            del embedding_model
            del fusion_model

            tf.keras.backend.clear_session()

        # ====================================================
        # VERIFY INNER OOF
        # ====================================================

        missing_mask = np.isnan(
            inner_meta
        ).any(
            axis=1
        )

        if np.any(missing_mask):

            missing_rows = np.where(
                missing_mask
            )[0]

            missing_subjects = (
                outer_train_subjects[
                    missing_rows
                ]
            )

            print(
                "\nERROR: Missing inner OOF predictions."
            )

            print(
                "Missing rows:",
                missing_rows.tolist()
            )

            print(
                "Missing subjects:",
                missing_subjects.tolist()
            )

            raise ValueError(
                "Some inner OOF predictions are missing."
            )

        print(
            "\n  Inner OOF matrix:",
            inner_meta.shape
        )

        print(
            "  All outer-training subjects have OOF predictions."
        )

        # ====================================================
        # META LEARNER
        # ====================================================

        print(
            "  Training Logistic Regression meta-learner..."
        )

        meta = LogisticRegression(
            max_iter=1000,
            C=0.5,
            random_state=SEED
        )

        meta.fit(
            inner_meta,
            y_outer_train
        )

        # ====================================================
        # OUTER CNN
        # ====================================================

        print(
            "  Training outer CNN..."
        )

        (
            outer_cnn,
            outer_embedding_model,
            outer_train_embeddings
        ) = train_cnn(
            outer_train_subjects.tolist(),
            subject_sequences,
            subject_labels,
            OUTER_EPOCHS
        )

        # ====================================================
        # OUTER TEST EMBEDDINGS
        # ====================================================

        X_test_seq, _, test_owners = (
            sequences_for_subjects(
                outer_test_subjects.tolist(),
                subject_sequences,
                subject_labels
            )
        )

        X_outer_train_seq, _, _ = (
            sequences_for_subjects(
                outer_train_subjects.tolist(),
                subject_sequences,
                subject_labels
            )
        )

        X_outer_train_seq, X_test_seq = (
            normalize_sequences(
                X_outer_train_seq,
                X_test_seq
            )
        )

        test_embeddings = (
            get_subject_embeddings(
                outer_embedding_model,
                X_test_seq,
                test_owners
            )
        )

        # ====================================================
        # OUTER FUSION
        # ====================================================

        print(
            "  Training outer fusion model..."
        )

        (
            X_fusion_train,
            valid_outer_train
        ) = build_fusion_matrix(
            outer_train_subjects,
            outer_train_embeddings,
            handcrafted_df
        )

        (
            X_fusion_test,
            valid_outer_test
        ) = build_fusion_matrix(
            outer_test_subjects,
            test_embeddings,
            handcrafted_df
        )

        y_fusion_train = np.asarray(
            [
                subject_labels[s]
                for s in valid_outer_train
            ]
        )

        y_fusion_test = np.asarray(
            [
                subject_labels[s]
                for s in valid_outer_test
            ]
        )

        fusion_imputer = (
            SimpleImputer(
                strategy="median"
            )
        )

        X_fusion_train = (
            fusion_imputer.fit_transform(
                X_fusion_train
            )
        )

        X_fusion_test = (
            fusion_imputer.transform(
                X_fusion_test
            )
        )

        fusion_outer = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=6,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            verbosity=-1
        )

        fusion_outer.fit(
            X_fusion_train,
            y_fusion_train
        )

        fusion_test_prob = (
            fusion_outer.predict_proba(
                X_fusion_test
            )
        )

        fusion_test_prob = (
            force_three_classes(
                fusion_test_prob,
                fusion_outer
            )
        )

        # ====================================================
        # OUTER TREE MODELS
        # ====================================================

        print(
            "  Training outer RF/CatBoost/LightGBM..."
        )

        tree_train = (
            handcrafted_df
            .loc[
                valid_outer_train,
                FEATURES
            ]
        )

        tree_test = (
            handcrafted_df
            .loc[
                valid_outer_test,
                FEATURES
            ]
        )

        tree_imputer = (
            SimpleImputer(
                strategy="median"
            )
        )

        X_tree_train = (
            tree_imputer.fit_transform(
                tree_train
            )
        )

        X_tree_test = (
            tree_imputer.transform(
                tree_test
            )
        )

        scaler = StandardScaler()

        X_tree_train = (
            scaler.fit_transform(
                X_tree_train
            )
        )

        X_tree_test = (
            scaler.transform(
                X_tree_test
            )
        )

        outer_tree_models = (
            train_tree_models(
                X_tree_train,
                y_fusion_train
            )
        )

        rf_test_prob = (
            outer_tree_models[0]
            .predict_proba(
                X_tree_test
            )
        )

        cat_test_prob = (
            outer_tree_models[1]
            .predict_proba(
                X_tree_test
            )
        )

        lgb_test_prob = (
            outer_tree_models[2]
            .predict_proba(
                X_tree_test
            )
        )

        rf_test_prob = force_three_classes(
            rf_test_prob,
            outer_tree_models[0]
        )

        cat_test_prob = force_three_classes(
            cat_test_prob,
            outer_tree_models[1]
        )

        lgb_test_prob = force_three_classes(
            lgb_test_prob,
            outer_tree_models[2]
        )

        # ====================================================
        # META TEST MATRIX
        # ====================================================

        outer_meta_test = np.column_stack(
            [
                rf_test_prob,
                cat_test_prob,
                lgb_test_prob,
                fusion_test_prob
            ]
        )

        # ====================================================
        # META PREDICTION
        # ====================================================

        meta_prob = (
            meta.predict_proba(
                outer_meta_test
            )
        )

        final_prob = np.zeros(
            (
                len(meta_prob),
                3
            )
        )

        for i, cls in enumerate(
            meta.classes_
        ):

            final_prob[:, int(cls)] = (
                meta_prob[:, i]
            )

        final_pred = np.argmax(
            final_prob,
            axis=1
        )

        # ====================================================
        # METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_fusion_test,
            final_pred
        )

        precision = precision_score(
            y_fusion_test,
            final_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_fusion_test,
            final_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_fusion_test,
            final_pred,
            average="weighted",
            zero_division=0
        )

        try:

            auc = roc_auc_score(
                y_fusion_test,
                final_prob,
                multi_class="ovr",
                average="weighted"
            )

        except Exception:

            auc = np.nan

        outer_results.append(
            {
                "fold": outer_fold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc
            }
        )

        # ====================================================
        # SAVE FOLD PREDICTIONS
        # ====================================================

        fold_predictions = pd.DataFrame(
            {
                "subject_id":
                    valid_outer_test,

                "true_label":
                    y_fusion_test,

                "pred_label":
                    final_pred,

                "p_healthy":
                    final_prob[:, 0],

                "p_depression":
                    final_prob[:, 1],

                "p_schizophrenia":
                    final_prob[:, 2],

                "fold":
                    outer_fold
            }
        )

        outer_predictions.append(
            fold_predictions
        )

        # ====================================================
        # PRINT FOLD RESULT
        # ====================================================

        print(
            "\n  OUTER FOLD RESULT"
        )

        print(
            f"  Accuracy : {accuracy:.4f}"
        )

        print(
            f"  Precision: {precision:.4f}"
        )

        print(
            f"  Recall   : {recall:.4f}"
        )

        print(
            f"  F1       : {f1:.4f}"
        )

        print(
            f"  AUC      : {auc:.4f}"
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del outer_cnn
        del outer_embedding_model
        del fusion_outer

        tf.keras.backend.clear_session()

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        outer_results
    )

    predictions_df = pd.concat(
        outer_predictions,
        ignore_index=True
    )

    predictions_df = (
        predictions_df
        .sort_values(
            "subject_id"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    results_df.to_csv(
        os.path.join(
            OUT_DIR,
            "full_nested_stepC_by_fold.csv"
        ),
        index=False
    )

    # ========================================================
    # SAVE OOF PREDICTIONS
    # ========================================================

    predictions_df.to_csv(
        os.path.join(
            OUT_DIR,
            "full_nested_stepC_oof_predictions.csv"
        ),
        index=False
    )

    # ========================================================
    # 5-FOLD SUMMARY
    # ========================================================

    summary_rows = []

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]:

        summary_rows.append(
            {
                "metric": metric,

                "mean":
                    results_df[
                        metric
                    ].mean(),

                "std":
                    results_df[
                        metric
                    ].std()
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        os.path.join(
            OUT_DIR,
            "full_nested_stepC_summary.csv"
        ),
        index=False
    )

    # ========================================================
    # OVERALL OOF
    # ========================================================

    overall_accuracy = accuracy_score(
        predictions_df["true_label"],
        predictions_df["pred_label"]
    )

    overall_precision = precision_score(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        average="weighted",
        zero_division=0
    )

    overall_recall = recall_score(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        average="weighted",
        zero_division=0
    )

    overall_f1 = f1_score(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        average="weighted",
        zero_division=0
    )

    try:

        overall_auc = roc_auc_score(
            predictions_df["true_label"],
            predictions_df[
                [
                    "p_healthy",
                    "p_depression",
                    "p_schizophrenia"
                ]
            ],
            multi_class="ovr",
            average="weighted"
        )

    except Exception:

        overall_auc = np.nan

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "FULL NESTED STEP C FINAL RESULTS"
    )

    print("=" * 70)

    print(
        "\n5-Fold Mean ± SD:"
    )

    for _, row in summary_df.iterrows():

        print(
            f"{row['metric']:10s}: "
            f"{row['mean']:.4f} ± "
            f"{row['std']:.4f}"
        )

    print(
        "\nOverall Strict OOF:"
    )

    print(
        f"Accuracy : {overall_accuracy:.4f}"
    )

    print(
        f"Precision: {overall_precision:.4f}"
    )

    print(
        f"Recall   : {overall_recall:.4f}"
    )

    print(
        f"F1       : {overall_f1:.4f}"
    )

    print(
        f"AUC      : {overall_auc:.4f}"
    )

    print(
        "\nSaved files:"
    )

    print(
        os.path.join(
            OUT_DIR,
            "full_nested_stepC_by_fold.csv"
        )
    )

    print(
        os.path.join(
            OUT_DIR,
            "full_nested_stepC_oof_predictions.csv"
        )
    )

    print(
        os.path.join(
            OUT_DIR,
            "full_nested_stepC_summary.csv"
        )
    )

    print(
        "\nFULL NESTED STEP C COMPLETE."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()