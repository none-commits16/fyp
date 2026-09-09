# ============================================================
# HYBRID FRAMEWORK - STEP B
# CNN Embedding + 27 Handcrafted Features -> LightGBM
#
# Uses:
#   Step A CNN models
#   Stage 3 enhanced 27-feature representation
#
# Evaluation:
#   Subject-level 5-fold CV
# ============================================================

import os
import glob
import ast
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import BorderlineSMOTE
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DEP = "data/depresjon"
PSY = "data/psykose"

CNN_DIR = "outputs/hybrid/stepA"
OUT = "outputs/hybrid/stepB"

os.makedirs(OUT, exist_ok=True)

BINS_PER_DAY = 96
MIN_COVERAGE = 0.50


# ============================================================
# 27 STAGE-3 FEATURES
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
    "lz_complexity",
]


# ============================================================
# LOAD EXACT FEATURE EXTRACTION FUNCTIONS FROM STAGE 3
# WITHOUT RUNNING ITS EXPERIMENTS
# ============================================================

def load_stage3_functions():

    path = "stage3_feature_enhancement.py"

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    namespace = {
        "__name__": "stage3_feature_module",
        "__file__": path,
    }

    # Keep imports, constants, and function/class definitions.
    # Do NOT execute the Stage-3 experiment section.
    allowed = []

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            )
        ):
            allowed.append(node)

        elif isinstance(node, ast.Assign):

            names = [
                t.id
                for t in node.targets
                if isinstance(t, ast.Name)
            ]

            if any(
                n in [
                    "SEED",
                    "DEP",
                    "PSY",
                    "BASE_FEATURES",
                    "ENHANCED",
                ]
                for n in names
            ):
                allowed.append(node)

    module = ast.Module(
        body=allowed,
        type_ignores=[]
    )

    ast.fix_missing_locations(module)

    exec(
        compile(
            module,
            path,
            "exec"
        ),
        namespace
    )

    return namespace


stage3 = load_stage3_functions()

extract = stage3["extract"]


# ============================================================
# LOAD SUBJECT RAW DATA
# ============================================================

def load_subjects():

    subjects = []

    # DEPRESJON depression
    for fp in sorted(
        glob.glob(
            os.path.join(
                DEP,
                "condition",
                "*.csv"
            )
        )
    ):

        subjects.append(
            {
                "subject_id":
                    "DEP_" +
                    os.path.splitext(
                        os.path.basename(fp)
                    )[0],

                "label": 1,
                "dataset": "DEPRESJON",
                "file": fp
            }
        )

    # DEPRESJON healthy
    for fp in sorted(
        glob.glob(
            os.path.join(
                DEP,
                "control",
                "*.csv"
            )
        )
    ):

        subjects.append(
            {
                "subject_id":
                    "DEP_" +
                    os.path.splitext(
                        os.path.basename(fp)
                    )[0],

                "label": 0,
                "dataset": "DEPRESJON",
                "file": fp
            }
        )

    # PSYKOSE schizophrenia
    for fp in sorted(
        glob.glob(
            os.path.join(
                PSY,
                "patient",
                "*.csv"
            )
        )
    ):

        subjects.append(
            {
                "subject_id":
                    "PSY_" +
                    os.path.splitext(
                        os.path.basename(fp)
                    )[0],

                "label": 2,
                "dataset": "PSYKOSE",
                "file": fp
            }
        )

    # PSYKOSE healthy
    for fp in sorted(
        glob.glob(
            os.path.join(
                PSY,
                "control",
                "*.csv"
            )
        )
    ):

        subjects.append(
            {
                "subject_id":
                    "PSY_" +
                    os.path.splitext(
                        os.path.basename(fp)
                    )[0],

                "label": 0,
                "dataset": "PSYKOSE",
                "file": fp
            }
        )

    return subjects


# ============================================================
# DAILY SEQUENCES
# Same representation as STEP A
# ============================================================

def make_daily_sequences(df):

    df = df.copy()

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
    )

    df = df.drop_duplicates(
        "timestamp"
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.set_index(
        "timestamp"
    )

    activity = df[
        "activity"
    ].resample("15min").mean()

    sequences = []

    for date, day in activity.groupby(
        activity.index.date
    ):

        coverage = (
            day.notna().sum()
            / BINS_PER_DAY
        )

        if coverage < MIN_COVERAGE:
            continue

        index = pd.date_range(
            start=pd.Timestamp(date),
            periods=BINS_PER_DAY,
            freq="15min"
        )

        day = day.reindex(index)

        day = day.interpolate(
            method="linear",
            limit_direction="both"
        )

        if day.isna().any():
            continue

        sequences.append(
            day.values.astype(
                np.float32
            )
        )

    return sequences


# ============================================================
# BUILD RAW SEQUENCES + FEATURES
# ============================================================

def build_data():

    subjects = load_subjects()

    sequences = []
    metadata = []
    feature_rows = []

    print(
        "\nBuilding Step-B dataset..."
    )

    for i, subject in enumerate(subjects):

        fp = subject["file"]

        df = pd.read_csv(fp)

        # ------------------------------
        # Stage-3 27 features
        # ------------------------------

        # Stage 3's extract() creates the
        # subject-level feature dictionary.
        #
        # It uses the original subject ID,
        # so we overwrite it with the
        # dataset-qualified ID used here.

        row = extract(
            fp,
            subject["label"],
            (
                "healthy"
                if subject["label"] == 0
                else
                "depression"
                if subject["label"] == 1
                else
                "schizophrenia"
            )
        )

        row["subject_id"] = subject[
            "subject_id"
        ]

        row["label"] = subject[
            "label"
        ]

        row["dataset"] = subject[
            "dataset"
        ]

        feature_rows.append(row)

        # ------------------------------
        # Daily sequences
        # ------------------------------

        seqs = make_daily_sequences(
            df
        )

        for j, seq in enumerate(seqs):

            sequences.append(seq)

            metadata.append(
                {
                    "subject_id":
                        subject["subject_id"],

                    "label":
                        subject["label"],

                    "sequence_id":
                        j
                }
            )

        print(
            f"{i+1:3d}/{len(subjects)} "
            f"{subject['subject_id']:15s} "
            f"days={len(seqs)}"
        )

    X_seq = np.array(
        sequences,
        dtype=np.float32
    )

    metadata = pd.DataFrame(
        metadata
    )

    features = pd.DataFrame(
        feature_rows
    )

    return (
        X_seq,
        metadata,
        features
    )


# ============================================================
# NORMALIZE CNN SEQUENCES
# ============================================================

def normalize_sequences(
    train,
    test
):

    mean = np.mean(train)
    std = np.std(train)

    if std < 1e-8:
        std = 1.0

    train = (
        train - mean
    ) / std

    test = (
        test - mean
    ) / std

    return train, test


# ============================================================
# GET CNN EMBEDDINGS
# ============================================================

def get_embeddings(
    model,
    X
):

    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(
            "cnn_embedding"
        ).output
    )

    return embedding_model.predict(
        X,
        batch_size=32,
        verbose=0
    )


# ============================================================
# SUBJECT-LEVEL EMBEDDING
# ============================================================

def aggregate_embeddings(
    embeddings,
    metadata
):

    temp = metadata.copy()

    for i in range(
        embeddings.shape[1]
    ):

        temp[
            f"emb_{i}"
        ] = embeddings[:, i]

    emb_cols = [
        c for c in temp.columns
        if c.startswith("emb_")
    ]

    result = (
        temp
        .groupby("subject_id")[emb_cols]
        .mean()
        .reset_index()
    )

    return result


# ============================================================
# METRICS
# ============================================================

def metrics(
    y_true,
    y_pred,
    prob
):

    result = {}

    result["accuracy"] = (
        accuracy_score(
            y_true,
            y_pred
        )
    )

    result["precision"] = (
        precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    result["recall"] = (
        recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    result["f1"] = (
        f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    try:

        result["auc"] = (
            roc_auc_score(
                y_true,
                prob,
                multi_class="ovr",
                average="weighted"
            )
        )

    except Exception:

        result["auc"] = np.nan

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "HYBRID FRAMEWORK - STEP B"
    )
    print(
        "CNN EMBEDDING + 27 FEATURES"
    )
    print("=" * 70)

    X_seq, metadata, features = (
        build_data()
    )

    # --------------------------------------------------------
    # SUBJECT INFORMATION
    # --------------------------------------------------------

    subject_info = features[
        [
            "subject_id",
            "label"
        ]
    ].drop_duplicates()

    subject_ids = (
        subject_info[
            "subject_id"
        ].values
    )

    labels = (
        subject_info[
            "label"
        ].values
    )

    print(
        "\nSubjects:",
        len(subject_ids)
    )

    print(
        "Sequences:",
        len(X_seq)
    )

    print(
        "Features:",
        len(FEATURES)
    )

    # --------------------------------------------------------
    # 5-FOLD SUBJECT CV
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    results = []
    all_predictions = []

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        skf.split(
            subject_ids,
            labels
        ),
        start=1
    ):

        print("\n")
        print("=" * 70)
        print(
            f"FOLD {fold}/5"
        )
        print("=" * 70)

        train_subjects = set(
            subject_ids[
                train_idx
            ]
        )

        test_subjects = set(
            subject_ids[
                test_idx
            ]
        )

        # ----------------------------------------------------
        # SEQUENCE SPLIT
        # ----------------------------------------------------

        train_mask = (
            metadata[
                "subject_id"
            ]
            .isin(train_subjects)
            .values
        )

        test_mask = (
            metadata[
                "subject_id"
            ]
            .isin(test_subjects)
            .values
        )

        X_train_seq = X_seq[
            train_mask
        ]

        X_test_seq = X_seq[
            test_mask
        ]

        train_meta = metadata.loc[
            train_mask
        ].copy()

        test_meta = metadata.loc[
            test_mask
        ].copy()

        print(
            "Train subjects:",
            len(train_subjects)
        )

        print(
            "Test subjects:",
            len(test_subjects)
        )

        # ----------------------------------------------------
        # LOAD CORRESPONDING STEP-A CNN
        # ----------------------------------------------------

        cnn_path = os.path.join(
            CNN_DIR,
            f"cnn_fold_{fold}.keras"
        )

        print(
            "Loading:",
            cnn_path
        )

        cnn = tf.keras.models.load_model(
            cnn_path,
            compile=False
        )

        # ----------------------------------------------------
        # TRAIN-ONLY NORMALIZATION
        # ----------------------------------------------------

        X_train_seq, X_test_seq = (
            normalize_sequences(
                X_train_seq,
                X_test_seq
            )
        )

        X_train_seq = (
            X_train_seq[..., np.newaxis]
        )

        X_test_seq = (
            X_test_seq[..., np.newaxis]
        )

        # ----------------------------------------------------
        # CNN EMBEDDINGS
        # ----------------------------------------------------

        train_emb = get_embeddings(
            cnn,
            X_train_seq
        )

        test_emb = get_embeddings(
            cnn,
            X_test_seq
        )

        train_emb_subject = (
            aggregate_embeddings(
                train_emb,
                train_meta
            )
        )

        test_emb_subject = (
            aggregate_embeddings(
                test_emb,
                test_meta
            )
        )

        # ----------------------------------------------------
        # HANDCRAFTED FEATURES
        # ----------------------------------------------------

        train_feat = features[
            features["subject_id"].isin(
                train_subjects
            )
        ][
            ["subject_id"] + FEATURES
        ].copy()

        test_feat = features[
            features["subject_id"].isin(
                test_subjects
            )
        ][
            ["subject_id"] + FEATURES
        ].copy()

        # ----------------------------------------------------
        # MERGE EMBEDDINGS + FEATURES
        # ----------------------------------------------------

        train_data = train_feat.merge(
            train_emb_subject,
            on="subject_id",
            how="inner"
        )

        test_data = test_feat.merge(
            test_emb_subject,
            on="subject_id",
            how="inner"
        )

        emb_cols = [
            c for c in train_data.columns
            if c.startswith("emb_")
        ]

        columns = FEATURES + emb_cols

        X_train = train_data[
            columns
        ].to_numpy(
            dtype=float
        )

        X_test = test_data[
            columns
        ].to_numpy(
            dtype=float
        )

        y_train = (
            train_data[
                "label"
            ]
            if "label" in train_data
            else features[
                features["subject_id"].isin(
                    train_subjects
                )
            ]["label"]
        )

        # Get labels directly from subject IDs
        label_map = dict(
            zip(
                features[
                    "subject_id"
                ],
                features[
                    "label"
                ]
            )
        )

        y_train = np.array([
            label_map[s]
            for s in train_data[
                "subject_id"
            ]
        ])

        y_test = np.array([
            label_map[s]
            for s in test_data[
                "subject_id"
            ]
        ])

        print(
            "Fusion dimensions:",
            X_train.shape[1]
        )

        # ----------------------------------------------------
        # IMPUTE + SCALE
        # TRAIN ONLY
        # ----------------------------------------------------

        imputer = SimpleImputer(
            strategy="median"
        )

        scaler = StandardScaler()

        X_train = imputer.fit_transform(
            X_train
        )

        X_test = imputer.transform(
            X_test
        )

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # BORDERLINE SMOTE
        # TRAIN ONLY
        # ----------------------------------------------------

        smote = BorderlineSMOTE(
            random_state=SEED,
            k_neighbors=3
        )

        X_train_res, y_train_res = (
            smote.fit_resample(
                X_train,
                y_train
            )
        )

        print(
            "After SMOTE:",
            len(X_train_res)
        )

        # ----------------------------------------------------
        # LIGHTGBM FUSION MODEL
        # ----------------------------------------------------

        model = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            verbosity=-1
        )

        model.fit(
            X_train_res,
            y_train_res
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        prob = model.predict_proba(
            X_test
        )

        pred = np.argmax(
            prob,
            axis=1
        )

        result = metrics(
            y_test,
            pred,
            prob
        )

        result["fold"] = fold
        result["model"] = (
            "CNN embedding + 27 features + LightGBM"
        )

        results.append(result)

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        fold_pred = pd.DataFrame(
            {
                "subject_id":
                    test_data[
                        "subject_id"
                    ],

                "true_label":
                    y_test,

                "pred_label":
                    pred,

                "p_healthy":
                    prob[:, 0],

                "p_depression":
                    prob[:, 1],

                "p_schizophrenia":
                    prob[:, 2],

                "fold":
                    fold
            }
        )

        all_predictions.append(
            fold_pred
        )

        print(
            f"Accuracy : {result['accuracy']:.4f}"
        )

        print(
            f"Precision: {result['precision']:.4f}"
        )

        print(
            f"Recall   : {result['recall']:.4f}"
        )

        print(
            f"F1       : {result['f1']:.4f}"
        )

        print(
            f"AUC      : {result['auc']:.4f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    pred_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    result_df.to_csv(
        os.path.join(
            OUT,
            "stepB_by_fold.csv"
        ),
        index=False
    )

    pred_df.to_csv(
        os.path.join(
            OUT,
            "stepB_subject_predictions.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

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
                    result_df[
                        metric
                    ].mean(),

                "std":
                    result_df[
                        metric
                    ].std()
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        os.path.join(
            OUT,
            "stepB_summary.csv"
        ),
        index=False
    )

    print("\n")
    print("=" * 70)
    print("STEP B FINAL RESULTS")
    print("=" * 70)

    for _, row in summary.iterrows():

        print(
            f"{row['metric']:10s}: "
            f"{row['mean']:.4f} ± "
            f"{row['std']:.4f}"
        )

    print("\nSaved to:")
    print(
        "outputs/hybrid/stepB/"
    )


if __name__ == "__main__":
    main()