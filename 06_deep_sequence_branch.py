import os
import glob
import numpy as np
import pandas as pd

# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

DEPRESJON_DIR = "data/depresjon"
PSYKOSE_DIR = "data/psykose"

print("DEPRESJON folder exists:", os.path.exists(DEPRESJON_DIR))
print("PSYKOSE folder exists:", os.path.exists(PSYKOSE_DIR))

# Find all CSV files
depresjon_files = glob.glob(
    os.path.join(DEPRESJON_DIR, "**", "*.csv"),
    recursive=True
)

psykose_files = glob.glob(
    os.path.join(PSYKOSE_DIR, "**", "*.csv"),
    recursive=True
)

print("\nNumber of DEPRESJON files:", len(depresjon_files))
print("Number of PSYKOSE files:", len(psykose_files))

print("\nFirst 5 DEPRESJON files:")
for file in depresjon_files[:5]:
    print(file)

print("\nFirst 5 PSYKOSE files:")
for file in psykose_files[:5]:
    print(file)

# ==========================================
# STEP 2: GET ONLY ACTUAL SUBJECT FILES
# ==========================================

# DEPRESJON
depresjon_condition_files = glob.glob(
    os.path.join(DEPRESJON_DIR, "data", "condition", "*.csv")
)

depresjon_control_files = glob.glob(
    os.path.join(DEPRESJON_DIR, "data", "control", "*.csv")
)


# PSYKOSE
psykose_control_files = glob.glob(
    os.path.join(PSYKOSE_DIR, "control", "*.csv")
)

# Check possible patient folder names
psykose_patient_files = glob.glob(
    os.path.join(PSYKOSE_DIR, "patient", "*.csv")
)

# Sometimes dataset may use another folder name
if len(psykose_patient_files) == 0:
    psykose_patient_files = glob.glob(
        os.path.join(PSYKOSE_DIR, "condition", "*.csv")
    )


print("\n" + "=" * 50)
print("ACTUAL SUBJECT FILES")
print("=" * 50)

print("\nDEPRESJON:")
print("Condition files:", len(depresjon_condition_files))
print("Control files:", len(depresjon_control_files))

print("\nPSYKOSE:")
print("Control files:", len(psykose_control_files))
print("Patient files:", len(psykose_patient_files))

print("\nExample files:")
if depresjon_condition_files:
    print("Depression condition:", depresjon_condition_files[0])

if depresjon_control_files:
    print("Depression control:", depresjon_control_files[0])

if psykose_control_files:
    print("Psychosis control:", psykose_control_files[0])

if psykose_patient_files:
    print("Psychosis patient:", psykose_patient_files[0])



# ==========================================
# STEP 3: INSPECT RAW ACTIVITY FILES
# ==========================================

def inspect_file(file_path, name):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    df_sample = pd.read_csv(file_path)

    print("File:", file_path)
    print("Shape:", df_sample.shape)
    print("\nColumns:")
    print(df_sample.columns.tolist())

    print("\nFirst 5 rows:")
    print(df_sample.head())

    print("\nMissing values:")
    print(df_sample.isnull().sum())


inspect_file(
    depresjon_condition_files[0],
    "DEPRESJON CONDITION SAMPLE"
)

inspect_file(
    depresjon_control_files[0],
    "DEPRESJON CONTROL SAMPLE"
)

inspect_file(
    psykose_patient_files[0],
    "PSYKOSE PATIENT SAMPLE"
)

inspect_file(
    psykose_control_files[0],
    "PSYKOSE CONTROL SAMPLE"
)

# ==========================================
# STEP 4: CREATE DAILY ACTIVITY SEQUENCES
# ==========================================

def create_daily_sequences(file_path, label, subject_id):
    """
    Convert one subject's minute-level activity data
    into 24-hour daily sequences.
    """

    df = pd.read_csv(file_path)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract calendar date from timestamp
    df["day"] = df["timestamp"].dt.date

    sequences = []
    labels = []
    subjects = []

    # Process each day separately
    for day, group in df.groupby("day"):

        # Sort chronologically
        group = group.sort_values("timestamp")

        activity = group["activity"].values.astype(np.float32)

        # We want exactly 1440 minutes = 24 hours
        # Skip incomplete days for now
        if len(activity) == 1440:

            sequences.append(activity)
            labels.append(label)
            subjects.append(subject_id)

    return sequences, labels, subjects


# ------------------------------------------
# LABEL DEFINITIONS
# ------------------------------------------

HEALTHY = 0
DEPRESSION = 1
SCHIZOPHRENIA = 2


all_sequences = []
all_labels = []
all_subjects = []


def process_files(file_list, label, dataset_name, group_name):

    for i, file_path in enumerate(file_list):

        subject_id = f"{dataset_name}_{group_name}_{i+1}"

        sequences, labels, subjects = create_daily_sequences(
            file_path=file_path,
            label=label,
            subject_id=subject_id
        )

        all_sequences.extend(sequences)
        all_labels.extend(labels)
        all_subjects.extend(subjects)


# DEPRESJON
process_files(
    depresjon_condition_files,
    DEPRESSION,
    "depresjon",
    "condition"
)

process_files(
    depresjon_control_files,
    HEALTHY,
    "depresjon",
    "control"
)


# PSYKOSE
process_files(
    psykose_patient_files,
    SCHIZOPHRENIA,
    "psykose",
    "patient"
)

process_files(
    psykose_control_files,
    HEALTHY,
    "psykose",
    "control"
)


# Convert to NumPy arrays
X = np.array(all_sequences, dtype=np.float32)
y = np.array(all_labels)
groups = np.array(all_subjects)


print("\n" + "=" * 60)
print("DAILY SEQUENCE DATASET")
print("=" * 60)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of unique subjects:", len(np.unique(groups)))

print("\nClass distribution by daily windows:")
unique, counts = np.unique(y, return_counts=True)

class_names = {
    HEALTHY: "Healthy",
    DEPRESSION: "Depression",
    SCHIZOPHRENIA: "Schizophrenia"
}

for cls, count in zip(unique, counts):
    print(f"{class_names[cls]}: {count}")

print("\nWindows per subject summary:")

unique_subjects, subject_counts = np.unique(
    groups,
    return_counts=True
)

print("Minimum:", subject_counts.min())
print("Maximum:", subject_counts.max())
print("Average:", round(subject_counts.mean(), 2))

# ==========================================
# STEP 5: SUBJECT-WISE STRATIFIED K-FOLD
# ==========================================

from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter

# CNN ke liye channel dimension add karo
X = X[..., np.newaxis]

print("\n" + "=" * 60)
print("CNN INPUT")
print("=" * 60)

print("X shape after channel addition:", X.shape)


# ------------------------------------------
# 5-FOLD SUBJECT-WISE CROSS VALIDATION
# ------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\n" + "=" * 60)
print("FOLD DISTRIBUTION CHECK")
print("=" * 60)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=groups),
    start=1
):

    y_train = y[train_idx]
    y_test = y[test_idx]

    train_subjects = np.unique(groups[train_idx])
    test_subjects = np.unique(groups[test_idx])

    print(f"\nFOLD {fold}")

    print("Train samples:", len(train_idx))
    print("Test samples:", len(test_idx))

    print("Train subjects:", len(train_subjects))
    print("Test subjects:", len(test_subjects))

    print("\nTrain class distribution:")

    for cls in [HEALTHY, DEPRESSION, SCHIZOPHRENIA]:
        print(
            f"  {class_names[cls]}: "
            f"{np.sum(y_train == cls)}"
        )

    print("\nTest class distribution:")

    for cls in [HEALTHY, DEPRESSION, SCHIZOPHRENIA]:
        print(
            f"  {class_names[cls]}: "
            f"{np.sum(y_test == cls)}"
        )

    # Safety check: koi subject overlap nahi hona chahiye
    overlap = set(train_subjects).intersection(set(test_subjects))

    print("\nSubject overlap:", len(overlap))

    assert len(overlap) == 0, "DATA LEAKAGE DETECTED!"

# ==========================================
# STEP 5B: SUBJECT-LEVEL FOLD VALIDATION
# ==========================================

# Each subject has only one class label
subject_labels = {}

for subject in np.unique(groups):
    subject_y = np.unique(y[groups == subject])

    # Safety check: one subject must belong to one class only
    assert len(subject_y) == 1, (
        f"Subject {subject} has multiple labels: {subject_y}"
    )

    subject_labels[subject] = subject_y[0]


print("\n" + "=" * 60)
print("SUBJECT-LEVEL CLASS DISTRIBUTION")
print("=" * 60)

for cls in [HEALTHY, DEPRESSION, SCHIZOPHRENIA]:

    subjects_in_class = [
        subject
        for subject, label in subject_labels.items()
        if label == cls
    ]

    print(
        f"{class_names[cls]} subjects: "
        f"{len(subjects_in_class)}"
    )

# ==========================================
# STEP 6: CREATE SUBJECT-LEVEL STRATIFIED FOLDS
# ==========================================

from sklearn.model_selection import StratifiedKFold

# Get one row per subject
unique_subjects = np.array(sorted(subject_labels.keys()))

subject_y = np.array([
    subject_labels[subject]
    for subject in unique_subjects
])

# 5-fold stratification at SUBJECT level
subject_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n" + "=" * 60)
print("SUBJECT-LEVEL STRATIFIED FOLD CHECK")
print("=" * 60)

subject_folds = []

for fold, (train_subject_idx, test_subject_idx) in enumerate(
    subject_cv.split(unique_subjects, subject_y),
    start=1
):

    train_subjects = unique_subjects[train_subject_idx]
    test_subjects = unique_subjects[test_subject_idx]

    # Convert subject split into daily-window indices
    train_idx = np.where(np.isin(groups, train_subjects))[0]
    test_idx = np.where(np.isin(groups, test_subjects))[0]

    subject_folds.append((train_idx, test_idx))

    print(f"\nFOLD {fold}")

    print("\nTest subjects by class:")

    for cls in [HEALTHY, DEPRESSION, SCHIZOPHRENIA]:

        count = np.sum(
            subject_y[test_subject_idx] == cls
        )

        print(f"  {class_names[cls]}: {count}")

    print("\nWindow distribution in test set:")

    for cls in [HEALTHY, DEPRESSION, SCHIZOPHRENIA]:

        count = np.sum(y[test_idx] == cls)

        print(f"  {class_names[cls]}: {count}")

    # Safety check
    overlap = set(train_subjects).intersection(
        set(test_subjects)
    )

    print("\nSubject overlap:", len(overlap))

    assert len(overlap) == 0, "DATA LEAKAGE DETECTED!"

# # ==========================================
# # STEP 7: CNN IMPORTS AND MODEL DEFINITION
# # ==========================================

# import tensorflow as tf

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import (
#     Input,
#     Conv1D,
#     BatchNormalization,
#     MaxPooling1D,
#     GlobalAveragePooling1D,
#     Dense,
#     Dropout
# )
# from tensorflow.keras.optimizers import Adam


# # Reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)


# def build_cnn(input_shape=(1440, 1), num_classes=3):
#     """
#     1D-CNN for minute-level daily activity sequences.

#     The embedding layer will later be used for
#     the Feature Fusion step.
#     """

#     inputs = Input(shape=input_shape, name="activity_input")

#     # CNN Block 1
#     x = Conv1D(
#         filters=32,
#         kernel_size=7,
#         padding="same",
#         activation="relu"
#     )(inputs)

#     x = BatchNormalization()(x)
#     x = MaxPooling1D(pool_size=2)(x)


#     # CNN Block 2
#     x = Conv1D(
#         filters=64,
#         kernel_size=5,
#         padding="same",
#         activation="relu"
#     )(x)

#     x = BatchNormalization()(x)
#     x = MaxPooling1D(pool_size=2)(x)


#     # CNN Block 3
#     x = Conv1D(
#         filters=128,
#         kernel_size=3,
#         padding="same",
#         activation="relu"
#     )(x)

#     x = BatchNormalization()(x)


#     # Convert sequence features into one vector
#     x = GlobalAveragePooling1D()(x)


#     # ------------------------------------------
#     # PENULTIMATE LAYER = DEEP EMBEDDING
#     # ------------------------------------------

#     embedding = Dense(
#         64,
#         activation="relu",
#         name="embedding"
#     )(x)

#     x = Dropout(0.3)(embedding)


#     # Final classification layer
#     outputs = Dense(
#         num_classes,
#         activation="softmax",
#         name="classification"
#     )(x)


#     model = Model(
#         inputs=inputs,
#         outputs=outputs,
#         name="Activity_1D_CNN"
#     )

#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )

#     return model


# # Build model once just to verify architecture
# cnn_model = build_cnn()

# print("\n" + "=" * 60)
# print("1D-CNN MODEL SUMMARY")
# print("=" * 60)

# cnn_model.summary()


# # ==========================================
# # STEP 8: CNN TRAINING AND EVALUATION
# # ==========================================

# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     roc_auc_score,
#     classification_report
# )
# from sklearn.utils.class_weight import compute_class_weight


# # X is already in CNN format: (samples, 1440, 1)
# # groups contains subject IDs
# print("\n" + "=" * 60)
# print("CNN TRAINING SETUP")
# print("=" * 60)

# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("Number of subjects:", len(np.unique(groups)))


# # ==========================================
# # STORE CROSS-VALIDATION RESULTS
# # ==========================================

# fold_results = []

# all_true = []
# all_pred = []
# all_prob = []


# # ==========================================
# # TRAIN CNN USING PRE-CREATED SUBJECT FOLDS
# # ==========================================

# for fold, (train_idx, test_idx) in enumerate(
#     subject_folds,
#     start=1
# ):

#     print("\n" + "=" * 60)
#     print(f"TRAINING FOLD {fold}")
#     print("=" * 60)


#     # ------------------------------------------
#     # 1. SPLIT DATA
#     # ------------------------------------------

#     X_train = X[train_idx].copy()
#     X_test = X[test_idx].copy()

#     y_train = y[train_idx]
#     y_test = y[test_idx]


#     # ------------------------------------------
#     # 2. NORMALIZATION
#     # TRAINING DATA ONLY
#     # ------------------------------------------

#     train_mean = X_train.mean()
#     train_std = X_train.std()

#     X_train = (
#         X_train - train_mean
#     ) / (train_std + 1e-8)

#     X_test = (
#         X_test - train_mean
#     ) / (train_std + 1e-8)


#     # ------------------------------------------
#     # 3. CLASS WEIGHTS
#     # ------------------------------------------

#     classes = np.unique(y_train)

#     weights = compute_class_weight(
#         class_weight="balanced",
#         classes=classes,
#         y=y_train
#     )

#     class_weights = dict(zip(classes, weights))

#     print("\nClass weights:")
#     print(class_weights)


#     # ------------------------------------------
#     # 4. CLEAR OLD MODEL
#     # ------------------------------------------

#     tf.keras.backend.clear_session()


#     # ------------------------------------------
#     # 5. BUILD NEW CNN
#     # ------------------------------------------

#     model = build_cnn(
#         input_shape=(1440, 1),
#         num_classes=3
#     )


#     # ------------------------------------------
#     # 6. EARLY STOPPING
#     # ------------------------------------------

#     early_stopping = EarlyStopping(
#         monitor="val_loss",
#         patience=8,
#         restore_best_weights=True,
#         verbose=1
#     )


#     # ------------------------------------------
#     # 7. TRAIN MODEL
#     # ------------------------------------------

#     history = model.fit(
#         X_train,
#         y_train,
#         validation_split=0.15,
#         epochs=50,
#         batch_size=32,
#         class_weight=class_weights,
#         callbacks=[early_stopping],
#         verbose=1
#     )


#     # ------------------------------------------
#     # 8. PREDICTIONS
#     # ------------------------------------------

#     y_prob = model.predict(
#         X_test,
#         verbose=0
#     )

#     y_pred = np.argmax(
#         y_prob,
#         axis=1
#     )


#     # ------------------------------------------
#     # 9. METRICS
#     # ------------------------------------------

#     accuracy = accuracy_score(
#         y_test,
#         y_pred
#     )

#     precision, recall, f1, _ = (
#         precision_recall_fscore_support(
#             y_test,
#             y_pred,
#             average="weighted",
#             zero_division=0
#         )
#     )


#     # ------------------------------------------
#     # 10. MULTI-CLASS AUC
#     # ------------------------------------------

#     auc = roc_auc_score(
#         y_test,
#         y_prob,
#         multi_class="ovr",
#         average="weighted"
#     )


#     # ------------------------------------------
#     # 11. PRINT FOLD RESULTS
#     # ------------------------------------------

#     print("\nFOLD RESULTS")

#     print(f"Accuracy : {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall   : {recall:.4f}")
#     print(f"F1 Score : {f1:.4f}")
#     print(f"AUC      : {auc:.4f}")


#     # ------------------------------------------
#     # 12. SAVE FOLD RESULTS
#     # ------------------------------------------

#     fold_results.append({
#         "fold": fold,
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "auc": auc
#     })


#     # ------------------------------------------
#     # 13. STORE OUT-OF-FOLD PREDICTIONS
#     # ------------------------------------------

#     all_true.extend(y_test)
#     all_pred.extend(y_pred)
#     all_prob.extend(y_prob)


# # ==========================================
# # STEP 9: FINAL CROSS-VALIDATION RESULTS
# # ==========================================

# results_df = pd.DataFrame(
#     fold_results
# )

# print("\n" + "=" * 60)
# print("CNN 5-FOLD CROSS-VALIDATION RESULTS")
# print("=" * 60)

# print(results_df)


# print("\nMEAN RESULTS")

# for metric in [
#     "accuracy",
#     "precision",
#     "recall",
#     "f1",
#     "auc"
# ]:

#     mean_score = results_df[metric].mean()
#     std_score = results_df[metric].std()

#     print(
#         f"{metric.upper():<10}: "
#         f"{mean_score:.4f} ± {std_score:.4f}"
#     )


# # ==========================================
# # STEP 10: SAVE RESULTS
# # ==========================================

# os.makedirs(
#     "outputs/deep_learning",
#     exist_ok=True
# )

# results_df.to_csv(
#     "outputs/deep_learning/cnn_fold_results.csv",
#     index=False
# )

# print(
#     "\nSaved: "
#     "outputs/deep_learning/cnn_fold_results.csv"
# )


# # ==========================================
# # STEP 11: OVERALL CLASSIFICATION REPORT
# # ==========================================

# print("\n" + "=" * 60)
# print("OVERALL CLASSIFICATION REPORT")
# print("=" * 60)

# print(
#     classification_report(
#         all_true,
#         all_pred,
#         target_names=[
#             "Healthy",
#             "Depression",
#             "Schizophrenia"
#         ],
#         zero_division=0
#     )
# )


import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


np.random.seed(42)
tf.random.set_seed(42)


def build_cnn(input_shape=(1440, 1), num_classes=3):

    inputs = Input(
        shape=input_shape,
        name="activity_input"
    )

    # Block 1
    x = Conv1D(
        32,
        kernel_size=7,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(inputs)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)


    # Block 2
    x = Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)


    # Block 3
    x = Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)


    # Sequence representation
    x = GlobalAveragePooling1D()(x)


    # Deep embedding
    embedding = Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="embedding"
    )(x)

    x = Dropout(0.4)(embedding)


    outputs = Dense(
        num_classes,
        activation="softmax",
        name="classification"
    )(x)


    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Activity_1D_CNN"
    )


    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight


fold_results = []

all_true = []
all_pred = []
all_prob = []

# PART B - CNN embeddings store karne ke liye
all_embeddings = []
all_embedding_subjects = []
all_embedding_labels = []
all_embedding_folds = []

# ============================================================
# TRAIN USING SUBJECT-LEVEL FOLDS
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    subject_folds,
    start=1
):

    print("\n" + "=" * 70)
    print(f"TRAINING FOLD {fold}")
    print("=" * 70)


    # --------------------------------------------------------
    # 1. OUTER TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_outer_train = X[train_idx].copy()
    y_outer_train = y[train_idx]

    X_test = X[test_idx].copy()
    y_test = y[test_idx]

    train_subject_ids = groups[train_idx]


    # --------------------------------------------------------
    # 2. CREATE SUBJECT-WISE VALIDATION SPLIT
    # --------------------------------------------------------

    unique_train_subjects = np.unique(train_subject_ids)

    subject_labels_train = []

    for subject in unique_train_subjects:

        subject_label = np.unique(
            y_outer_train[
                train_subject_ids == subject
            ]
        )

        subject_labels_train.append(
            subject_label[0]
        )

    subject_labels_train = np.array(
        subject_labels_train
    )


    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=42
    )


    train_subject_index, val_subject_index = next(
        splitter.split(
            unique_train_subjects,
            subject_labels_train
        )
    )


    final_train_subjects = (
        unique_train_subjects[
            train_subject_index
        ]
    )

    val_subjects = (
        unique_train_subjects[
            val_subject_index
        ]
    )


    train_mask = np.isin(
        train_subject_ids,
        final_train_subjects
    )

    val_mask = np.isin(
        train_subject_ids,
        val_subjects
    )


    X_train = X_outer_train[
        train_mask
    ].copy()

    y_train = y_outer_train[
        train_mask
    ]


    X_val = X_outer_train[
        val_mask
    ].copy()

    y_val = y_outer_train[
        val_mask
    ]


    print("\nSubject-wise split:")

    print(
        "Training subjects:",
        len(final_train_subjects)
    )

    print(
        "Validation subjects:",
        len(val_subjects)
    )

    print(
        "Test subjects:",
        len(np.unique(groups[test_idx]))
    )


    # Safety check
    overlap = set(final_train_subjects).intersection(
        set(val_subjects)
    )

    assert len(overlap) == 0


    # --------------------------------------------------------
    # 3. PER-SEQUENCE NORMALIZATION
    # --------------------------------------------------------

    def normalize_sequences(data):

        data = data.copy()

        mean = np.mean(
            data,
            axis=1,
            keepdims=True
        )

        std = np.std(
            data,
            axis=1,
            keepdims=True
        )

        return (
            data - mean
        ) / (
            std + 1e-8
        )


    X_train = normalize_sequences(
        X_train
    )

    X_val = normalize_sequences(
        X_val
    )

    X_test = normalize_sequences(
        X_test
    )


    # --------------------------------------------------------
    # 4. CLASS WEIGHTS
    # --------------------------------------------------------

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = dict(
        zip(classes, weights)
    )

    print("\nClass weights:")
    print(class_weights)


    # --------------------------------------------------------
    # 5. CLEAR SESSION
    # --------------------------------------------------------

    tf.keras.backend.clear_session()


    # --------------------------------------------------------
    # 6. BUILD MODEL
    # --------------------------------------------------------

    model = build_cnn(
        input_shape=(1440, 1),
        num_classes=3
    )


    # --------------------------------------------------------
    # 7. CALLBACKS
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )


    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )


    # --------------------------------------------------------
    # 8. TRAIN
    # --------------------------------------------------------

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=50,

        batch_size=32,

        class_weight=class_weights,

        callbacks=[
            early_stopping,
            reduce_lr
        ],

        verbose=1
    )


    # --------------------------------------------------------
    # 9. PREDICTION
    # --------------------------------------------------------

    y_prob = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_prob,
        axis=1
    )


    # --------------------------------------------------------
    # 10. METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )


    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )
    )


    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="macro"
    )


    # --------------------------------------------------------
    # 11. PRINT RESULTS
    # --------------------------------------------------------

    print("\nFOLD RESULTS")

    print(
        f"Accuracy             : {accuracy:.4f}"
    )

    print(
        f"Weighted Precision   : {precision_weighted:.4f}"
    )

    print(
        f"Weighted Recall      : {recall_weighted:.4f}"
    )

    print(
        f"Weighted F1          : {f1_weighted:.4f}"
    )

    print(
        f"Macro Precision      : {precision_macro:.4f}"
    )

    print(
        f"Macro Recall         : {recall_macro:.4f}"
    )

    print(
        f"Macro F1             : {f1_macro:.4f}"
    )

    print(
        f"Macro AUC            : {auc:.4f}"
    )


    # --------------------------------------------------------
    # 12. SAVE FOLD RESULT
    # --------------------------------------------------------

    fold_results.append({

        "fold": fold,

        "accuracy": accuracy,

        "weighted_precision":
            precision_weighted,

        "weighted_recall":
            recall_weighted,

        "weighted_f1":
            f1_weighted,

        "macro_precision":
            precision_macro,

        "macro_recall":
            recall_macro,

        "macro_f1":
            f1_macro,

        "macro_auc":
            auc
    })


    # --------------------------------------------------------
    # 13. SAVE OOF PREDICTIONS
    # --------------------------------------------------------

    all_true.extend(
        y_test
    )

    all_pred.extend(
        y_pred
    )

    all_prob.extend(
        y_prob
    )

results_df = pd.DataFrame(
    fold_results
)

print("\n" + "=" * 70)
print("CNN 5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)

print(results_df)


print("\nMEAN RESULTS")

for metric in results_df.columns:

    if metric != "fold":

        mean_score = results_df[
            metric
        ].mean()

        std_score = results_df[
            metric
        ].std()

        print(
            f"{metric.upper():<25}: "
            f"{mean_score:.4f} ± {std_score:.4f}"
        )


os.makedirs(
    "outputs/deep_learning",
    exist_ok=True
)


results_df.to_csv(
    "outputs/deep_learning/cnn_fold_results.csv",
    index=False
)


print(
    "\nSaved: "
    "outputs/deep_learning/cnn_fold_results.csv"
)


print("\n" + "=" * 70)
print("OVERALL CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        all_true,
        all_pred,
        target_names=[
            "Healthy",
            "Depression",
            "Schizophrenia"
        ],
        zero_division=0
    )
)


print("\nCONFUSION MATRIX")

cm = confusion_matrix(
    all_true,
    all_pred
)

print(cm)