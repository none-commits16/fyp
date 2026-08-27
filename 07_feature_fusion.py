# # ============================================================
# # 07 — FEATURE FUSION (PART B)
# # CNN/LSTM Embeddings + Handcrafted Features
# # ============================================================

# import os
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import StratifiedGroupKFold
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
#     confusion_matrix
# )

# from catboost import CatBoostClassifier


# # ============================================================
# # SETTINGS
# # ============================================================

# SEED = 42

# DEP_PATH = "data/depresjon"
# PSY_PATH = "data/psykose"

# OUT_DIR = "outputs/stage4_feature_fusion"
# os.makedirs(OUT_DIR, exist_ok=True)


# print("=" * 70)
# print("STAGE 4 — FEATURE FUSION (PART B)")
# print("=" * 70)

# print("\nOutput directory:")
# print(OUT_DIR)

# # ============================================================
# # LOAD RAW SUBJECT FILES AND CREATE DAILY SEQUENCES
# # ============================================================

# import glob


# def get_daily_sequences(file_path, label, subject_id):
#     """
#     Read one subject file and convert activity data
#     into one 1440-minute sequence per day.
#     """

#     df = pd.read_csv(file_path)

#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df["date"] = pd.to_datetime(df["date"])

#     sequences = []
#     labels = []
#     subjects = []

#     # Process each day separately
#     for date, day_df in df.groupby("date"):

#         # Create a full 1440-minute day
#         day_df = day_df.copy()
#         day_df["minute_of_day"] = (
#             day_df["timestamp"].dt.hour * 60
#             + day_df["timestamp"].dt.minute
#         )

#         sequence = np.zeros(1440)

#         minutes = day_df["minute_of_day"].to_numpy()
#         activities = day_df["activity"].to_numpy()

#         valid = (minutes >= 0) & (minutes < 1440)

#         sequence[minutes[valid]] = activities[valid]

#         sequences.append(sequence)
#         labels.append(label)
#         subjects.append(subject_id)

#     return sequences, labels, subjects


# X_list = []
# y_list = []
# subject_ids = []


# # ---------------- DEPRESJON ----------------

# print("\nLoading DEPRESJON subjects...")

# for file_path in sorted(
#     glob.glob(os.path.join(DEP_PATH, "data", "condition", "*.csv"))
# ):
#     subject_id = "dep_condition_" + os.path.splitext(
#         os.path.basename(file_path)
#     )[0]

#     sequences, labels, subjects = get_daily_sequences(
#         file_path,
#         label=1,
#         subject_id=subject_id
#     )

#     X_list.extend(sequences)
#     y_list.extend(labels)
#     subject_ids.extend(subjects)


# for file_path in sorted(
#     glob.glob(os.path.join(DEP_PATH, "data", "control", "*.csv"))
# ):
#     subject_id = "dep_control_" + os.path.splitext(
#         os.path.basename(file_path)
#     )[0]

#     sequences, labels, subjects = get_daily_sequences(
#         file_path,
#         label=0,
#         subject_id=subject_id
#     )

#     X_list.extend(sequences)
#     y_list.extend(labels)
#     subject_ids.extend(subjects)


# # ---------------- PSYKOSE ----------------

# print("Loading PSYKOSE subjects...")

# for file_path in sorted(
#     glob.glob(os.path.join(PSY_PATH, "patient", "*.csv"))
# ):
#     subject_id = "psy_patient_" + os.path.splitext(
#         os.path.basename(file_path)
#     )[0]

#     sequences, labels, subjects = get_daily_sequences(
#         file_path,
#         label=2,
#         subject_id=subject_id
#     )

#     X_list.extend(sequences)
#     y_list.extend(labels)
#     subject_ids.extend(subjects)


# for file_path in sorted(
#     glob.glob(os.path.join(PSY_PATH, "control", "*.csv"))
# ):
#     subject_id = "psy_control_" + os.path.splitext(
#         os.path.basename(file_path)
#     )[0]

#     sequences, labels, subjects = get_daily_sequences(
#         file_path,
#         label=0,
#         subject_id=subject_id
#     )

#     X_list.extend(sequences)
#     y_list.extend(labels)
#     subject_ids.extend(subjects)


# # Convert to NumPy arrays
# X = np.array(X_list, dtype=np.float32)
# y = np.array(y_list)
# subject_ids = np.array(subject_ids)

# print("\n" + "=" * 70)
# print("DAILY SEQUENCE DATASET")
# print("=" * 70)

# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("Unique subjects:", len(np.unique(subject_ids)))

# print("\nClass distribution:")
# print("Healthy:", np.sum(y == 0))
# print("Depression:", np.sum(y == 1))
# print("Schizophrenia:", np.sum(y == 2))

# # ============================================================
# # CNN MODEL FOR SEQUENCE EMBEDDING
# # ============================================================

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Conv1D,
#     MaxPooling1D,
#     GlobalAveragePooling1D,
#     Dense,
#     Dropout
# )
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import StratifiedGroupKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score

# # CNN ke liye channel dimension add karo
# X_cnn = X[..., np.newaxis]

# print("\nCNN input shape:", X_cnn.shape)

# # ============================================================
# # BUILD CNN MODEL
# # ============================================================

# def build_cnn_model():
    
#     model = Sequential([
        
#         Conv1D(
#             filters=32,
#             kernel_size=7,
#             activation="relu",
#             input_shape=(1440, 1)
#         ),
        
#         MaxPooling1D(pool_size=2),
        
#         Conv1D(
#             filters=64,
#             kernel_size=5,
#             activation="relu"
#         ),
        
#         MaxPooling1D(pool_size=2),
        
#         Conv1D(
#             filters=128,
#             kernel_size=3,
#             activation="relu"
#         ),
        
#         GlobalAveragePooling1D(),
        
#         Dense(
#             64,
#             activation="relu",
#             name="embedding"
#         ),
        
#         Dropout(0.3),
        
#         Dense(
#             3,
#             activation="softmax",
#             name="classification"
#         )
#     ])
    
#     model.compile(
#         optimizer="adam",
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )
    
#     return model


# # Create model
# cnn_model = build_cnn_model()

# print("\n" + "=" * 70)
# print("CNN MODEL SUMMARY")
# print("=" * 70)

# cnn_model.summary()

# # ============================================================
# # TRAIN CNN USING SUBJECT-WISE SPLIT
# # ============================================================

# from sklearn.model_selection import StratifiedGroupKFold

# cv = StratifiedGroupKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )

# # Pehla fold use kar rahe hain abhi
# train_idx, test_idx = next(
#     cv.split(X_cnn, y, groups=subject_ids)
# )

# X_train = X_cnn[train_idx]
# X_test = X_cnn[test_idx]

# y_train = y[train_idx]
# y_test = y[test_idx]

# print("\n" + "=" * 70)
# print("CNN TRAIN / TEST SPLIT")
# print("=" * 70)

# print("Train samples:", len(X_train))
# print("Test samples:", len(X_test))

# print("Train subjects:", len(np.unique(subject_ids[train_idx])))
# print("Test subjects:", len(np.unique(subject_ids[test_idx])))

# print(
#     "Subject overlap:",
#     len(
#         set(subject_ids[train_idx]) &
#         set(subject_ids[test_idx])
#     )
# )

# # Build fresh CNN
# cnn_model = build_cnn_model()

# # ============================================================
# # TRAIN CNN
# # ============================================================

# print("\nTraining CNN...\n")

# history = cnn_model.fit(
#     X_train,
#     y_train,
#     epochs=10,
#     batch_size=32,
#     verbose=1
# )

# print("\nCNN TRAINING COMPLETED")

# # ============================================================
# # EXTRACT CNN EMBEDDINGS
# # ============================================================

# from tensorflow.keras.models import Model

# print("\n" + "=" * 70)
# print("EXTRACTING CNN EMBEDDINGS")
# print("=" * 70)

# embedding_model = Model(
#     inputs=cnn_model.input,
#     outputs=cnn_model.get_layer("embedding").output
# )

# train_embeddings = embedding_model.predict(X_train, verbose=0)
# test_embeddings = embedding_model.predict(X_test, verbose=0)

# print("Train embedding shape:", train_embeddings.shape)
# print("Test embedding shape:", test_embeddings.shape)






# ============================================================
# 07 — FEATURE FUSION (PART B)
# CNN EMBEDDINGS + HANDCRAFTED FEATURES
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# MACHINE LEARNING IMPORTS
# ============================================================

from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# TENSORFLOW / KERAS IMPORTS
# ============================================================

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

DEP_PATH = "data/depresjon"
PSY_PATH = "data/psykose"

OUT_DIR = "outputs/stage4_feature_fusion"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# START MESSAGE
# ============================================================

print("=" * 70)
print("STAGE 4 — FEATURE FUSION (PART B)")
print("=" * 70)

print("\nOutput directory:")
print(OUT_DIR)


# ============================================================
# FUNCTION:
# CONVERT ONE SUBJECT INTO DAILY 1440-MINUTE SEQUENCES
# ============================================================

def get_daily_sequences(file_path, label, subject_id):

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "date", "activity"]
    )

    sequences = []
    labels = []
    subjects = []

    # Process every day separately
    for date, day_df in df.groupby("date"):

        day_df = day_df.copy()

        # Convert timestamp into minute number
        day_df["minute_of_day"] = (
            day_df["timestamp"].dt.hour * 60
            + day_df["timestamp"].dt.minute
        )

        # Full day = 1440 minutes
        sequence = np.zeros(
            1440,
            dtype=np.float32
        )

        minutes = day_df["minute_of_day"].to_numpy()

        activities = (
            day_df["activity"]
            .to_numpy(dtype=np.float32)
        )

        valid = (
            (minutes >= 0)
            &
            (minutes < 1440)
        )

        sequence[
            minutes[valid]
        ] = activities[valid]

        sequences.append(sequence)

        labels.append(label)

        subjects.append(subject_id)

    return sequences, labels, subjects


# ============================================================
# LOAD ALL SUBJECTS
# ============================================================

X_list = []

y_list = []

subject_ids = []


# ============================================================
# DEPRESJON DATASET
# ============================================================

print("\nLoading DEPRESJON subjects...")


# ----------------------------
# Depression patients
# ----------------------------

for file_path in sorted(
    glob.glob(
        os.path.join(
            DEP_PATH,
            "data",
            "condition",
            "*.csv"
        )
    )
):

    subject_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    subject_id = (
        "dep_condition_"
        + subject_name
    )

    sequences, labels, subjects = (
        get_daily_sequences(
            file_path=file_path,
            label=1,
            subject_id=subject_id
        )
    )

    X_list.extend(sequences)

    y_list.extend(labels)

    subject_ids.extend(subjects)


# ----------------------------
# Healthy controls
# ----------------------------

for file_path in sorted(
    glob.glob(
        os.path.join(
            DEP_PATH,
            "data",
            "control",
            "*.csv"
        )
    )
):

    subject_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    subject_id = (
        "dep_control_"
        + subject_name
    )

    sequences, labels, subjects = (
        get_daily_sequences(
            file_path=file_path,
            label=0,
            subject_id=subject_id
        )
    )

    X_list.extend(sequences)

    y_list.extend(labels)

    subject_ids.extend(subjects)


# ============================================================
# PSYKOSE DATASET
# ============================================================

print("Loading PSYKOSE subjects...")


# ----------------------------
# Schizophrenia patients
# ----------------------------

for file_path in sorted(
    glob.glob(
        os.path.join(
            PSY_PATH,
            "patient",
            "*.csv"
        )
    )
):

    subject_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    subject_id = (
        "psy_patient_"
        + subject_name
    )

    sequences, labels, subjects = (
        get_daily_sequences(
            file_path=file_path,
            label=2,
            subject_id=subject_id
        )
    )

    X_list.extend(sequences)

    y_list.extend(labels)

    subject_ids.extend(subjects)


# ----------------------------
# Healthy controls
# ----------------------------

for file_path in sorted(
    glob.glob(
        os.path.join(
            PSY_PATH,
            "control",
            "*.csv"
        )
    )
):

    subject_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    subject_id = (
        "psy_control_"
        + subject_name
    )

    sequences, labels, subjects = (
        get_daily_sequences(
            file_path=file_path,
            label=0,
            subject_id=subject_id
        )
    )

    X_list.extend(sequences)

    y_list.extend(labels)

    subject_ids.extend(subjects)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(
    X_list,
    dtype=np.float32
)

y = np.array(
    y_list,
    dtype=np.int32
)

subject_ids = np.array(
    subject_ids
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DAILY SEQUENCE DATASET")
print("=" * 70)

print("X shape:", X.shape)

print("y shape:", y.shape)

print(
    "Unique subjects:",
    len(np.unique(subject_ids))
)

print("\nClass distribution:")

print(
    "Healthy:",
    np.sum(y == 0)
)

print(
    "Depression:",
    np.sum(y == 1)
)

print(
    "Schizophrenia:",
    np.sum(y == 2)
)


# ============================================================
# ADD CHANNEL DIMENSION FOR CNN
# ============================================================

X_cnn = X[..., np.newaxis]

print("\nCNN input shape:", X_cnn.shape)


# ============================================================
# BUILD CNN MODEL
# ============================================================

def build_cnn_model():

    model = Sequential([

        # Explicit input layer
        Input(
            shape=(1440, 1)
        ),

        # CNN Block 1
        Conv1D(
            filters=32,
            kernel_size=7,
            activation="relu"
        ),

        MaxPooling1D(
            pool_size=2
        ),

        # CNN Block 2
        Conv1D(
            filters=64,
            kernel_size=5,
            activation="relu"
        ),

        MaxPooling1D(
            pool_size=2
        ),

        # CNN Block 3
        Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu"
        ),

        # Convert sequence into feature vector
        GlobalAveragePooling1D(),

        # ====================================================
        # PENULTIMATE LAYER
        # THIS IS OUR CNN EMBEDDING
        # ====================================================

        Dense(
            64,
            activation="relu",
            name="embedding"
        ),

        Dropout(
            0.3
        ),

        # Final classification layer
        Dense(
            3,
            activation="softmax",
            name="classification"
        )

    ])

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )

    return model


# ============================================================
# CREATE CNN MODEL
# ============================================================

cnn_model = build_cnn_model()


print("\n" + "=" * 70)
print("CNN MODEL SUMMARY")
print("=" * 70)

cnn_model.summary()


# ============================================================
# SUBJECT-WISE TRAIN / TEST SPLIT
# ============================================================

cv = StratifiedGroupKFold(

    n_splits=5,

    shuffle=True,

    random_state=SEED

)


# Use first fold

train_idx, test_idx = next(

    cv.split(

        X_cnn,

        y,

        groups=subject_ids

    )

)


# ============================================================
# CREATE TRAIN / TEST DATA
# ============================================================

X_train = X_cnn[train_idx]

X_test = X_cnn[test_idx]

y_train = y[train_idx]

y_test = y[test_idx]


train_subject_ids = subject_ids[
    train_idx
]

test_subject_ids = subject_ids[
    test_idx
]


# ============================================================
# SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("CNN TRAIN / TEST SPLIT")
print("=" * 70)

print(
    "Train samples:",
    len(X_train)
)

print(
    "Test samples:",
    len(X_test)
)

print(
    "Train subjects:",
    len(np.unique(train_subject_ids))
)

print(
    "Test subjects:",
    len(np.unique(test_subject_ids))
)

overlap = len(

    set(train_subject_ids)

    &

    set(test_subject_ids)

)

print(
    "Subject overlap:",
    overlap
)


# ============================================================
# SAFETY CHECK
# ============================================================

if overlap == 0:

    print(
        "\n✓ Subject-wise split is correct."
    )

else:

    print(
        "\nWARNING: SUBJECT LEAKAGE DETECTED!"
    )


# ============================================================
# TRAIN CNN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CNN")
print("=" * 70)


history = cnn_model.fit(

    X_train,

    y_train,

    epochs=10,

    batch_size=32,

    verbose=1

)


print("\n" + "=" * 70)
print("CNN TRAINING COMPLETED")
print("=" * 70)


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING CNN EMBEDDINGS")
print("=" * 70)


embedding_model = Model(

    inputs=cnn_model.inputs,

    outputs=cnn_model.get_layer(
        "embedding"
    ).output

)


# ============================================================
# EXTRACT TRAIN EMBEDDINGS
# ============================================================

train_embeddings = embedding_model.predict(

    X_train,

    verbose=0

)


# ============================================================
# EXTRACT TEST EMBEDDINGS
# ============================================================

test_embeddings = embedding_model.predict(

    X_test,

    verbose=0

)


# ============================================================
# PRINT EMBEDDING INFORMATION
# ============================================================

print("\nCNN EMBEDDINGS CREATED SUCCESSFULLY")

print(
    "Train embedding shape:",
    train_embeddings.shape
)

print(
    "Test embedding shape:",
    test_embeddings.shape
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(

    os.path.join(
        OUT_DIR,
        "train_cnn_embeddings.npy"
    ),

    train_embeddings

)


np.save(

    os.path.join(
        OUT_DIR,
        "test_cnn_embeddings.npy"
    ),

    test_embeddings

)


np.save(

    os.path.join(
        OUT_DIR,
        "y_train.npy"
    ),

    y_train

)


np.save(

    os.path.join(
        OUT_DIR,
        "y_test.npy"
    ),

    y_test

)


np.save(

    os.path.join(
        OUT_DIR,
        "train_indices.npy"
    ),

    train_idx

)


np.save(

    os.path.join(
        OUT_DIR,
        "test_indices.npy"
    ),

    test_idx

)


print("\n" + "=" * 70)
print("STAGE 4A COMPLETED")
print("=" * 70)

print("\nSaved files:")

print(
    "train_cnn_embeddings.npy"
)

print(
    "test_cnn_embeddings.npy"
)

print(
    "y_train.npy"
)

print(
    "y_test.npy"
)

# ============================================================
# STAGE 4B — CREATE HANDCRAFTED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING HANDCRAFTED FEATURES")
print("=" * 70)


def extract_handcrafted_features(sequence):
    """
    Extract statistical, activity and entropy features
    from one daily activity sequence.
    """

    x = np.array(sequence, dtype=float)

    # Basic statistics
    mean_activity = np.mean(x)
    std_activity = np.std(x)
    median_activity = np.median(x)
    max_activity = np.max(x)

    q25 = np.percentile(x, 25)
    q75 = np.percentile(x, 75)
    iqr = q75 - q25

    # Activity proportions
    prop_inactive = np.mean(x < 5)
    prop_active = np.mean(x > 100)

    active_minutes = np.sum(x > 100)
    sleep_minutes = np.sum(x < 5)

    # Day / night activity
    day_activity = np.mean(x[360:1320])   # 6 AM - 10 PM

    night_part = np.concatenate([
        x[:360],       # 12 AM - 6 AM
        x[1320:]       # 10 PM - 12 AM
    ])

    night_activity = np.mean(night_part)

    day_night_ratio = (
        (day_activity + 1e-6)
        / (night_activity + 1e-6)
    )

    # Morning / afternoon / evening
    morning_activity = np.mean(x[360:720])       # 6 AM - 12 PM
    afternoon_activity = np.mean(x[720:1020])    # 12 PM - 5 PM
    evening_activity = np.mean(x[1020:1320])     # 5 PM - 10 PM

    # Entropy
    hist, _ = np.histogram(x, bins=20)

    probabilities = hist / (np.sum(hist) + 1e-10)

    entropy_value = -np.sum(
        probabilities * np.log(probabilities + 1e-10)
    )

    # Zero crossings around mean
    zero_crossings = np.sum(
        np.diff(np.sign(x - mean_activity)) != 0
    )

    return [
        mean_activity,
        std_activity,
        median_activity,
        max_activity,
        q25,
        q75,
        iqr,
        prop_inactive,
        prop_active,
        active_minutes,
        sleep_minutes,
        day_activity,
        night_activity,
        day_night_ratio,
        morning_activity,
        afternoon_activity,
        evening_activity,
        entropy_value,
        zero_crossings
    ]


print("Extracting handcrafted features from training data...")

X_train_handcrafted = np.array([
    extract_handcrafted_features(seq)
    for seq in X[train_idx]
])

print("Extracting handcrafted features from test data...")

X_test_handcrafted = np.array([
    extract_handcrafted_features(seq)
    for seq in X[test_idx]
])


print("\nHANDCRAFTED FEATURES CREATED SUCCESSFULLY")

print(
    "Train handcrafted shape:",
    X_train_handcrafted.shape
)

print(
    "Test handcrafted shape:",
    X_test_handcrafted.shape
)

# ============================================================
# FEATURE FUSION
# CNN EMBEDDINGS + HANDCRAFTED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE FUSION")
print("=" * 70)

# Combine CNN embeddings and handcrafted features

# X_train_fusion = np.concatenate(
#     [train_embeddings, train_handcrafted],
#     axis=1
# )

# X_test_fusion = np.concatenate(
#     [test_embeddings, test_handcrafted],
#     axis=1
# )
X_train_fusion = np.concatenate(
    [train_embeddings, X_train_handcrafted],
    axis=1
)

X_test_fusion = np.concatenate(
    [test_embeddings, X_test_handcrafted],
    axis=1
)

print("\nFEATURE FUSION COMPLETED SUCCESSFULLY")

print("CNN embedding features:", train_embeddings.shape[1])
print("Handcrafted features:",  X_train_handcrafted.shape[1])
print("Total fused features:", X_train_fusion.shape[1])

print("\nTrain fused shape:", X_train_fusion.shape)
print("Test fused shape:", X_test_fusion.shape)


# Save fused features

np.save(
    os.path.join(OUT_DIR, "X_train_fusion.npy"),
    X_train_fusion
)

np.save(
    os.path.join(OUT_DIR, "X_test_fusion.npy"),
    X_test_fusion
)

print("\nFused feature files saved successfully:")
print("X_train_fusion.npy")
print("X_test_fusion.npy")


# ============================================================
# TRAIN CATBOOST ON FUSED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CATBOOST ON FUSED FEATURES")
print("=" * 70)

fusion_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="MultiClass",
    random_seed=SEED,
    verbose=50,
    allow_writing_files=False
)

fusion_model.fit(
    X_train_fusion,
    y_train
)

print("\nCATBOOST TRAINING COMPLETED")

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred_fusion = fusion_model.predict(X_test_fusion)

# CatBoost prediction sometimes returns shape (n, 1)
y_pred_fusion = y_pred_fusion.reshape(-1).astype(int)

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE FUSION RESULTS")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred_fusion)

precision = precision_score(
    y_test,
    y_pred_fusion,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred_fusion,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred_fusion,
    average="weighted",
    zero_division=0
)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred_fusion,
        target_names=[
            "Healthy",
            "Depression",
            "Schizophrenia"
        ],
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred_fusion
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision_weighted",
        "Recall_weighted",
        "F1_weighted"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

results.to_csv(
    os.path.join(
        OUT_DIR,
        "feature_fusion_results.csv"
    ),
    index=False
)

pd.DataFrame(
    cm,
    index=[
        "Actual Healthy",
        "Actual Depression",
        "Actual Schizophrenia"
    ],
    columns=[
        "Predicted Healthy",
        "Predicted Depression",
        "Predicted Schizophrenia"
    ]
).to_csv(
    os.path.join(
        OUT_DIR,
        "feature_fusion_confusion_matrix.csv"
    )
)

print("\n" + "=" * 70)
print("STAGE 4 — FEATURE FUSION COMPLETED")
print("=" * 70)

print("\nSaved:")
print("feature_fusion_results.csv")
print("feature_fusion_confusion_matrix.csv")