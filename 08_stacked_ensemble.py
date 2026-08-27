# ============================================================
# 08 — STACKED ENSEMBLE (PART C)
# Actigraphy Dataset
#
# Base Models:
# 1. Random Forest
# 2. CatBoost
# 3. LightGBM
# 4. CNN + Handcrafted Feature Fusion
#
# Meta-Learner:
# Logistic Regression
# ============================================================

import os
import glob
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
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
# SETTINGS
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

DEP_PATH = "data/depresjon"
PSY_PATH = "data/psykose"

OUT_DIR = "outputs/stage4_stacked_ensemble"
os.makedirs(OUT_DIR, exist_ok=True)


print("=" * 70)
print("STAGE 4 — STACKED ENSEMBLE (PART C)")
print("=" * 70)


# ============================================================
# LOAD DAILY ACTIVITY SEQUENCES
# ============================================================

def get_daily_sequences(file_path, label, subject_id):

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = pd.to_datetime(df["date"])

    sequences = []
    labels = []
    subjects = []

    for date, day_df in df.groupby("date"):

        day_df = day_df.copy()

        day_df["minute_of_day"] = (
            day_df["timestamp"].dt.hour * 60
            + day_df["timestamp"].dt.minute
        )

        sequence = np.zeros(1440, dtype=np.float32)

        minutes = day_df["minute_of_day"].to_numpy()
        activities = day_df["activity"].to_numpy()

        valid = (
            (minutes >= 0) &
            (minutes < 1440)
        )

        sequence[minutes[valid]] = activities[valid]

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

print("\nLoading DEPRESJON data...")


# Depression patients
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

    subject_id = (
        "dep_condition_"
        + os.path.splitext(
            os.path.basename(file_path)
        )[0]
    )

    sequences, labels, subjects = get_daily_sequences(
        file_path,
        label=1,
        subject_id=subject_id
    )

    X_list.extend(sequences)
    y_list.extend(labels)
    subject_ids.extend(subjects)


# Depression controls
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

    subject_id = (
        "dep_control_"
        + os.path.splitext(
            os.path.basename(file_path)
        )[0]
    )

    sequences, labels, subjects = get_daily_sequences(
        file_path,
        label=0,
        subject_id=subject_id
    )

    X_list.extend(sequences)
    y_list.extend(labels)
    subject_ids.extend(subjects)


print("Loading PSYKOSE data...")


# Schizophrenia patients
for file_path in sorted(
    glob.glob(
        os.path.join(
            PSY_PATH,
            "patient",
            "*.csv"
        )
    )
):

    subject_id = (
        "psy_patient_"
        + os.path.splitext(
            os.path.basename(file_path)
        )[0]
    )

    sequences, labels, subjects = get_daily_sequences(
        file_path,
        label=2,
        subject_id=subject_id
    )

    X_list.extend(sequences)
    y_list.extend(labels)
    subject_ids.extend(subjects)


# Healthy controls
for file_path in sorted(
    glob.glob(
        os.path.join(
            PSY_PATH,
            "control",
            "*.csv"
        )
    )
):

    subject_id = (
        "psy_control_"
        + os.path.splitext(
            os.path.basename(file_path)
        )[0]
    )

    sequences, labels, subjects = get_daily_sequences(
        file_path,
        label=0,
        subject_id=subject_id
    )

    X_list.extend(sequences)
    y_list.extend(labels)
    subject_ids.extend(subjects)


X = np.array(
    X_list,
    dtype=np.float32
)

y = np.array(y_list)

subject_ids = np.array(subject_ids)


print("\n" + "=" * 70)
print("DATASET LOADED")
print("=" * 70)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Unique subjects:", len(np.unique(subject_ids)))

print("\nClass distribution:")

for class_id, class_name in [
    (0, "Healthy"),
    (1, "Depression"),
    (2, "Schizophrenia")
]:

    print(
        f"{class_name}:",
        np.sum(y == class_id)
    )


# ============================================================
# SAME SUBJECT-WISE TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING SUBJECT-WISE SPLIT")
print("=" * 70)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

train_idx, test_idx = next(
    cv.split(
        X,
        y,
        groups=subject_ids
    )
)


X_train_seq = X[train_idx]
X_test_seq = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

train_subjects = subject_ids[train_idx]
test_subjects = subject_ids[test_idx]


print("Train samples:", len(X_train_seq))
print("Test samples:", len(X_test_seq))

print(
    "Train subjects:",
    len(np.unique(train_subjects))
)

print(
    "Test subjects:",
    len(np.unique(test_subjects))
)

overlap = len(
    set(train_subjects)
    &
    set(test_subjects)
)

print("Subject overlap:", overlap)

if overlap == 0:
    print("\n✓ Subject-wise split is correct.")


# ============================================================
# HANDCRAFTED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING HANDCRAFTED FEATURES")
print("=" * 70)


def extract_features(sequence):

    sequence = np.array(sequence)

    active = sequence[sequence > 0]

    if len(active) == 0:

        active = np.array([0])

    features = [

        # Basic statistics
        np.mean(sequence),
        np.std(sequence),
        np.min(sequence),
        np.max(sequence),
        np.median(sequence),

        # Activity statistics
        np.sum(sequence),
        np.mean(active),
        np.std(active),

        # Percentiles
        np.percentile(sequence, 25),
        np.percentile(sequence, 75),
        np.percentile(sequence, 90),

        # Activity proportion
        np.mean(sequence > 0),

        # Temporal activity
        np.sum(sequence[:360]),
        np.sum(sequence[360:720]),
        np.sum(sequence[720:1080]),
        np.sum(sequence[1080:]),

        # Variability
        np.mean(np.abs(np.diff(sequence))),
        np.std(np.diff(sequence)),

        # Peak count
        np.sum(sequence > np.mean(sequence))

    ]

    return features


X_train_handcrafted = np.array(
    [
        extract_features(seq)
        for seq in X_train_seq
    ]
)

X_test_handcrafted = np.array(
    [
        extract_features(seq)
        for seq in X_test_seq
    ]
)


print(
    "Train handcrafted shape:",
    X_train_handcrafted.shape
)

print(
    "Test handcrafted shape:",
    X_test_handcrafted.shape
)


# ============================================================
# PREPROCESS HANDCRAFTED FEATURES
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

scaler = StandardScaler()


X_train_handcrafted = imputer.fit_transform(
    X_train_handcrafted
)

X_test_handcrafted = imputer.transform(
    X_test_handcrafted
)


X_train_handcrafted = scaler.fit_transform(
    X_train_handcrafted
)

X_test_handcrafted = scaler.transform(
    X_test_handcrafted
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(

    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,

    class_weight="balanced",

    random_state=SEED,
    n_jobs=-1
)

rf_model.fit(
    X_train_handcrafted,
    y_train
)

rf_train_prob = rf_model.predict_proba(
    X_train_handcrafted
)

rf_test_prob = rf_model.predict_proba(
    X_test_handcrafted
)

print("✓ Random Forest completed")


# ============================================================
# CATBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CATBOOST")
print("=" * 70)

cat_model = CatBoostClassifier(

    iterations=300,
    depth=6,
    learning_rate=0.05,

    loss_function="MultiClass",

    random_seed=SEED,

    verbose=100
)

cat_model.fit(
    X_train_handcrafted,
    y_train
)

cat_train_prob = cat_model.predict_proba(
    X_train_handcrafted
)

cat_test_prob = cat_model.predict_proba(
    X_test_handcrafted
)

print("✓ CatBoost completed")


# ============================================================
# LIGHTGBM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

lgb_model = LGBMClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    num_leaves=31,

    subsample=0.8,

    colsample_bytree=0.8,

    class_weight="balanced",

    objective="multiclass",

    random_state=SEED,

    verbose=-1
)

lgb_model.fit(
    X_train_handcrafted,
    y_train
)

lgb_train_prob = lgb_model.predict_proba(
    X_train_handcrafted
)

lgb_test_prob = lgb_model.predict_proba(
    X_test_handcrafted
)

print("✓ LightGBM completed")


# ============================================================
# CNN FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CNN FOR FUSION MODEL")
print("=" * 70)


X_train_cnn = (
    X_train_seq[..., np.newaxis]
)

X_test_cnn = (
    X_test_seq[..., np.newaxis]
)


def build_cnn():

    model = Sequential([

        Input(shape=(1440, 1)),

        Conv1D(
            32,
            kernel_size=7,
            activation="relu"
        ),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            64,
            kernel_size=5,
            activation="relu"
        ),

        MaxPooling1D(
            pool_size=2
        ),

        Conv1D(
            128,
            kernel_size=3,
            activation="relu"
        ),

        GlobalAveragePooling1D(),

        Dense(
            64,
            activation="relu",
            name="embedding"
        ),

        Dropout(0.3),

        Dense(
            3,
            activation="softmax"
        )

    ])


    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


cnn_model = build_cnn()


cnn_model.fit(

    X_train_cnn,
    y_train,

    epochs=10,

    batch_size=32,

    verbose=1
)


print("✓ CNN training completed")


# ============================================================
# EXTRACT CNN EMBEDDINGS
# ============================================================

embedding_model = Model(

    inputs=cnn_model.inputs,

    outputs=cnn_model.get_layer(
        "embedding"
    ).output
)


cnn_train_embedding = embedding_model.predict(
    X_train_cnn,
    verbose=0
)

cnn_test_embedding = embedding_model.predict(
    X_test_cnn,
    verbose=0
)


print(
    "CNN train embedding:",
    cnn_train_embedding.shape
)

print(
    "CNN test embedding:",
    cnn_test_embedding.shape
)


# ============================================================
# FEATURE FUSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FEATURE FUSION MODEL")
print("=" * 70)


X_train_fusion = np.hstack([

    cnn_train_embedding,

    X_train_handcrafted

])

X_test_fusion = np.hstack([

    cnn_test_embedding,

    X_test_handcrafted

])


fusion_model = CatBoostClassifier(

    iterations=500,

    depth=6,

    learning_rate=0.05,

    loss_function="MultiClass",

    random_seed=SEED,

    verbose=100
)


fusion_model.fit(

    X_train_fusion,

    y_train
)


fusion_train_prob = fusion_model.predict_proba(
    X_train_fusion
)

fusion_test_prob = fusion_model.predict_proba(
    X_test_fusion
)


print("✓ Feature Fusion completed")


# ============================================================
# CREATE STACKING DATA
# ============================================================

print("\n" + "=" * 70)
print("CREATING STACKING FEATURES")
print("=" * 70)


X_meta_train = np.hstack([

    rf_train_prob,

    cat_train_prob,

    lgb_train_prob,

    fusion_train_prob

])


X_meta_test = np.hstack([

    rf_test_prob,

    cat_test_prob,

    lgb_test_prob,

    fusion_test_prob

])


print(
    "Meta train shape:",
    X_meta_train.shape
)

print(
    "Meta test shape:",
    X_meta_test.shape
)


# ============================================================
# META LEARNER
# ============================================================

print("\n" + "=" * 70)
print("TRAINING META-LEARNER")
print("=" * 70)


meta_model = LogisticRegression(

    max_iter=2000,

    class_weight="balanced",

    random_state=SEED
)


meta_model.fit(

    X_meta_train,

    y_train
)


y_pred = meta_model.predict(
    X_meta_test
)


y_prob = meta_model.predict_proba(
    X_meta_test
)


print("✓ Meta-learner training completed")


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL STACKED ENSEMBLE RESULTS")
print("=" * 70)


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print(f"\nAccuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1:.4f}")


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


print(

    classification_report(

        y_test,

        y_pred,

        target_names=[

            "Healthy",

            "Depression",

            "Schizophrenia"

        ],

        zero_division=0

    )

)


cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([{

    "Accuracy": accuracy,

    "Precision_weighted": precision,

    "Recall_weighted": recall,

    "F1_weighted": f1

}])


results.to_csv(

    os.path.join(
        OUT_DIR,
        "stacked_ensemble_results.csv"
    ),

    index=False

)


pd.DataFrame(

    cm,

    index=[
        "Healthy",
        "Depression",
        "Schizophrenia"
    ],

    columns=[
        "Healthy",
        "Depression",
        "Schizophrenia"
    ]

).to_csv(

    os.path.join(
        OUT_DIR,
        "stacked_ensemble_confusion_matrix.csv"
    )

)


np.save(

    os.path.join(
        OUT_DIR,
        "meta_test_features.npy"
    ),

    X_meta_test
)


np.save(

    os.path.join(
        OUT_DIR,
        "final_predictions.npy"
    ),

    y_pred
)


print("\n" + "=" * 70)
print("STAGE 4 — STACKED ENSEMBLE COMPLETED")
print("=" * 70)

print("\nSaved files:")

print(
    "stacked_ensemble_results.csv"
)

print(
    "stacked_ensemble_confusion_matrix.csv"
)

print(
    "meta_test_features.npy"
)

print(
    "final_predictions.npy"
)