"""
STAGE 3 — FEATURE ENHANCEMENT

Dataset:
    109 subjects
    DEPRESJON + PSYKOSE

Adds:
    - Interdaily Stability (IS)
    - Intradaily Variability (IV)
    - Relative Amplitude (RA)
    - Sample Entropy
    - Lempel-Ziv Complexity
    - Morning activity
    - Afternoon activity
    - Evening activity
    - Night activity

Experiments:
    A) Existing 18 baseline features
    B) Enhanced 27 features
    C) Enhanced 27 features + Borderline-SMOTE

All preprocessing and resampling are performed
inside each training fold.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import entropy as scipy_entropy

import antropy as ant

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEP = "data/depresjon"
PSY = "data/psykose"

OUT = "outputs/stage3_feature_enhancement"

os.makedirs(OUT, exist_ok=True)


# ============================================================
# BASELINE FEATURES — EXISTING 18
# ============================================================

BASE_FEATURES = [
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
    "n_days"
]


# ============================================================
# SAMPLE ENTROPY
# ============================================================

def sample_entropy(x):
    """
    Sample Entropy using AntroPy.

    The signal is limited to 2000 samples to keep the
    computation practical for long actigraphy recordings.
    """

    x = np.asarray(x, dtype=float)

    # Remove invalid values
    x = x[np.isfinite(x)]

    if len(x) < 100:
        return np.nan

    # Downsample long recordings
    if len(x) > 2000:
        idx = np.linspace(
            0,
            len(x) - 1,
            2000
        ).astype(int)

        x = x[idx]

    # Constant signal
    sd = np.std(x)

    if sd < 1e-12:
        return 0.0

    try:

        return float(
            ant.sample_entropy(
                x,
                order=2,
                tolerance=0.2 * sd
            )
        )

    except Exception:

        return np.nan


# ============================================================
# LEMPEL-ZIV COMPLEXITY
# ============================================================

def lz_complexity(x):
    """
    Normalized Lempel-Ziv complexity using AntroPy.

    Continuous activity is converted into a binary sequence
    using the median activity as the threshold.
    """

    x = np.asarray(x, dtype=float)

    # Remove invalid values
    x = x[np.isfinite(x)]

    if len(x) < 50:
        return np.nan

    # Limit sequence length
    if len(x) > 5000:

        idx = np.linspace(
            0,
            len(x) - 1,
            5000
        ).astype(int)

        x = x[idx]

    # Convert continuous activity to binary sequence
    binary = (
        x > np.median(x)
    ).astype(int)

    try:

        return float(
            ant.lziv_complexity(
                binary,
                normalize=True
            )
        )

    except Exception:

        return np.nan


# ============================================================
# CIRCADIAN FEATURES
# ============================================================

def circadian_features(d):

    x = d["activity"].to_numpy(
        dtype=float
    )

    # If timestamp unavailable
    if "timestamp" not in d.columns:

        return {
            "IS": np.nan,
            "IV": np.nan,
            "RA": np.nan,
            "morning_activity": np.nan,
            "afternoon_activity": np.nan,
            "evening_activity": np.nan,
            "night_activity": np.nan
        }

    t = pd.to_datetime(
        d["timestamp"],
        errors="coerce"
    )

    tmp = pd.DataFrame({
        "activity": x,
        "timestamp": t
    }).dropna()

    if tmp.empty:

        return {
            "IS": np.nan,
            "IV": np.nan,
            "RA": np.nan,
            "morning_activity": np.nan,
            "afternoon_activity": np.nan,
            "evening_activity": np.nan,
            "night_activity": np.nan
        }

    # Hour
    tmp["hour"] = (
        tmp["timestamp"].dt.hour
    )

    # Hourly activity profile
    hourly = (
        tmp.groupby("hour")["activity"]
        .mean()
    )

    hourly_complete = (
        hourly.reindex(range(24))
    )

    # Overall variance
    overall_variance = (
        tmp["activity"].var()
    )

    # ========================================================
    # IS — Interdaily Stability
    # ========================================================

    IS = (
        hourly_complete.var(
            skipna=True
        )
        /
        (overall_variance + 1e-12)
    )

    # ========================================================
    # IV — Intradaily Variability
    # ========================================================

    IV = (
        np.mean(
            np.diff(
                tmp["activity"].to_numpy()
            ) ** 2
        )
        /
        (overall_variance + 1e-12)
    )

    # ========================================================
    # RA — Relative Amplitude
    # ========================================================

    prof = (
        hourly_complete
        .interpolate(
            limit_direction="both"
        )
        .to_numpy()
    )

    if np.isfinite(prof).all():

        rolling10 = np.array([
            prof[
                np.arange(
                    i,
                    i + 10
                ) % 24
            ].mean()
            for i in range(24)
        ])

        rolling5 = np.array([
            prof[
                np.arange(
                    i,
                    i + 5
                ) % 24
            ].mean()
            for i in range(24)
        ])

        RA = (
            rolling10.max()
            -
            rolling5.min()
        ) / (
            rolling10.max()
            +
            rolling5.min()
            +
            1e-12
        )

    else:

        RA = np.nan

    # ========================================================
    # Time segments
    # ========================================================

    def mean_hours(
        start_hour,
        end_hour
    ):

        z = tmp[
            tmp["hour"].between(
                start_hour,
                end_hour - 1
            )
        ]["activity"]

        if len(z):

            return z.mean()

        return np.nan

    # Morning: 06–12
    morning = mean_hours(
        6,
        12
    )

    # Afternoon: 12–17
    afternoon = mean_hours(
        12,
        17
    )

    # Evening: 17–22
    evening = mean_hours(
        17,
        22
    )

    # Night: 22–06
    night_data = tmp[
        (tmp["hour"] >= 22)
        |
        (tmp["hour"] < 6)
    ]["activity"]

    night = (
        night_data.mean()
        if len(night_data)
        else np.nan
    )

    return {

        "IS": IS,

        "IV": IV,

        "RA": RA,

        "morning_activity":
            morning,

        "afternoon_activity":
            afternoon,

        "evening_activity":
            evening,

        "night_activity":
            night
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract(
    fp,
    label,
    label_str
):

    d = pd.read_csv(fp)

    # Raw activity
    a = d[
        "activity"
    ].to_numpy(
        dtype=float
    )

    # Valid activity values
    a_clean = a[
        np.isfinite(a)
    ]

    if len(a_clean) == 0:

        a_clean = np.array(
            [0.0]
        )

    # ========================================================
    # Percentiles
    # ========================================================

    q25, q75 = np.percentile(
        a_clean,
        [25, 75]
    )

    # ========================================================
    # Existing histogram entropy
    # ========================================================

    hist, _ = np.histogram(
        a_clean,
        bins=20,
        density=True
    )

    histogram_entropy = scipy_entropy(
        hist + 1e-10
    )

    # ========================================================
    # Day / Night features
    # ========================================================

    if "timestamp" in d.columns:

        t = pd.to_datetime(
            d["timestamp"],
            errors="coerce"
        )

        hour = (
            t.dt.hour
            .fillna(0)
            .astype(int)
        )

        day_mask = (
            hour.between(
                6,
                21
            )
            .to_numpy()
        )

        day = a[
            day_mask
        ]

        night = a[
            ~day_mask
        ]

        day = day[
            np.isfinite(day)
        ]

        night = night[
            np.isfinite(night)
        ]

        dayv = (
            np.mean(day)
            if len(day)
            else np.nan
        )

        nightv = (
            np.mean(night)
            if len(night)
            else np.nan
        )

        ratio = (
            (dayv + 1e-5)
            /
            (nightv + 1e-5)
        )

        circ = (
            d.assign(
                hour=hour
            )
            .groupby("hour")[
                "activity"
            ]
            .mean()
            .std()
        )

    else:

        dayv = np.nan
        nightv = np.nan
        ratio = np.nan
        circ = np.nan

    # ========================================================
    # NEW FEATURES
    # ========================================================

    samp_entropy = sample_entropy(
        a_clean
    )

    lz = lz_complexity(
        a_clean
    )

    # ========================================================
    # Subject ID
    # ========================================================

    sid = os.path.splitext(
        os.path.basename(fp)
    )[0]

    # ========================================================
    # Feature dictionary
    # ========================================================

    r = {

        # ----------------------------------------------------
        # Identification
        # ----------------------------------------------------

        "subject_id":
            sid,

        "label":
            label,

        "label_str":
            label_str,

        # ----------------------------------------------------
        # Existing statistical features
        # ----------------------------------------------------

        "mean_activity":
            np.mean(a_clean),

        "std_activity":
            np.std(a_clean),

        "median_activity":
            np.median(a_clean),

        "max_activity":
            np.max(a_clean),

        "q25_activity":
            q25,

        "q75_activity":
            q75,

        "iqr_activity":
            q75 - q25,

        "entropy":
            histogram_entropy,

        # ----------------------------------------------------
        # NEW: complexity features
        # ----------------------------------------------------

        "sample_entropy":
            samp_entropy,

        "lz_complexity":
            lz,

        # ----------------------------------------------------
        # Remaining baseline features
        # ----------------------------------------------------

        "zero_crossings":
            np.sum(
                np.diff(
                    np.sign(
                        a_clean
                        -
                        np.mean(a_clean)
                    )
                ) != 0
            ),

        "sleep_hours":
            np.sum(
                a_clean < 5
            ) / 60,

        "active_minutes":
            np.sum(
                a_clean > 100
            ),

        "day_activity":
            dayv,

        "night_activity":
            nightv,

        "day_night_ratio":
            ratio,

        "circ_regularity":
            circ,

        "prop_inactive":
            np.mean(
                a_clean < 5
            ),

        "prop_active":
            np.mean(
                a_clean > 100
            ),

        "n_days":
            (
                len(
                    d["date"].unique()
                )
                if "date" in d.columns
                else 1
            )
    }

    # Add IS / IV / RA / time segments
    r.update(
        circadian_features(d)
    )

    return r


# ============================================================
# LOAD DATA
# ============================================================

def load():

    dep = []

    # --------------------------------------------------------
    # DEPRESJON — Depression
    # --------------------------------------------------------

    for f in sorted(
        glob.glob(
            os.path.join(
                DEP,
                "condition",
                "*.csv"
            )
        )
    ):

        dep.append(
            extract(
                f,
                1,
                "depression"
            )
        )

    # --------------------------------------------------------
    # DEPRESJON — Healthy
    # --------------------------------------------------------

    for f in sorted(
        glob.glob(
            os.path.join(
                DEP,
                "control",
                "*.csv"
            )
        )
    ):

        dep.append(
            extract(
                f,
                0,
                "healthy"
            )
        )

    psy = []

    # --------------------------------------------------------
    # PSYKOSE — Schizophrenia
    # --------------------------------------------------------

    for f in sorted(
        glob.glob(
            os.path.join(
                PSY,
                "patient",
                "*.csv"
            )
        )
    ):

        psy.append(
            extract(
                f,
                2,
                "schizophrenia"
            )
        )

    # --------------------------------------------------------
    # PSYKOSE — Healthy
    # --------------------------------------------------------

    for f in sorted(
        glob.glob(
            os.path.join(
                PSY,
                "control",
                "*.csv"
            )
        )
    ):

        psy.append(
            extract(
                f,
                0,
                "healthy"
            )
        )

    return (
        pd.DataFrame(dep),
        pd.DataFrame(psy)
    )


# ============================================================
# MODELS
# ============================================================

def model(name):

    if name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=SEED,
            n_jobs=-1
        )

    if name == "CatBoost":

        return CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False
        )

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    X,
    y,
    feature_set,
    imbalance
):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    rows = []

    for name in [
        "Random Forest",
        "CatBoost",
        "LightGBM"
    ]:

        for fold, (
            tr,
            te
        ) in enumerate(
            cv.split(X, y),
            1
        ):

            # =================================================
            # BORDERLINE-SMOTE
            # =================================================

            if imbalance == "Borderline-SMOTE":

                pipe = ImbPipeline([

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    ),

                    (
                        "sampler",
                        BorderlineSMOTE(
                            random_state=SEED,
                            k_neighbors=3
                        )
                    ),

                    (
                        "model",
                        model(name)
                    )
                ])

            # =================================================
            # NO RESAMPLING
            # =================================================

            else:

                pipe = Pipeline([

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    ),

                    (
                        "model",
                        model(name)
                    )
                ])

            # =================================================
            # TRAIN ONLY ON TRAINING FOLD
            # =================================================

            pipe.fit(
                X[tr],
                y[tr]
            )

            # =================================================
            # TEST
            # =================================================

            yp = pipe.predict(
                X[te]
            )

            prob = pipe.predict_proba(
                X[te]
            )

            # =================================================
            # MULTICLASS AUC
            # =================================================

            y_test_bin = label_binarize(
                y[te],
                classes=[0, 1, 2]
            )

            auc = roc_auc_score(
                y_test_bin,
                prob,
                average="weighted",
                multi_class="ovr"
            )

            # =================================================
            # STORE RESULT
            # =================================================

            rows.append({

                "feature_set":
                    feature_set,

                "imbalance":
                    imbalance,

                "model":
                    name,

                "fold":
                    fold,

                "accuracy":
                    accuracy_score(
                        y[te],
                        yp
                    ),

                "precision":
                    precision_score(
                        y[te],
                        yp,
                        average="weighted",
                        zero_division=0
                    ),

                "recall":
                    recall_score(
                        y[te],
                        yp,
                        average="weighted",
                        zero_division=0
                    ),

                "f1":
                    f1_score(
                        y[te],
                        yp,
                        average="weighted",
                        zero_division=0
                    ),

                "auc":
                    auc
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

print(
    "\nStarting Stage 3..."
)

print(
    "Extracting subject-level features..."
)

dep, psy = load()

data = pd.concat(
    [
        dep,
        psy
    ],
    ignore_index=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print(
    f"\nDEPRESJON: {len(dep)}"
)

print(
    f"PSYKOSE: {len(psy)}"
)

print(
    f"TOTAL: {len(data)}"
)

print(
    "\nCLASS DISTRIBUTION:"
)

print(
    data["label_str"].value_counts()
)


# ============================================================
# ENHANCED FEATURE SET — 27 FEATURES
# ============================================================

ENHANCED = BASE_FEATURES + [

    # Circadian
    "IS",
    "IV",
    "RA",

    # Time segments
    "morning_activity",
    "afternoon_activity",
    "evening_activity",
    "night_activity",

    # Complexity
    "sample_entropy",
    "lz_complexity"
]


# ============================================================
# DATA MATRICES
# ============================================================

Xbase = data[
    BASE_FEATURES
].to_numpy()

Xenh = data[
    ENHANCED
].to_numpy()

y = data[
    "label"
].to_numpy()


# ============================================================
# FEATURE INFORMATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FEATURE INFORMATION"
)

print(
    "=" * 70
)

print(
    f"Baseline feature count: "
    f"{len(BASE_FEATURES)}"
)

print(
    f"Enhanced feature count: "
    f"{len(ENHANCED)}"
)

print(
    "\nNew features:"
)

for feature in ENHANCED[
    len(BASE_FEATURES):
]:

    print(
        f"  + {feature}"
    )


# ============================================================
# CHECK NEW FEATURE VALUES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "NEW FEATURE CHECK"
)

print(
    "=" * 70
)

print(
    data[
        [
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
    ].describe()
)


# ============================================================
# EXPERIMENT A — BASELINE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "A) BASELINE — 18 FEATURES"
)

print(
    "=" * 70
)

results = []

results.append(
    evaluate(
        Xbase,
        y,
        "18 baseline features",
        "None"
    )
)


# ============================================================
# EXPERIMENT B — ENHANCED
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "B) ENHANCED — 27 FEATURES"
)

print(
    "=" * 70
)

results.append(
    evaluate(
        Xenh,
        y,
        "27 enhanced features",
        "None"
    )
)


# ============================================================
# EXPERIMENT C — ENHANCED + BORDERLINE-SMOTE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "C) ENHANCED + BORDERLINE-SMOTE"
)

print(
    "=" * 70
)

results.append(
    evaluate(
        Xenh,
        y,
        "27 enhanced features",
        "Borderline-SMOTE"
    )
)


# ============================================================
# COMBINE RESULTS
# ============================================================

df = pd.concat(
    results,
    ignore_index=True
)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_path = os.path.join(
    OUT,
    "stage3_by_fold.csv"
)

df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    df
    .groupby(
        [
            "feature_set",
            "imbalance",
            "model"
        ]
    )
    .agg(

        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_std=(
            "accuracy",
            "std"
        ),

        precision_mean=(
            "precision",
            "mean"
        ),

        precision_std=(
            "precision",
            "std"
        ),

        recall_mean=(
            "recall",
            "mean"
        ),

        recall_std=(
            "recall",
            "std"
        ),

        f1_mean=(
            "f1",
            "mean"
        ),

        f1_std=(
            "f1",
            "std"
        ),

        auc_mean=(
            "auc",
            "mean"
        ),

        auc_std=(
            "auc",
            "std"
        )
    )
    .reset_index()
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUT,
    "stage3_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STAGE 3 SUMMARY — MEAN ± STD"
)

print(
    "=" * 100
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# BEST MODELS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "BEST RESULTS"
)

print(
    "=" * 100
)

best_accuracy = summary.loc[
    summary["accuracy_mean"].idxmax()
]

best_f1 = summary.loc[
    summary["f1_mean"].idxmax()
]

best_auc = summary.loc[
    summary["auc_mean"].idxmax()
]

print(
    "\nBest Accuracy:"
)

print(
    best_accuracy.to_string()
)

print(
    "\nBest F1:"
)

print(
    best_f1.to_string()
)

print(
    "\nBest AUC:"
)

print(
    best_auc.to_string()
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STAGE 3 COMPLETED"
)

print(
    "=" * 100
)

print(
    "\nResults saved:"
)

print(
    fold_path
)

print(
    summary_path
)

print(
    "\nPrimary metrics: Recall and F1."
)

print(
    "Accuracy and AUC are secondary."
)