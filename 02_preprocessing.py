import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib, os

os.makedirs("outputs/models", exist_ok=True)

df = pd.read_csv("data/data_collection.csv")
df.columns = [
    "timestamp","age","smartphone_hours","social_media_hours",
    "study_proportion","sleep_hours","living_arrangement",
    "meditation_freq","academic_pressure",
    "phq4_q1","phq4_q2","phq4_q3","phq4_q4"
]

# --- Encode ordinals ---
phone_map  = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10+":10}
sleep_map  = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"10+":10}
social_map = {str(i):i for i in range(1,10)}; social_map["10+"]=10
study_map  = {"Less than 25%":1,"25% - 50%":2,"50% - 75%":3,"More than 75%":4}
med_map    = {"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Daily":4}
living_map = {"With family":0,"In hostel":1,"Alone":2,"With friends":3}

df["phone"]   = df["smartphone_hours"].astype(str).str.strip().map(phone_map).fillna(6)
df["sleep"]   = df["sleep_hours"].astype(str).str.strip().map(sleep_map).fillna(6)
df["social"]  = df["social_media_hours"].astype(str).str.strip().map(social_map).fillna(3)
df["study"]   = df["study_proportion"].str.strip().map(study_map).fillna(2)
df["meditate"]= df["meditation_freq"].str.strip().map(med_map).fillna(1)
df["living"]  = df["living_arrangement"].str.strip().map(living_map).fillna(0)

# --- PHQ-4 scores ---
df["GAD2"]  = df["phq4_q1"] + df["phq4_q2"]
df["PHQ2"]  = df["phq4_q3"] + df["phq4_q4"]
df["PHQ4"]  = df["GAD2"] + df["PHQ2"]

# --- Engineered features (KEY for accuracy boost) ---
df["phone_x_social"]    = df["phone"] * df["social"]          # digital overload
df["sleep_deficit"]     = np.maximum(0, 7 - df["sleep"])      # hours below healthy
df["pressure_x_sleep"]  = df["academic_pressure"] * df["sleep_deficit"]
df["screen_study_ratio"]= df["phone"] / (df["study"] + 1)
df["social_isolation"]  = (df["living"] == 2).astype(int)
df["low_meditation"]    = (df["meditate"] <= 1).astype(int)
df["high_phone"]        = (df["phone"] >= 8).astype(int)
df["poor_sleep"]        = (df["sleep"] <= 5).astype(int)
df["high_pressure"]     = (df["academic_pressure"] >= 4).astype(int)
df["risk_score"]        = (df["poor_sleep"] + df["high_phone"] +
                           df["high_pressure"] + df["low_meditation"])

# --- Labels ---
df["severity"]  = pd.cut(df["PHQ4"],bins=[-1,3,5,8,12],
                          labels=[0,1,2,3]).astype(int)
df["binary"]    = (df["PHQ4"] >= 6).astype(int)   # moderate+severe
df["anxiety"]   = (df["GAD2"] >= 3).astype(int)
df["depression"]= (df["PHQ2"] >= 3).astype(int)

FEATURES = [
    "age","phone","sleep","social","study","living","meditate",
    "academic_pressure",
    # engineered
    "phone_x_social","sleep_deficit","pressure_x_sleep",
    "screen_study_ratio","social_isolation","low_meditation",
    "high_phone","poor_sleep","high_pressure","risk_score"
]

X = df[FEATURES]
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

splits = {}
for task, col in [("severity","severity"),("binary","binary"),
                   ("anxiety","anxiety"),("depression","depression")]:
    y = df[col]
    Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,
                                        random_state=42,stratify=y)
    splits[task] = (Xtr,Xte,ytr,yte)
    print(f"{task}: train={len(ytr)}, test={len(yte)}, classes={y.value_counts().to_dict()}")

joblib.dump(scaler,"outputs/models/scaler.pkl")
joblib.dump(splits,"outputs/models/splits.pkl")
joblib.dump(FEATURES,"outputs/models/features.pkl")
df.to_csv("data/df_processed.csv",index=False)
print("\nDone. Features:", len(FEATURES))