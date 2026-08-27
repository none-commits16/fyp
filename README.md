# Mental Health Monitoring — FYP

Mental-health classification using DEPRESJON and PSYKOSE actigraphy datasets, with a separate smartphone survey component.

## Completed

- 5-fold cross-validation
- Nested CV
- RF, CatBoost and LightGBM
- Class weighting, Borderline-SMOTE and ADASYN
- Circadian features: IS, IV and RA
- Time-based activity features

## Current Best Result

Enhanced LightGBM:

- Accuracy: 80.7%
- F1: 0.778
- AUC: 0.927

Note: 80.7% is an ordinary 5-fold CV result, not the final unbiased performance.

## Main Files

- `baseline_cv.py` — baseline
- `stage1_validation.py` — validation + nested CV
- `stage2_imbalance.py` — imbalance experiments
- `stage2_imbalance_verification.py` — imbalance verification
- `stage3_feature_enhancement.py` — feature enhancement
- `final_hybrid_pipeline.py` — own survey dataset / hybrid pipeline

## Next

CNN / GRU / LSTM → temporal embeddings → feature fusion → stacking → explainability → own dataset validation → final evaluation.

## Setup

```bash
git clone https://github.com/none-commits16/fyp.git
cd fyp
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
