Mental Health Monitoring — FYP
Project overview
This project investigates mental-health classification from actigraphy/behavioral data using the DEPRESJON and PSYKOSE datasets, with a separate component for the team's own smartphone survey dataset.
The current public-dataset experiments use 109 subjects:
DEPRESJON: 55 subjects — 23 depression, 32 healthy
PSYKOSE: 54 subjects — 22 schizophrenia, 32 healthy
The team's own 1,823-row smartphone survey dataset is kept as a separate component and has not been used in the Stage 1–3 public-dataset validation experiments.
---
Current status
Stage 1 — Rigorous validation ✅
Completed:
5-fold stratified cross-validation
RF, CatBoost and LightGBM baseline evaluation
Accuracy, precision, recall, F1 and AUC-ROC
Nested CV for hyperparameter tuning
Nested-CV LightGBM result:
Accuracy: 76.2% ± 8.0%
F1: 0.726 ± 0.114
AUC-ROC: 0.876 ± 0.072
Stage 2 — Class imbalance ✅
Compared:
Class weighting
Borderline-SMOTE
ADASYN
Imbalance handling was performed inside training folds only; outer/test folds were not resampled.
Best screening experiment:
Borderline-SMOTE + LightGBM
Accuracy: 79.8% ± 8.2%
Precision: 0.796 ± 0.105
Recall: 0.798 ± 0.082
F1: 0.779 ± 0.102
AUC-ROC: 0.911 ± 0.094
For the clinical-screening framing, recall/sensitivity and F1 should be prioritized over raw accuracy, because false negatives represent potentially missed cases.
Stage 3 — Feature enhancement ✅
Added:
Interdaily Stability (IS)
Intradaily Variability (IV)
Relative Amplitude (RA)
Morning activity
Afternoon activity
Evening activity
Night activity
Best ordinary 5-fold CV experiment:
Enhanced features + LightGBM
Accuracy: 80.7% ± 4.1%
Precision: 0.822 ± 0.070
Recall: 0.807 ± 0.041
F1: 0.778 ± 0.053
AUC-ROC: 0.927 ± 0.030
Important: 80.7% is an experimental 5-fold CV result, not the final nested-CV publication estimate. Do not present it as the final unbiased performance without further validation.
---
File guide
Current experiment scripts
File	Purpose
`baseline_cv.py`	Original 5-fold baseline experiment / historical 78% result
`stage1_validation.py`	Rigorous baseline metrics + nested CV
`stage2_imbalance.py`	Class-weighting vs Borderline-SMOTE vs ADASYN
`stage2_imbalance_verification.py`	Per-fold class-distribution verification
`stage3_feature_enhancement.py`	IS/IV/RA + time-segment feature experiment
Earlier project pipelines
File	Purpose
`depresjon_psykose_pipeline.py`	Earlier DEPRESJON + PSYKOSE pipeline, including survey integration
`depresjon_psykose_pipeline_v2.py`	Earlier richer per-day/per-hour and demographic feature pipeline
`final_hybrid_pipeline.py`	Hybrid architecture containing the team's own 1,823-row survey component
Earlier project scripts
`01_eda.py`, `02_preprocessing.py`, `03_models.py`, `04_shap_explainability.py`, and `05_results_summary.py` are earlier project-stage scripts and historical analysis utilities.
---
Results
Current structured experiment outputs are stored under:
```text
outputs/
├── stage1_validation/
├── stage2_imbalance/
├── stage2_imbalance_verification/
└── stage3_feature_enhancement/
```
These should be treated as the main results folders for the current work.
---
Dataset setup
Raw datasets are not included in this repository.
Expected structure:
```text
data/
├── depresjon/
│   ├── condition/
│   ├── control/
│   └── scores.csv
├── psykose/
│   ├── patient/
│   ├── control/
│   └── patients_info.csv
└── data_collection.csv
```
DEPRESJON:
https://datasets.simula.no/depresjon/
PSYKOSE:
https://datasets.simula.no/psykose/
The team's own `data_collection.csv` should be obtained from the team rather than committed publicly.
---
Next work for the team
Deep temporal models: CNN / GRU / LSTM
Learn representations directly from actigraphy sequences
Fuse learned temporal embeddings with handcrafted features
Evaluate stacking/ensemble methods without leakage
Explainability: SHAP / Integrated Gradients as appropriate
Evaluate the own 1,823-row smartphone survey dataset
Perform ablation studies and statistical significance testing
Run final leakage-safe validation and prepare the paper/report
---
Important reproducibility notes
Keep subjects separated between train and test folds.
Fit preprocessing only on training data.
Perform resampling only inside training folds.
Do not tune hyperparameters using outer test folds.
Do not treat the 80.7% ordinary-CV result as the final unbiased test performance.
Record mean ± standard deviation across folds.