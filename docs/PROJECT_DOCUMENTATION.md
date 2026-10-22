# Job Acceptance Prediction System - Project Documentation

*Domain: HR Analytics / Recruitment & Placement Analytics (Machine Learning)*

> This document explains the **method**, the **data dictionary**, the **design decisions** and the **assumptions**.
> The numbers (model metrics, findings, charts) are produced by your run and written to `docs/RESULTS_REPORT.md`
> (plus `reports/*.json|csv|png`), so they always match the dataset you ran.

## 1. Problem statement

Recruitment teams handle large volumes of candidate data - academics, skills, experience, interview outcomes and job-market
conditions - yet not every eligible candidate or offer ends in an acceptance. The project builds a **Job Acceptance
Prediction System** that

1. predicts whether a candidate will accept (be placed, `status = Placed`) or reject (`Not Placed`);
2. identifies the factors that influence the outcome;
3. handles real-world data problems (missing values, duplicates, inconsistent labels);
4. turns the findings into recommendations for recruitment and placement strategy.

**Business use cases:** estimate offer-acceptance probability, reduce offer drop-outs and hiring cycle time, find the drivers of
placement success, tune interview/evaluation strategy, support data-driven decisions for HR managers.

## 2. Data dictionary (`HR_Job_Placement_Dataset.csv`, 50,000 x 26)

| # | Column | Type | Meaning / role |
|---|---|---|---|
| 1 | `age_years` | int | Demographic analysis only; **not a model input** (bias avoidance) |
| 2 | `gender` | Male/Female | Bias-aware analysis only; **not a model input** |
| 3-5 | `ssc_percentage`, `hsc_percentage`, `degree_percentage` | float | Academic strength / consistency / core indicator |
| 6 | `degree_specialization` | cat | CS, IT, Electronics, Mechanical, Others |
| 7 | `technical_score` | float | Primary screening criterion; below a threshold = not eligible |
| 8 | `aptitude_score` | float | Analytical thinking / job readiness |
| 9 | `communication_score` | float | Verbal & written communication |
| 10 | `skills_match_percentage` | float | Skills vs. job requirements (strong predictor) |
| 11 | `certifications_count` | int | Continuous learning |
| 12 | `internship_experience` | Yes/No | Practical exposure |
| 13 | `years_of_experience` | int | Experience depth |
| 14 | `career_switch_willingness` | Willing/Not Willing | Flexibility |
| 15 | `relevant_experience` | Relevant/Not Relevant | Domain alignment |
| 16-17 | `previous_ctc_lpa`, `expected_ctc_lpa` | float | Compensation history / expectation (large gap hurts) |
| 18 | `company_tier` | Tier 1/2/3 | Employer brand / offer attractiveness |
| 19 | `job_role_match` | Matched/Not Matched | Offered role vs. profile |
| 20 | `competition_level` | Low/Medium/High | Competition for the role |
| 21 | `bond_requirement` | Required/Not Required | Service bond (often lowers acceptance) |
| 22 | `notice_period_days` | int | Longer notice hurts hiring urgency |
| 23 | `layoff_history` | Yes/No | Risk signal |
| 24 | `employment_gap_months` | int | Long gaps hurt |
| 25 | `relocation_willingness` | Willing/Not Willing | Location flexibility |
| 26 | `status` | **target** | `Placed` -> job accepted (1); `Not Placed` -> not accepted (0) |

The brief says the data intentionally contains **missing values, duplicate-like records and inconsistent categorical values**.

## 3. Pipeline

```
raw CSV
  │  data_understanding.py  → reports/data_understanding.md
  ▼
preprocessing.py ── standardised (NaNs kept) ──────────────┐
  │                                                         │ (model imputes inside CV folds)
  ▼ imputed                                                 ▼
features.py → analytical_dataset.csv                   modeling.py
  │                    ▲  placement_probability_score (out-of-fold) + risk_band
  ├──► eda.py (figures, EDA-guide answers)                  │
  ├──► sql_loader.py (MySQL)                                ▼
  └──► Streamlit dashboard  ◄──────────── models/best_model.joblib, reports/*
                                  generate_report.py → docs/RESULTS_REPORT.md
```

### 3.1 Data cleaning & preprocessing (`src/preprocessing.py`)

| Problem | Treatment |
|---|---|
| Case / whitespace / synonym variants (`yes `, `male`, `Tier 2 `) | Mapped to canonical labels via the alias tables in `config.py`; unrecognised labels -> missing (and counted in the quality report) |
| Non-numeric text in numeric columns | Coerced to numbers, failures -> missing |
| Impossible values (e.g. >100 %, negative CTC) | Treated as noise -> missing (ranges in `config.py`) |
| Over-long decimals | Rounded to 2 places |
| Duplicates | Removed **after** standardisation, so formatting-only duplicates are caught |
| Missing target | Row dropped |
| Missing values | Median (numeric, robust to skew/outliers) and mode (categorical) |
| Logical consistency | Flags (experience > age - 16, expected < previous CTC, placed below the technical threshold) are **reported, not altered** |
| Outliers | IQR rule counted and reported; plausible values kept |

**Leakage control:** two versions are saved. `cleaned_dataset.csv` is imputed with global medians/modes (EDA/SQL/dashboard).
For modelling the imputer, scaler and encoder are part of the sklearn `Pipeline`, so they are fitted on training folds only.

### 3.2 Feature engineering (`src/features.py`)

`interview_score` (mean of technical, aptitude, communication), `avg_academic_pct`, `ctc_gap_lpa`, `ctc_ratio`,
`experience_category` (Fresher / Junior / Senior), `academic_band`, `skills_match_level` (Low / Medium / High),
`interview_performance` (Low / Medium / High) and `placement_probability_score`.

The probability score is an **out-of-fold** prediction (5-fold) so no candidate is scored by a model that saw their label, and it
is **never fed back as a model input** (that would be target leakage).

### 3.3 Exploratory analysis (`src/eda.py`)

28 figures in `reports/figures/` covering missing values, target balance, distributions, outliers, correlations, interview score vs
acceptance, skills match vs placement, company tier, experience, competition, certifications, specialization, tier x competition heat-map,
and score distributions by outcome. `reports/eda_guide_answers.md` answers the 15 questions of the supplied EDA guide using the data.

### 3.4 Modelling (`src/modeling.py`)

* Target: `Placed` = 1, `Not Placed` = 0. Stratified 80/20 train/test split (seed 42).
* Inputs: 27 features - raw columns except `age_years`/`gender`, plus the numeric engineered features.
* Three candidates: Logistic Regression, Random Forest, Histogram Gradient Boosting. Compared by stratified CV ROC-AUC on the
  training split; the best is tuned with `RandomizedSearchCV` and evaluated **once** on the untouched test set.
* Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC curves (decision threshold 0.5, no class re-weighting so probabilities stay interpretable).
* Explainability: permutation importance on the test set (model-agnostic) and logistic-regression coefficients / odds ratios for direction of effect.
* Fairness: per-gender predicted-positive rate, true-positive rate and mean probability on the test set.

### 3.5 SQL storage (`src/sql_loader.py`, `sql/`)

`schema.sql` defines `job_placement_cleaned` with proper types (DECIMAL for scores/CTC, TINYINT/SMALLINT for counts, VARCHAR for labels),
a primary key and indexes on the most-filtered columns. Rows are inserted with SQLAlchemy + PyMySQL. `analysis_queries.sql` holds the
reporting queries (KPIs, tier / competition / experience / skills / certification analysis, risk bands, candidates to prioritise).

### 3.6 Streamlit dashboard (`app/streamlit_app.py`)

**KPIs:** Total Candidates, Placement Rate, Job Acceptance Rate, Average Interview Score, Average Skills Match %, Offer Dropout Rate,
High-Risk Candidate %.
**Filters:** specialization, company tier, competition, gender, experience category, eligibility threshold, risk cut-off.
**Tabs:** Overview · Candidate performance · Placement & acceptance · Interview & evaluation · Model insights · Predict a candidate · Data.

## 4. KPI definitions and assumptions

| KPI | Definition |
|---|---|
| Total candidates | rows after filters |
| Placement rate | placed / all candidates |
| Job acceptance rate | placed / candidates with `technical_score >= threshold` (default 50) |
| Offer dropout rate | 100 % - acceptance rate (eligible candidates not placed) |
| Avg interview score | mean of `interview_score` |
| Avg skills match % | mean of `skills_match_percentage` |
| High-risk candidate % | share with predicted placement probability < 0.30 (adjustable) |

**Why these definitions?** The dataset has only the final outcome (`status`) - there is no separate "offer made" column - and the
technical-score threshold is mentioned but not given. These are therefore *documented assumptions* and are adjustable. Please
confirm them with your mentor/evaluator; `reports/data_understanding.md` shows the placement rate by technical-score band, which
helps identify the real threshold.

## 5. Evaluation-criteria mapping

| Criterion (weight) | Where it is addressed |
|---|---|
| Data cleaning & EDA (25 %) | `preprocessing.py` (+ `reports/data_quality_report.json`), `features.py`, `eda.py`, consistency flags |
| Machine learning (45 %) | `modeling.py`, `reports/model_metrics.json`, importance + direction + fairness files |
| Visualisation & dashboard (10 %) | `app/streamlit_app.py` |
| Documentation & communication (10 %) | this file, `docs/RESULTS_REPORT.md`, `README.md`, `reports/eda_guide_answers.md` |
| Organisation & reproducibility (10 %) | folder layout, `requirements.txt`, fixed random seeds, `run_pipeline.py`, `.env.example`, VS Code launch configs |

## 6. Limitations & ethical notes

* Associations in historical data are not causes; use the score as decision support with a human in the loop.
* Age and gender are excluded from the model; proxies could still exist, so the gender audit should be reviewed with each retrain.
* No date column: no time trend or seasonality analysis is possible.
* Probability cut-offs (0.5 for the class decision, 0.30 / 0.60 for risk bands) are business choices, not statistical truths.
