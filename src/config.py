"""Central configuration: paths, column groups, category aliases and business assumptions."""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
# You can keep the CSV anywhere and point to it with the HR_RAW_CSV environment variable.
RAW_CSV = Path(os.getenv("HR_RAW_CSV", str(DATA_DIR / "raw" / "HR_Job_Placement_Dataset.csv")))
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
SQL_DIR = ROOT / "sql"

STANDARDIZED_CSV = INTERIM_DIR / "standardized_with_nulls.csv"  # clean labels, NaNs kept (for ML)
CLEANED_CSV = PROCESSED_DIR / "cleaned_dataset.csv"             # fully imputed (for EDA / SQL)
ANALYTICAL_CSV = PROCESSED_DIR / "analytical_dataset.csv"       # cleaned + engineered + model score
SQLITE_DB = PROCESSED_DIR / "hr_placement.db"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

DATA_UNDERSTANDING_MD = REPORTS_DIR / "data_understanding.md"
DATA_QUALITY_JSON = REPORTS_DIR / "data_quality_report.json"
IMPUTATION_JSON = REPORTS_DIR / "imputation_values.json"
METRICS_JSON = REPORTS_DIR / "model_metrics.json"
EDA_SUMMARY_JSON = REPORTS_DIR / "eda_summary.json"
EDA_ANSWERS_MD = REPORTS_DIR / "eda_guide_answers.md"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
FEATURE_DIRECTION_CSV = REPORTS_DIR / "feature_direction_logreg.csv"
FAIRNESS_CSV = REPORTS_DIR / "fairness_by_gender.csv"
RESULTS_REPORT_MD = DOCS_DIR / "RESULTS_REPORT.md"

TABLE_NAME = "job_placement_cleaned"


def ensure_dirs() -> None:
    for p in (INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, DOCS_DIR):
        p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- modelling
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ASSUMPTIONS (change here, or in the dashboard sidebar) --------------------------
# The dataset description says candidates below a technical-score threshold are not
# eligible. The threshold value is not given, so 50 is used for the dashboard KPIs.
TECHNICAL_THRESHOLD = 50
# Candidates whose predicted acceptance probability is below this are "high risk".
HIGH_RISK_PROB = 0.30
MEDIUM_RISK_PROB = 0.60

# --------------------------------------------------------------------------- columns
ID_COL = "candidate_id"
TARGET_COL = "status"
TARGET_BIN = "accepted"  # Placed -> 1, Not Placed -> 0

NUMERIC_COLUMNS = [
    "age_years", "ssc_percentage", "hsc_percentage", "degree_percentage",
    "technical_score", "aptitude_score", "communication_score",
    "skills_match_percentage", "certifications_count", "years_of_experience",
    "previous_ctc_lpa", "expected_ctc_lpa", "notice_period_days", "employment_gap_months",
]
INTEGER_COLUMNS = [
    "age_years", "certifications_count", "years_of_experience",
    "notice_period_days", "employment_gap_months",
]
FLOAT_COLUMNS = [c for c in NUMERIC_COLUMNS if c not in INTEGER_COLUMNS]

CATEGORICAL_COLUMNS = [
    "gender", "degree_specialization", "internship_experience",
    "career_switch_willingness", "relevant_experience", "company_tier",
    "job_role_match", "competition_level", "bond_requirement",
    "layoff_history", "relocation_willingness",
]
ALL_COLUMNS = [
    "age_years", "gender", "ssc_percentage", "hsc_percentage", "degree_percentage",
    "degree_specialization", "technical_score", "aptitude_score", "communication_score",
    "skills_match_percentage", "certifications_count", "internship_experience",
    "years_of_experience", "career_switch_willingness", "relevant_experience",
    "previous_ctc_lpa", "expected_ctc_lpa", "company_tier", "job_role_match",
    "competition_level", "bond_requirement", "notice_period_days", "layoff_history",
    "employment_gap_months", "relocation_willingness", "status",
]

# Valid (inclusive) numeric ranges; values outside are treated as noise -> missing.
VALID_RANGES = {
    "age_years": (16, 70),
    "ssc_percentage": (0, 100), "hsc_percentage": (0, 100), "degree_percentage": (0, 100),
    "technical_score": (0, 100), "aptitude_score": (0, 100), "communication_score": (0, 100),
    "skills_match_percentage": (0, 100),
    "certifications_count": (0, 50), "years_of_experience": (0, 50),
    "previous_ctc_lpa": (0.1, 500), "expected_ctc_lpa": (0.1, 500),
    "notice_period_days": (0, 365), "employment_gap_months": (0, 240),
}

# Fairness: demographic columns are NOT used as model inputs (see dataset description).
MODEL_EXCLUDED = ["age_years", "gender"]


def _aliases(canonical: dict[str, list[str]]) -> dict[str, str]:
    """Build {normalised-key: canonical label}. Key = lowercase letters/digits only."""
    import re

    out: dict[str, str] = {}
    for label, variants in canonical.items():
        for v in [label, *variants]:
            out[re.sub(r"[^a-z0-9]", "", v.lower())] = label
    return out


_YES_NO = _aliases({"Yes": ["y", "true", "1"], "No": ["n", "false", "0"]})
_WILLING = _aliases({"Willing": ["yes", "y", "true"], "Not Willing": ["unwilling", "no", "n", "false"]})

CATEGORY_ALIASES: dict[str, dict[str, str]] = {
    "gender": _aliases({"Male": ["m"], "Female": ["f"]}),
    "degree_specialization": _aliases({
        "Computer Science": ["cs", "cse", "computerscience"],
        "Information Technology": ["it", "infotech"],
        "Electronics": ["ece", "eee", "electronic"],
        "Mechanical": ["mech"],
        "Others": ["other", "misc"],
    }),
    "internship_experience": _YES_NO,
    "career_switch_willingness": _WILLING,
    "relevant_experience": _aliases({"Relevant": [], "Not Relevant": ["irrelevant", "notrelevant"]}),
    "company_tier": _aliases({"Tier 1": ["1", "t1"], "Tier 2": ["2", "t2"], "Tier 3": ["3", "t3"]}),
    "job_role_match": _aliases({"Matched": ["match"], "Not Matched": ["unmatched", "mismatch", "mismatched"]}),
    "competition_level": _aliases({"Low": [], "Medium": ["med", "moderate"], "High": []}),
    "bond_requirement": _aliases({"Required": ["yes"], "Not Required": ["no", "none"]}),
    "layoff_history": _YES_NO,
    "relocation_willingness": _WILLING,
}
STATUS_ALIASES = _aliases({
    "Placed": ["accepted", "jobaccepted", "1"],
    "Not Placed": ["rejected", "jobrejected", "notaccepted", "0"],
})

# Display orders used by EDA and the dashboard
ORDERS = {
    "company_tier": ["Tier 1", "Tier 2", "Tier 3"],
    "competition_level": ["Low", "Medium", "High"],
    "experience_category": ["Fresher", "Junior", "Senior"],
    "academic_band": ["Average", "Good", "Very Good", "Excellent"],
    "skills_match_level": ["Low", "Medium", "High"],
    "interview_performance": ["Low", "Medium", "High"],
    "risk_band": ["High Risk", "Medium Risk", "Low Risk"],
}
