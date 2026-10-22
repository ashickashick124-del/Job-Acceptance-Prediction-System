"""Feature engineering (Step 5). Pure functions so the same code is used in
training, the analytical dataset and the Streamlit prediction form."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src.utils import banner

ENGINEERED_NUMERIC = ["avg_academic_pct", "interview_score", "ctc_gap_lpa", "ctc_ratio"]
ENGINEERED_CATEGORICAL = [
    "experience_category", "academic_band", "skills_match_level", "interview_performance",
]

# Model inputs: raw columns minus demographic ones, plus numeric engineered features.
# (Banded categories are kept for EDA/dashboard only - they add no information to trees/LR.)
MODEL_NUMERIC = [c for c in C.NUMERIC_COLUMNS if c not in C.MODEL_EXCLUDED] + ENGINEERED_NUMERIC
MODEL_CATEGORICAL = [c for c in C.CATEGORICAL_COLUMNS if c not in C.MODEL_EXCLUDED]
MODEL_FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL


def _cut(series: pd.Series, bins, labels, right=False) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, right=right).astype(object)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived analytical features. NaNs propagate (they are imputed later)."""
    out = df.copy()

    # Mean of the three assessments the company sees at interview time.
    out["interview_score"] = out[["technical_score", "aptitude_score", "communication_score"]].mean(axis=1).round(2)
    out["avg_academic_pct"] = out[["ssc_percentage", "hsc_percentage", "degree_percentage"]].mean(axis=1).round(2)
    out["ctc_gap_lpa"] = (out["expected_ctc_lpa"] - out["previous_ctc_lpa"]).round(2)
    prev = out["previous_ctc_lpa"].where(out["previous_ctc_lpa"] > 0)
    out["ctc_ratio"] = (out["expected_ctc_lpa"] / prev).round(3)

    # Experience: 0 yrs = Fresher, 1-3 = Junior, 4+ = Senior
    out["experience_category"] = _cut(out["years_of_experience"], [-np.inf, 0.5, 3.5, np.inf],
                                      ["Fresher", "Junior", "Senior"])
    out["academic_band"] = _cut(out["avg_academic_pct"], [-np.inf, 60, 70, 80, np.inf],
                                ["Average", "Good", "Very Good", "Excellent"])
    out["skills_match_level"] = _cut(out["skills_match_percentage"], [-np.inf, 60, 80, np.inf],
                                     ["Low", "Medium", "High"])
    out["interview_performance"] = _cut(out["interview_score"], [-np.inf, 60, 70, np.inf],
                                        ["Low", "Medium", "High"])
    return out


def assign_risk_band(prob: pd.Series | np.ndarray) -> pd.Series:
    """High risk = low predicted probability of accepting / being placed."""
    p = pd.Series(prob)
    return pd.cut(p, bins=[-np.inf, C.HIGH_RISK_PROB, C.MEDIUM_RISK_PROB, np.inf],
                  labels=["High Risk", "Medium Risk", "Low Risk"], right=False).astype(object)


def main() -> None:
    banner("STEP 3 - Feature engineering")
    cleaned = pd.read_csv(C.CLEANED_CSV)
    analytical = add_features(cleaned)
    analytical.to_csv(C.ANALYTICAL_CSV, index=False)
    print(f"Saved {C.ANALYTICAL_CSV.relative_to(C.ROOT)}  shape={analytical.shape}")
    for col in ENGINEERED_CATEGORICAL:
        print(f"\n{col}:\n{analytical[col].value_counts(dropna=False).to_string()}")


if __name__ == "__main__":
    main()
