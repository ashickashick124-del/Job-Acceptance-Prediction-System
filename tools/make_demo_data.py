"""Generate a SYNTHETIC stand-in for HR_Job_Placement_Dataset.csv.

Only for smoke-testing the pipeline when you do not have the real file yet.
It mimics the structure and the data-quality problems of the real data
(case / whitespace variants, missing values, duplicates, long decimals) but the
relationships in it are invented - never report its results as real findings.

    python tools/make_demo_data.py --out data/raw/demo_dataset.csv
    set HR_RAW_CSV=data/raw/demo_dataset.csv      (Windows)   /   export HR_RAW_CSV=...  (Mac/Linux)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 50_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clip = lambda a, lo, hi: np.clip(a, lo, hi)  # noqa: E731

    df = pd.DataFrame({
        "age_years": rng.integers(21, 35, n),
        "gender": rng.choice(["Male", "Female"], n),
        "ssc_percentage": clip(rng.normal(70, 8, n), 50, 92),
        "hsc_percentage": clip(rng.normal(70, 8, n), 50, 92),
        "degree_percentage": clip(rng.normal(74, 7, n), 55, 92),
        "degree_specialization": rng.choice(["Computer Science", "Information Technology", "Electronics", "Mechanical", "Others"], n),
        "technical_score": clip(rng.normal(67, 12, n), 40, 100),
        "aptitude_score": clip(rng.normal(63, 9, n), 40, 100),
        "communication_score": clip(rng.normal(67, 10, n), 45, 100),
        "skills_match_percentage": clip(rng.normal(74, 12, n), 45, 100),
        "certifications_count": clip(rng.poisson(1.3, n), 0, 5),
        "internship_experience": rng.choice(["Yes", "No"], n),
        "years_of_experience": rng.integers(0, 6, n),
        "career_switch_willingness": rng.choice(["Willing", "Not Willing"], n, p=[0.4, 0.6]),
        "relevant_experience": rng.choice(["Relevant", "Not Relevant"], n, p=[0.6, 0.4]),
        "previous_ctc_lpa": clip(rng.normal(5, 2.5, n), 1, 12),
        "company_tier": rng.choice(["Tier 1", "Tier 2", "Tier 3"], n),
        "job_role_match": rng.choice(["Matched", "Not Matched"], n, p=[0.65, 0.35]),
        "competition_level": rng.choice(["Low", "Medium", "High"], n, p=[0.45, 0.4, 0.15]),
        "bond_requirement": rng.choice(["Required", "Not Required"], n, p=[0.25, 0.75]),
        "notice_period_days": rng.choice([0, 15, 30, 60, 90], n, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
        "layoff_history": rng.choice(["Yes", "No"], n, p=[0.2, 0.8]),
        "employment_gap_months": rng.choice([0, 3, 6, 12, 18], n, p=[0.55, 0.15, 0.15, 0.1, 0.05]),
        "relocation_willingness": rng.choice(["Willing", "Not Willing"], n),
    })
    df["expected_ctc_lpa"] = clip(df["previous_ctc_lpa"] * rng.uniform(0.6, 2.0, n) + rng.normal(0, 1, n), 3, 18)

    # INVENTED relationship between features and outcome
    logit = (-0.3 + 0.045 * (df.technical_score - 65) + 0.03 * (df.skills_match_percentage - 74)
             + 0.7 * (df.job_role_match == "Matched") + 0.5 * (df.relevant_experience == "Relevant")
             - 0.6 * (df.bond_requirement == "Required") - 0.25 * (df.expected_ctc_lpa - df.previous_ctc_lpa).clip(0)
             - 0.01 * df.notice_period_days - 0.04 * df.employment_gap_months
             - 0.4 * (df.competition_level == "High") + 0.25 * (df.company_tier == "Tier 1")
             + 0.15 * df.certifications_count + rng.normal(0, 0.8, n))
    placed = (rng.random(n) < 1 / (1 + np.exp(-logit))) & (df.technical_score >= 50)
    df["status"] = np.where(placed, "Placed", "Not Placed")

    # ---- inject realistic mess -------------------------------------------------------
    m = lambda p: rng.random(n) < p  # noqa: E731
    df["gender"] = np.where(m(0.08), df["gender"].str.lower(), df["gender"])
    for c in ("internship_experience", "layoff_history"):
        df[c] = np.where(m(0.06), df[c].str.lower() + " ", df[c])
    df["company_tier"] = np.where(m(0.05), df["company_tier"] + " ", df["company_tier"])
    df["expected_ctc_lpa"] = np.where(m(0.1), df["expected_ctc_lpa"].round(10), df["expected_ctc_lpa"].round(6))
    for c in ["ssc_percentage", "hsc_percentage", "relevant_experience", "job_role_match",
              "career_switch_willingness", "relocation_willingness", "layoff_history",
              "notice_period_days", "employment_gap_months"]:
        df[c] = df[c].astype(object).where(~m(0.06), np.nan)
    dups = df.sample(int(n * 0.008), random_state=seed)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/demo_dataset.csv")
    ap.add_argument("--rows", type=int, default=50_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make(a.rows).to_csv(out, index=False)
    print(f"Wrote synthetic demo data -> {out}")
