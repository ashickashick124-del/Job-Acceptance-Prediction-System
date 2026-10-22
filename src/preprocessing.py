"""Step 3: cleaning & preprocessing.

Outputs
-------
data/interim/standardized_with_nulls.csv  labels fixed, noise -> NaN, duplicates removed, NaNs KEPT.
                                          The ML pipeline imputes these inside cross-validation
                                          (no information leaks from the test set).
data/processed/cleaned_dataset.csv        same rows, missing values imputed (median / mode).
                                          Used for EDA, SQL and the dashboard.
reports/data_quality_report.json          everything that was fixed, with counts.
reports/imputation_values.json            the median / mode used for every column.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src.utils import banner, save_json


# --------------------------------------------------------------------------- loading
def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else C.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download HR_Job_Placement_Dataset.csv and place it in data/raw/ "
            "(or set the HR_RAW_CSV environment variable to its location)."
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in C.ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}\nFound: {list(df.columns)}")
    return df[C.ALL_COLUMNS].copy()


# --------------------------------------------------------------------------- cleaning steps
def _key(v) -> str | None:
    if pd.isna(v):
        return None
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _map_labels(raw: pd.Series, aliases: dict[str, str]) -> tuple[pd.Series, dict]:
    keys = raw.map(_key)
    mapped = keys.map(aliases)
    had_value = keys.notna()
    unknown = had_value & mapped.isna()
    differs = (raw.astype("string") != mapped.astype("string")).fillna(False).astype(bool)
    changed = had_value & mapped.notna() & differs
    info = {
        "values_fixed": int(changed.sum()),
        "unrecognised_set_to_null": int(unknown.sum()),
        "unrecognised_examples": {str(k): int(v) for k, v in keys[unknown].value_counts().head(5).items()},
    }
    return mapped.astype(object), info


def standardize_categoricals(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Fix case, whitespace and synonym variants ('yes ', 'MALE', 'tier 2 ' ...)."""
    log["categorical"] = {}
    for col, aliases in C.CATEGORY_ALIASES.items():
        df[col], log["categorical"][col] = _map_labels(df[col], aliases)
    df[C.TARGET_COL], log["target"] = _map_labels(df[C.TARGET_COL], C.STATUS_ALIASES)
    return df


def coerce_numeric(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    log["numeric_unparsable_to_null"] = {}
    for col in C.NUMERIC_COLUMNS:
        was_null = df[col].isna()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        log["numeric_unparsable_to_null"][col] = int((df[col].isna() & ~was_null).sum())
    return df


def enforce_ranges(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Impossible values (e.g. 140 %, negative CTC) are noise -> NaN (imputed later)."""
    log["out_of_range_set_to_null"] = {}
    for col, (lo, hi) in C.VALID_RANGES.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        log["out_of_range_set_to_null"][col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    # Over-long float noise (e.g. 11.2151523302) -> 2 decimals
    df[C.FLOAT_COLUMNS] = df[C.FLOAT_COLUMNS].round(2)
    return df


def drop_duplicates(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Runs AFTER standardisation so 'duplicate-like' rows that only differed in
    formatting (case, spaces, decimals) are caught as well."""
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    log["duplicates_removed_after_standardisation"] = before - len(df)
    return df


def consistency_checks(df: pd.DataFrame, log: dict) -> None:
    """Logical-consistency checks. Flagged and reported, not silently altered."""
    checks = {
        "experience_exceeds_age_minus_16": int((df["years_of_experience"] > df["age_years"] - 16).sum()),
        "expected_ctc_below_previous_ctc": int((df["expected_ctc_lpa"] < df["previous_ctc_lpa"]).sum()),
        "placed_but_technical_below_threshold": int(
            ((df[C.TARGET_COL] == "Placed") & (df["technical_score"] < C.TECHNICAL_THRESHOLD)).sum()
        ),
        "skills_match_above_100_or_below_0": int(((df["skills_match_percentage"] > 100) | (df["skills_match_percentage"] < 0)).sum()),
    }
    log["consistency_flags"] = checks
    log["technical_threshold_used_for_flag"] = C.TECHNICAL_THRESHOLD


def outlier_report(df: pd.DataFrame, log: dict) -> None:
    """IQR rule (1.5 x IQR). Reported only - values are plausible so they are kept."""
    out = {}
    for col in C.NUMERIC_COLUMNS:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[col] = int(((s < lo) | (s > hi)).sum())
    log["iqr_outliers_kept"] = out


# --------------------------------------------------------------------------- imputation
def impute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Numeric -> median (robust to skew/outliers); categorical -> mode."""
    out = df.copy()
    values: dict[str, object] = {}
    for col in C.NUMERIC_COLUMNS:
        med = float(out[col].median())
        values[col] = med
        out[col] = out[col].fillna(med)
    for col in C.CATEGORICAL_COLUMNS:
        mode = out[col].mode(dropna=True)
        values[col] = str(mode.iloc[0])
        out[col] = out[col].fillna(values[col])
    for col in C.INTEGER_COLUMNS:
        out[col] = out[col].round().astype(int)
    for col in C.FLOAT_COLUMNS:
        out[col] = out[col].round(2)
    return out, values


# --------------------------------------------------------------------------- main
def run(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    log: dict = {"rows_raw": int(len(raw)), "exact_duplicates_in_raw": int(raw.duplicated().sum())}
    df = raw.copy()
    df = standardize_categoricals(df, log)
    df = coerce_numeric(df, log)
    df = enforce_ranges(df, log)

    # The target must exist to learn from a row.
    no_target = df[C.TARGET_COL].isna()
    log["rows_dropped_missing_target"] = int(no_target.sum())
    df = df.loc[~no_target].copy()

    df = drop_duplicates(df, log)
    log["missing_before_imputation"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    log["missing_pct_before_imputation"] = {c: round(float(v) * 100, 2) for c, v in df.isna().mean().items() if v}
    consistency_checks(df, log)
    outlier_report(df, log)

    df.insert(0, C.ID_COL, np.arange(1, len(df) + 1))
    df.insert(df.columns.get_loc(C.TARGET_COL) + 1, C.TARGET_BIN, (df[C.TARGET_COL] == "Placed").astype(int))
    cleaned, impute_values = impute(df)
    log["rows_final"] = int(len(df))
    log["class_balance_pct"] = {k: round(float(v) * 100, 2) for k, v in df[C.TARGET_COL].value_counts(normalize=True).items()}
    return df, cleaned, log, impute_values


def main() -> None:
    banner("STEP 2 - Cleaning & preprocessing")
    C.ensure_dirs()
    raw = load_raw()
    standardized, cleaned, log, values = run(raw)

    standardized.to_csv(C.STANDARDIZED_CSV, index=False)
    cleaned.to_csv(C.CLEANED_CSV, index=False)
    save_json(log, C.DATA_QUALITY_JSON)
    save_json(values, C.IMPUTATION_JSON)

    print(f"Raw rows            : {log['rows_raw']:,}")
    print(f"Duplicates removed  : {log['duplicates_removed_after_standardisation']:,} "
          f"(exact in raw: {log['exact_duplicates_in_raw']:,})")
    print(f"Rows dropped (no target): {log['rows_dropped_missing_target']:,}")
    print(f"Final rows          : {log['rows_final']:,}")
    fixed = sum(v["values_fixed"] for v in log["categorical"].values())
    print(f"Categorical labels fixed: {fixed:,}")
    print(f"Missing values imputed  : {sum(log['missing_before_imputation'].values()):,} cells "
          f"in {len(log['missing_before_imputation'])} columns")
    print(f"Class balance (%)   : {log['class_balance_pct']}")
    print(f"Consistency flags   : {log['consistency_flags']}")
    print(f"Saved -> {C.STANDARDIZED_CSV.relative_to(C.ROOT)}, {C.CLEANED_CSV.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
