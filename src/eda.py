"""Step 4 / 8: Exploratory data analysis. Saves figures to reports/figures and a numeric
summary (reports/eda_summary.json) + answers to the 15-question EDA guide (reports/eda_guide_answers.md)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as C
from src.features import ENGINEERED_NUMERIC
from src.utils import banner, load_json, md_table, rate_table, save_json, savefig

sns.set_theme(style="whitegrid", context="notebook")
PLACED_COLOR, NOT_COLOR = "#2E7D32", "#C62828"


# --------------------------------------------------------------------------- plot helpers
def plot_rate(df: pd.DataFrame, col: str, title: str, fname: str, order: list | None = None) -> pd.DataFrame:
    t = rate_table(df, col)
    if order:
        t[col] = pd.Categorical(t[col], categories=[o for o in order if o in set(t[col])], ordered=True)
        t = t.sort_values(col)
    overall = df[C.TARGET_BIN].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(t[col].astype(str), t["rate"], color="#4C78A8")
    ax.axhline(overall, color="grey", ls="--", lw=1, label=f"Overall {overall:.1f}%")
    for b, r, n in zip(bars, t["rate"], t["n"]):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.6, f"{r:.1f}%\n(n={n:,})", ha="center", fontsize=8)
    ax.set_ylim(0, min(100, t["rate"].max() * 1.3 + 5))
    ax.set_ylabel("Placement / acceptance rate (%)")
    ax.set_xlabel(col.replace("_", " "))
    ax.set_title(title)
    ax.legend()
    savefig(fig, fname)
    return t


def plot_decile_rate(df: pd.DataFrame, col: str, title: str, fname: str, q: int = 10) -> pd.DataFrame:
    d = df[[col, C.TARGET_BIN]].copy()
    d["bin"] = pd.qcut(d[col], q, duplicates="drop")
    t = d.groupby("bin", observed=True).agg(mid=(col, "mean"), rate=(C.TARGET_BIN, "mean"), n=(C.TARGET_BIN, "size")).reset_index()
    t["rate"] *= 100
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t["mid"], t["rate"], marker="o", color="#4C78A8")
    ax.axhline(df[C.TARGET_BIN].mean() * 100, color="grey", ls="--", lw=1)
    ax.set_xlabel(f"{col.replace('_', ' ')} (decile means)")
    ax.set_ylabel("Placement rate (%)")
    ax.set_title(title)
    savefig(fig, fname)
    return t.drop(columns="bin")


def main() -> None:
    banner("STEP 5 - Exploratory data analysis")
    C.ensure_dirs()
    df = pd.read_csv(C.ANALYTICAL_CSV)
    quality = load_json(C.DATA_QUALITY_JSON)
    y = C.TARGET_BIN
    summary: dict = {"rows": len(df), "overall_placement_rate_%": df[y].mean() * 100}

    # 1. missing values (before imputation) ---------------------------------------------
    miss = pd.Series(quality["missing_pct_before_imputation"]).sort_values()
    if len(miss):
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(miss) + 1.5))
        ax.barh(miss.index, miss.values, color="#E45756")
        ax.set_xlabel("% missing in raw data")
        ax.set_title("Missing values before imputation")
        savefig(fig, "missing_values.png")

    # 2. target ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4))
    vc = df[C.TARGET_COL].value_counts()
    ax.bar(vc.index, vc.values, color=[PLACED_COLOR if i == "Placed" else NOT_COLOR for i in vc.index])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Target distribution")
    ax.set_ylim(0, vc.max() * 1.18)
    savefig(fig, "target_distribution.png")

    # 3. distributions + outliers ---------------------------------------------------------
    num_cols = C.NUMERIC_COLUMNS + ENGINEERED_NUMERIC
    fig, axes = plt.subplots(5, 4, figsize=(16, 14))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.histplot(df[col], bins=30, ax=ax, color="#4C78A8")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(num_cols):]:
        ax.axis("off")
    fig.suptitle("Numeric distributions", y=1.0)
    fig.tight_layout()
    savefig(fig, "numeric_distributions.png")

    fig, axes = plt.subplots(5, 4, figsize=(16, 14))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.boxplot(x=df[col], ax=ax, color="#9ECAE9", fliersize=2)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(num_cols):]:
        ax.axis("off")
    fig.suptitle("Outlier check (box plots)", y=1.0)
    fig.tight_layout()
    savefig(fig, "outlier_boxplots.png")

    # 4. correlation ------------------------------------------------------------------------
    corr = df[num_cols + [y]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax)
    ax.set_title("Correlation matrix (accepted = 1 if Placed)")
    savefig(fig, "correlation_heatmap.png")
    summary["top_correlations_with_target"] = (corr[y].drop(y).sort_values(key=np.abs, ascending=False)
                                               .head(10).round(3).to_dict())

    # 5. the analyses requested in the brief ------------------------------------------------
    O = C.ORDERS
    rates = {
        "company_tier": plot_rate(df, "company_tier", "Company tier vs acceptance rate", "rate_company_tier.png", O["company_tier"]),
        "competition_level": plot_rate(df, "competition_level", "Competition level vs acceptance rate", "rate_competition.png", O["competition_level"]),
        "experience_category": plot_rate(df, "experience_category", "Experience vs placement", "rate_experience.png", O["experience_category"]),
        "skills_match_level": plot_rate(df, "skills_match_level", "Skills match level vs placement", "rate_skills_level.png", O["skills_match_level"]),
        "interview_performance": plot_rate(df, "interview_performance", "Interview performance vs placement", "rate_interview_cat.png", O["interview_performance"]),
        "academic_band": plot_rate(df, "academic_band", "Academic band vs placement", "rate_academic_band.png", O["academic_band"]),
        "certifications_count": plot_rate(df, "certifications_count", "Certifications vs acceptance", "rate_certifications.png"),
        "degree_specialization": plot_rate(df, "degree_specialization", "Specialization vs placement", "rate_specialization.png"),
    }
    cat_grid = [c for c in C.CATEGORICAL_COLUMNS if c not in ("company_tier", "competition_level", "degree_specialization")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), cat_grid):
        t = rate_table(df, col)
        ax.bar(t[col].astype(str), t["rate"], color="#4C78A8")
        for i, (r, n) in enumerate(zip(t["rate"], t["n"])):
            ax.text(i, r + 0.5, f"{r:.1f}%", ha="center", fontsize=8)
        ax.set_title(col.replace("_", " "), fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes.ravel()[len(cat_grid):]:
        ax.axis("off")
    fig.suptitle("Placement rate (%) by categorical feature", y=1.0)
    fig.tight_layout()
    savefig(fig, "rate_categoricals_grid.png")
    for col in cat_grid:
        rates[col] = rate_table(df, col)

    deciles = {
        "interview_score": plot_decile_rate(df, "interview_score", "Interview score vs placement probability", "decile_interview_score.png"),
        "skills_match_percentage": plot_decile_rate(df, "skills_match_percentage", "Skills match % vs placement", "decile_skills_match.png"),
        "technical_score": plot_decile_rate(df, "technical_score", "Technical score vs placement", "decile_technical.png"),
        "aptitude_score": plot_decile_rate(df, "aptitude_score", "Aptitude (employability) score vs placement", "decile_aptitude.png"),
        "avg_academic_pct": plot_decile_rate(df, "avg_academic_pct", "Academic performance vs placement", "decile_academic.png"),
        "ctc_gap_lpa": plot_decile_rate(df, "ctc_gap_lpa", "Expected - previous CTC gap vs placement", "decile_ctc_gap.png"),
    }

    # candidate performance: scores by outcome
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    cols = ["ssc_percentage", "hsc_percentage", "degree_percentage", "technical_score",
            "aptitude_score", "communication_score", "skills_match_percentage", "interview_score"]
    for ax, col in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x=C.TARGET_COL, y=col, hue=C.TARGET_COL, order=["Not Placed", "Placed"],
                    palette={"Placed": PLACED_COLOR, "Not Placed": NOT_COLOR}, ax=ax, legend=False)
        ax.set_xlabel("")
        ax.set_title(col.replace("_", " "), fontsize=10)
    fig.suptitle("Scores by placement outcome", y=1.0)
    fig.tight_layout()
    savefig(fig, "scores_by_outcome.png")

    # skills match vs interview performance
    s = df.sample(min(4000, len(df)), random_state=C.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=s, x="skills_match_percentage", y="interview_score", hue=C.TARGET_COL,
                    palette={"Placed": PLACED_COLOR, "Not Placed": NOT_COLOR}, alpha=0.45, s=14, ax=ax)
    ax.set_title("Skills match vs interview score")
    savefig(fig, "skills_vs_interview.png")
    summary["corr_skills_vs_interview_score"] = float(df["skills_match_percentage"].corr(df["interview_score"]))

    # tier x competition heat-map
    pv = df.pivot_table(index="company_tier", columns="competition_level", values=y, aggfunc="mean").reindex(
        index=O["company_tier"], columns=O["competition_level"]) * 100
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(pv, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "% placed"})
    ax.set_title("Placement rate: tier x competition")
    savefig(fig, "heatmap_tier_competition.png")

    # model score
    if "placement_probability_score" in df:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x="placement_probability_score", hue=C.TARGET_COL, bins=30, element="step",
                     palette={"Placed": PLACED_COLOR, "Not Placed": NOT_COLOR}, ax=ax)
        ax.axvline(C.HIGH_RISK_PROB, color="k", ls=":", label="High-risk cut-off")
        ax.legend()
        ax.set_title("Predicted placement probability (out-of-fold)")
        savefig(fig, "probability_distribution.png")

    # 6. numbers for the report -------------------------------------------------------------
    summary["group_rates"] = {k: v.assign(**{k: v[k].astype(str)}).rename(columns={k: "group"}).round(2).to_dict(orient="records")
                              for k, v in rates.items()}
    summary["decile_rates"] = {k: v.round(2).to_dict(orient="records") for k, v in deciles.items()}
    summary["numeric_stats"] = df[num_cols].describe().T.round(3).to_dict(orient="index")
    summary["iqr_outliers"] = quality["iqr_outliers_kept"]
    save_json(summary, C.EDA_SUMMARY_JSON)
    write_guide_answers(df, quality, corr, summary)
    print(f"Saved {len(list(C.FIGURES_DIR.glob('*.png')))} figures -> {C.FIGURES_DIR.relative_to(C.ROOT)}")
    print(f"Saved {C.EDA_SUMMARY_JSON.relative_to(C.ROOT)}, {C.EDA_ANSWERS_MD.relative_to(C.ROOT)}")


def write_guide_answers(df: pd.DataFrame, quality: dict, corr: pd.DataFrame, summary: dict) -> None:
    """Answers the questions of the supplied 'EDA Guide' using the actual data."""
    n = len(df)
    num = C.NUMERIC_COLUMNS
    miss = quality["missing_before_imputation"]
    cv = (df[num].std() / df[num].mean().replace(0, np.nan)).round(2).sort_values(ascending=False)
    top_corr = summary["top_correlations_with_target"]
    placed = df[df[C.TARGET_BIN] == 1]
    not_placed = df[df[C.TARGET_BIN] == 0]
    diff = ((placed[num].mean() - not_placed[num].mean()) / df[num].std()).round(2).sort_values(key=np.abs, ascending=False)
    out_n = pd.Series(quality["iqr_outliers_kept"]).sort_values(ascending=False)
    cons = quality["consistency_flags"]
    L = ["# EDA Guide - answers from this dataset\n"]
    L += [
        "## 1-2. Characteristics and structure",
        f"{n:,} candidates after cleaning (raw rows: {quality['rows_raw']:,}); 26 original columns "
        f"({len(num)} numeric, {len(C.CATEGORICAL_COLUMNS)} categorical, 1 target) plus engineered features. "
        f"Class balance: {quality['class_balance_pct']}.\n",
        "## 3. Patterns",
        "Strongest linear associations with placement (correlation with `accepted`): "
        + ", ".join(f"`{k}` ({v:+.2f})" for k, v in list(top_corr.items())[:6]) + ".\n",
        "Standardised mean difference, Placed minus Not Placed (in std-devs): "
        + ", ".join(f"`{k}` {v:+.2f}" for k, v in diff.head(6).items()) + ".\n",
        "## 4. Outliers",
        "IQR-rule outlier counts (kept - values are plausible): "
        + ", ".join(f"`{k}` {v:,}" for k, v in out_n.head(6).items() if v) + ". See `figures/outlier_boxplots.png`.\n",
        "## 5. Missing values",
        "Cells missing before imputation: " + (", ".join(f"`{k}` {v:,} ({quality['missing_pct_before_imputation'][k]}%)"
                                                  for k, v in miss.items()) or "none") +
        ". Numeric -> median, categorical -> mode (imputation values in `reports/imputation_values.json`).\n",
        "## 6. Correctness",
        f"Exact duplicates in raw: {quality['exact_duplicates_in_raw']:,}; removed after standardisation: "
        f"{quality['duplicates_removed_after_standardisation']:,}. Categorical labels repaired: "
        f"{sum(v['values_fixed'] for v in quality['categorical'].values()):,}. Out-of-range values set to null: "
        f"{sum(quality['out_of_range_set_to_null'].values()):,}. Consistency flags: {cons}.\n",
        "## 7. Correlation between variables",
        "See `figures/correlation_heatmap.png`. Strongest feature-feature pairs: "
        + ", ".join(f"`{a}`~`{b}` ({v:+.2f})" for a, b, v in _top_pairs(corr.drop(index=C.TARGET_BIN, columns=C.TARGET_BIN), 4)) + ".\n",
        "## 8-9. Comparison with past performance / seasonality",
        "Not applicable - the dataset has no date/time column.\n",
        "## 10. Variability",
        "Highest coefficient of variation: " + ", ".join(f"`{k}` {v}" for k, v in cv.head(5).items()) + ".\n",
        "## 11-12. Discrepancies and unexpected results",
        f"{cons['expected_ctc_below_previous_ctc']:,} candidates expect less CTC than they earned before; "
        f"{cons['placed_but_technical_below_threshold']:,} placed candidates have technical score below the assumed "
        f"threshold ({quality['technical_threshold_used_for_flag']}). Review these against the real eligibility rule.\n",
        "## 13. Subsets",
        "Placement rate by group is in `eda_summary.json` (`group_rates`) and the `rate_*.png` figures: "
        + ", ".join(f"{k}: " + "/".join(f"{r['group']} {r['rate']:.1f}%" for r in v) for k, v in summary["group_rates"].items()
                    if k in ("company_tier", "competition_level", "experience_category", "bond_requirement", "job_role_match")) + ".\n",
        "## 14. Transformations needed",
        "Median/mode imputation, label standardisation, one-hot encoding of categoricals, standard-scaling of numerics "
        "(inside the model pipeline), derived features (see `src/features.py`).\n",
        "## 15. Gaps in understanding",
        "The exact technical-score eligibility threshold, the definition of 'offer made' (the data only has the final "
        "status) and the time period of the data are not provided - the dashboard KPIs therefore rest on documented assumptions.\n",
    ]
    C.EDA_ANSWERS_MD.write_text("\n".join(L), encoding="utf-8")


def _top_pairs(corr: pd.DataFrame, k: int):
    c = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    c = c.reindex(c.abs().sort_values(ascending=False).index).head(k)
    return [(a, b, v) for (a, b), v in c.items()]


if __name__ == "__main__":
    main()
