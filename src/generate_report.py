"""Step 9: build docs/RESULTS_REPORT.md from the artefacts of THIS run.

Every number in the report is read from reports/*.json|csv, so it always matches the data you ran
the pipeline on. Narrative sentences are generated from those numbers (no hard-coded findings).
Optionally converts to .docx if `pandoc` is installed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config as C
from src.utils import banner, load_json, md_table

FIG = "../reports/figures/"


def _rates(eda: dict, key: str) -> pd.DataFrame:
    return pd.DataFrame(eda["group_rates"][key])


def _spread(eda: dict, key: str, label: str, min_gap: float = 2.0) -> str:
    t = _rates(eda, key).sort_values("rate")
    lo, hi = t.iloc[0], t.iloc[-1]
    gap = hi["rate"] - lo["rate"]
    if gap < min_gap:
        return f"- **{label}:** no material difference between groups (spread {gap:.1f} percentage points)."
    return (f"- **{label}:** highest placement rate for *{hi['group']}* ({hi['rate']:.1f}%), lowest for "
            f"*{lo['group']}* ({lo['rate']:.1f}%) - a gap of {gap:.1f} percentage points.")


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def main() -> None:
    banner("STEP 7 - Results report")
    C.ensure_dirs()
    q, m, eda = load_json(C.DATA_QUALITY_JSON), load_json(C.METRICS_JSON), load_json(C.EDA_SUMMARY_JSON)
    imp = pd.read_csv(C.FEATURE_IMPORTANCE_CSV)
    direction = pd.read_csv(C.FEATURE_DIRECTION_CSV)
    fair = pd.read_csv(C.FAIRNESS_CSV)
    fm = m["final_test_metrics"]
    cm = fm["confusion_matrix"]

    comp = pd.DataFrame(m["baseline_comparison"]).set_index("model")[
        ["cv_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]]
    comp.columns = ["CV ROC-AUC", "Test accuracy", "Test precision", "Test recall", "Test F1", "Test ROC-AUC"]
    top = imp.head(10).copy()
    top["importance_mean"] = top["importance_mean"].round(4)
    top = top[["feature", "importance_mean"]].rename(columns={"feature": "Feature", "importance_mean": "Drop in ROC-AUC when shuffled"})

    dir_tbl = direction.head(10).copy()
    dir_tbl["effect"] = dir_tbl["coefficient"].map(lambda c: "raises acceptance odds" if c > 0 else "lowers acceptance odds")
    dir_tbl = dir_tbl[["feature", "coefficient", "odds_ratio", "effect"]]

    gap_bits = []
    for key, label in [("bond_requirement", "Service bond"), ("job_role_match", "Job-role match"),
                       ("relevant_experience", "Relevant experience"), ("relocation_willingness", "Relocation willingness"),
                       ("career_switch_willingness", "Career-switch willingness"), ("company_tier", "Company tier"),
                       ("competition_level", "Competition level"), ("experience_category", "Experience category"),
                       ("skills_match_level", "Skills-match level"), ("interview_performance", "Interview performance"),
                       ("academic_band", "Academic band"), ("degree_specialization", "Degree specialization"),
                       ("layoff_history", "Layoff history")]:
        if key in eda["group_rates"]:
            gap_bits.append(_spread(eda, key, label))

    cons = q["consistency_flags"]
    fixed = {k: v["values_fixed"] for k, v in q["categorical"].items() if v["values_fixed"]}
    miss = q["missing_before_imputation"]
    corr = eda["top_correlations_with_target"]

    # recommendations - only emitted when the data supports them
    recs = []
    t = _rates(eda, "bond_requirement") if "bond_requirement" in eda["group_rates"] else None
    if t is not None and len(t) == 2:
        d = t.set_index("group")["rate"]
        if "Required" in d and "Not Required" in d and d["Not Required"] - d["Required"] >= 2:
            recs.append(f"**Review service bonds.** Offers with a bond convert at {d['Required']:.1f}% versus "
                        f"{d['Not Required']:.1f}% without one.")
    t = _rates(eda, "job_role_match") if "job_role_match" in eda["group_rates"] else None
    if t is not None and len(t) == 2:
        d = t.set_index("group")["rate"]
        if "Matched" in d and "Not Matched" in d and d["Matched"] - d["Not Matched"] >= 2:
            recs.append(f"**Match roles to profiles before making offers.** Matched roles convert at {d['Matched']:.1f}% "
                        f"versus {d['Not Matched']:.1f}% for mismatched roles.")
    tf = top["Feature"].tolist()[:3]
    recs.append(f"**Prioritise screening on the strongest model drivers:** {', '.join(f'`{f}`' for f in tf)}.")
    hr = m["risk_bands_%"].get("High Risk")
    if hr is not None:
        recs.append(f"**Target retention effort at the high-risk group** ({hr:.1f}% of candidates have a predicted "
                    f"probability below {C.HIGH_RISK_PROB:.0%}); contact them earlier and address the drivers above.")
    recs.append("**Use the score as decision support, not as an automatic filter** - age and gender are excluded from the "
                "model and a gender audit is included below, but any HR use should keep a human in the loop.")

    L = [
        f"# Job Acceptance Prediction System - Results Report\n",
        f"*Generated on {date.today():%d %B %Y} by `src/generate_report.py` from the artefacts of the latest pipeline run.*\n",
        "## 1. Executive summary\n",
        f"- Dataset: **{q['rows_final']:,}** candidates after cleaning (raw rows {q['rows_raw']:,}); "
        f"**{m['accepted_rate']:.1%}** were placed (job accepted).",
        f"- Best model: **{m['best_baseline']}**{' (tuned)' if m['tuned'] else ''} - on a held-out test set of {m['test_rows']:,} "
        f"candidates it reaches ROC-AUC **{fm['roc_auc']:.3f}**, accuracy **{fm['accuracy']:.3f}**, precision **{fm['precision']:.3f}**, "
        f"recall **{fm['recall']:.3f}**, F1 **{fm['f1']:.3f}** (decision threshold {m['decision_threshold']}).",
        f"- Most influential features (permutation importance): " + ", ".join(f"`{f}`" for f in top["Feature"].head(5)) + ".",
        f"- Highest linear correlations with placement: " + ", ".join(f"`{k}` ({v:+.2f})" for k, v in list(corr.items())[:5]) + ".\n",
        "## 2. Data understanding & quality\n",
        f"- Raw file: {q['rows_raw']:,} rows x 26 columns; exact duplicate rows in raw: {q['exact_duplicates_in_raw']:,}; "
        f"removed after standardisation (catches formatting-only duplicates): {q['duplicates_removed_after_standardisation']:,}.",
        f"- Rows dropped for missing target: {q['rows_dropped_missing_target']:,}.",
        f"- Missing cells before imputation: " + (", ".join(f"`{k}` {v:,} ({q['missing_pct_before_imputation'][k]}%)" for k, v in miss.items()) or "none") + ".",
        f"- Inconsistent labels repaired (case / whitespace / synonyms): " + (", ".join(f"`{k}` {v:,}" for k, v in fixed.items()) or "none") + ".",
        f"- Out-of-range numeric values set to missing: {sum(q['out_of_range_set_to_null'].values()):,}.",
        f"- Logical-consistency flags (reported, not altered): {cons}.\n",
        f"![Missing values]({FIG}missing_values.png)\n",
        "## 3. Preprocessing\n",
        "1. Column names normalised; categorical labels mapped to canonical values (e.g. `yes ` -> `Yes`, `tier 2 ` -> `Tier 2`).",
        "2. Numeric columns coerced to numbers; impossible values (outside valid ranges in `config.py`) set to missing; floats rounded to 2 decimals.",
        "3. Duplicates removed *after* standardisation.",
        "4. Missing values: median (numeric) / mode (categorical). For modelling the imputer is fitted inside each training fold "
        "(sklearn `Pipeline`), so no information leaks from validation/test rows.",
        "5. One-hot encoding for categoricals, standard scaling for numerics (model pipeline).\n",
        "## 4. Feature engineering\n",
        "| Feature | Definition |\n|---|---|\n"
        "| `interview_score` | mean of technical, aptitude and communication scores |\n"
        "| `avg_academic_pct` | mean of SSC, HSC and degree percentages |\n"
        "| `ctc_gap_lpa`, `ctc_ratio` | expected minus previous CTC; expected / previous CTC |\n"
        "| `experience_category` | Fresher (0 yrs), Junior (1-3), Senior (4+) |\n"
        "| `academic_band` | Average (<60), Good (60-70), Very Good (70-80), Excellent (80+) |\n"
        "| `skills_match_level` | Low (<60), Medium (60-80), High (80+) |\n"
        "| `interview_performance` | Low (<60), Medium (60-70), High (70+) on `interview_score` |\n"
        "| `placement_probability_score` | out-of-fold model probability of being placed (analysis/dashboard only, never a model input) |\n",
        "## 5. Exploratory analysis - key findings\n",
        f"Overall placement rate: **{eda['overall_placement_rate_%']:.1f}%**.\n",
        *gap_bits, "",
        f"![Correlation]({FIG}correlation_heatmap.png)\n",
        f"![Company tier]({FIG}rate_company_tier.png)\n",
        f"![Interview score]({FIG}decile_interview_score.png)\n",
        f"![Skills match]({FIG}decile_skills_match.png)\n",
        f"![Scores by outcome]({FIG}scores_by_outcome.png)\n",
        "The full set of figures is in `reports/figures/`; answers to the 15-question EDA guide are in `reports/eda_guide_answers.md`.\n",
        "## 6. Modelling\n",
        f"Target: `status` (Placed = 1, Not Placed = 0). Train/test split {1 - C.TEST_SIZE:.0%}/{C.TEST_SIZE:.0%}, stratified; "
        f"{m['cv_folds']}-fold stratified CV on the training split. Features used ({len(m['features_used'])}): demographic columns "
        f"{m['excluded_from_model']} are excluded.\n",
        md_table(comp, floatfmt=".4f"), "",
        (f"The best baseline by CV ROC-AUC was **{m['best_baseline']}**; it was tuned with randomised search "
         f"(best parameters: `{m['tuned_params']}`)." if m["tuned"] else f"Best baseline: **{m['best_baseline']}** (no tuning)."),
        "",
        f"**Final test metrics:** accuracy {fm['accuracy']:.4f}, precision {fm['precision']:.4f}, recall {fm['recall']:.4f}, "
        f"F1 {fm['f1']:.4f}, ROC-AUC {fm['roc_auc']:.4f}, PR-AUC {fm['pr_auc']:.4f}. "
        f"Confusion matrix: TN {cm['tn']:,}, FP {cm['fp']:,}, FN {cm['fn']:,}, TP {cm['tp']:,}.\n",
        f"![Model comparison]({FIG}model_comparison.png)\n",
        f"![ROC]({FIG}roc_curves.png)\n",
        f"![Confusion matrix]({FIG}confusion_matrix.png)\n",
        "## 7. Explainability\n",
        "Permutation importance on the held-out test set (how much ROC-AUC falls when a feature is shuffled):\n",
        md_table(top, index=False, floatfmt=".4f"), "",
        f"![Feature importance]({FIG}feature_importance.png)\n",
        "Direction of effect from the logistic-regression baseline (numeric features are standardised, so odds ratios are per 1 standard deviation; "
        "categorical dummies are relative to the dropped baseline):\n",
        md_table(dir_tbl, index=False, floatfmt=".3f"), "",
        "## 8. Fairness check (gender)\n",
        "Gender and age are **not** model inputs. The table checks that predictions do not differ sharply by gender on the test set.\n",
        md_table(fair, index=False, floatfmt=".2f"), "",
        "## 9. Business recommendations\n",
        *[f"{i}. {r}" for i, r in enumerate(recs, 1)], "",
        "## 10. Assumptions & limitations\n",
        f"- The technical-score eligibility threshold is not specified; **{C.TECHNICAL_THRESHOLD}** is assumed for the dashboard KPIs (adjustable in the sidebar / `config.py`).",
        "- The target `status` is the final outcome only: there is no separate 'offer made' field, so *Job Acceptance Rate* is defined as placed / eligible candidates and *Offer Dropout Rate* as its complement.",
        "- The data contains no dates, so trends over time and seasonality cannot be analysed.",
        "- Probabilities are produced by a model trained on this dataset; they describe associations, not causes.",
        f"- High-risk = predicted probability below {C.HIGH_RISK_PROB:.2f}; this cut-off is a business choice.",
        "- Imputation uses global medians/modes for the EDA/SQL dataset; the model itself imputes inside cross-validation.\n",
    ]
    C.RESULTS_REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {C.RESULTS_REPORT_MD.relative_to(C.ROOT)}")

    pandoc = shutil.which("pandoc")
    if pandoc:
        out = C.RESULTS_REPORT_MD.with_suffix(".docx")
        try:
            subprocess.run([pandoc, str(C.RESULTS_REPORT_MD), "-o", str(out), "--resource-path", str(C.DOCS_DIR)],
                           check=True, capture_output=True)
            print(f"Wrote {out.relative_to(C.ROOT)}")
        except subprocess.CalledProcessError as e:
            print("pandoc conversion skipped:", e.stderr.decode()[:200])


if __name__ == "__main__":
    main()
