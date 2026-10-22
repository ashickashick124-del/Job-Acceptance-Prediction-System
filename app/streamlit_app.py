"""Interactive Streamlit dashboard - Job Acceptance Prediction System.

Run:  streamlit run app/streamlit_app.py
Data: data/processed/analytical_dataset.csv (default) or MySQL when DATA_SOURCE=mysql in .env
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src import config as C
from src.features import add_features, assign_risk_band
from src.utils import load_json

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Job Acceptance Prediction", page_icon="🎯", layout="wide")

# ------------------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    :root {
        --navy: #153B5B;
        --blue: #2589C7;
        --sky: #EAF6FF;
        --pale: #F5FAFF;
        --border: #D8EAF7;
        --text: #20384B;
    }

    .stApp {
        background: linear-gradient(180deg, #F4FAFF 0%, #FFFFFF 38%, #F7FBFF 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 250, 255, 0.88);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 800 !important;
        line-height: 1.2;
    }

    [data-testid="stCaptionContainer"] p {
        color: #607D91;
        font-size: 0.96rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E5F3FF 0%, #F4FAFF 100%);
        border-right: 1px solid #D3E8F8;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #17486C !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #41647D;
    }

    /* KPI metric cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 5px 18px rgba(31, 111, 166, 0.07);
        min-height: 112px;
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(31, 111, 166, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #58758A !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #176FA6 !important;
        font-weight: 800;
    }

    [data-testid="stMetricDelta"] {
        color: #2589C7 !important;
    }

    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 8px;
        border-bottom: 1px solid #D7EAF8;
    }

    [data-testid="stTabs"] button[role="tab"] {
        background: #EAF5FD;
        border: 1px solid #D7EAF8;
        border-radius: 10px 10px 0 0;
        color: #35627F;
        padding: 10px 15px;
        font-weight: 650;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #FFFFFF;
        color: #167AB5;
        border-bottom: 3px solid #2589C7;
    }

    /* Inputs and buttons */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input {
        border-radius: 10px;
        border-color: #C9E2F3;
    }

    .stButton button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #2589C7, #55B5E8);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1.1rem;
        box-shadow: 0 4px 12px rgba(37, 137, 199, 0.18);
    }

    .stButton button:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(135deg, #176FA6, #2589C7);
        color: white;
        border: 0;
    }

    /* Tables, expanders, alerts, and forms */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: #FFFFFF;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 5px 18px rgba(31, 111, 166, 0.05);
    }

    hr {
        border-color: #D8EAF7;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        [data-testid="stMetric"] {
            padding: 12px;
            min-height: 95px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PLACED, NOT_PLACED = "#2589C7", "#E06B78"
COLOR_MAP = {"Placed": PLACED, "Not Placed": NOT_PLACED}
ORD = C.ORDERS


# ------------------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    if os.getenv("DATA_SOURCE", "csv").lower() == "mysql":
        from sqlalchemy import text

        from src.sql_loader import mysql_engine

        with mysql_engine().connect() as conn:
            return pd.read_sql(text(f"SELECT * FROM {C.TABLE_NAME}"), conn)
    return pd.read_csv(C.ANALYTICAL_CSV)


@st.cache_resource
def load_model():
    return joblib.load(C.MODEL_PATH) if C.MODEL_PATH.exists() else None


@st.cache_data
def load_reports() -> dict:
    out = {}
    for key, path in {"metrics": C.METRICS_JSON, "quality": C.DATA_QUALITY_JSON}.items():
        out[key] = load_json(path) if path.exists() else None
    for key, path in {"importance": C.FEATURE_IMPORTANCE_CSV, "direction": C.FEATURE_DIRECTION_CSV, "fairness": C.FAIRNESS_CSV}.items():
        out[key] = pd.read_csv(path) if path.exists() else None
    return out


if not C.ANALYTICAL_CSV.exists() and os.getenv("DATA_SOURCE", "csv").lower() != "mysql":
    st.error("Processed data not found. Put the dataset in `data/raw/` and run `python run_pipeline.py` first.")
    st.stop()

df = load_data()
rep = load_reports()
model_bundle = load_model()


# ------------------------------------------------------------------------------ helpers
def rate_table(d: pd.DataFrame, col: str) -> pd.DataFrame:
    t = d.groupby(col, observed=True)["accepted"].agg(rate="mean", n="size").reset_index()
    t["rate"] *= 100
    return t


def bar_rate(d: pd.DataFrame, col: str, title: str, order: list | None = None, key: str | None = None) -> None:
    t = rate_table(d, col)
    t[col] = t[col].astype(str)
    fig = px.bar(t, x=col, y="rate", text=t["rate"].round(1).astype(str) + "%", hover_data={"n": True, "rate": ":.1f"},
                 category_orders={col: order} if order else None, title=title,
                 labels={"rate": "Placement rate (%)", col: col.replace("_", " ")}, color_discrete_sequence=["#2589C7"])
    overall = d["accepted"].mean() * 100
    fig.add_hline(y=overall, line_dash="dot", line_color="grey", annotation_text=f"overall {overall:.1f}%")
    fig.update_yaxes(range=[0, min(100, max(t["rate"].max() * 1.25, overall * 1.25, 10))])
    fig.update_layout(template="plotly_white", height=380, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch", key=key or f"bar_{col}_{title}")


def decile_line(d: pd.DataFrame, col: str, title: str, q: int = 10, key: str | None = None) -> None:
    x = d[[col, "accepted"]].copy()
    x["bin"] = pd.qcut(x[col], q, duplicates="drop")
    t = x.groupby("bin", observed=True).agg(mean_value=(col, "mean"), rate=("accepted", "mean"), n=("accepted", "size")).reset_index()
    t["rate"] *= 100
    fig = px.line(t, x="mean_value", y="rate", markers=True, hover_data={"n": True}, title=title,
                  labels={"mean_value": col.replace("_", " ") + " (decile mean)", "rate": "Placement rate (%)"})
    fig.add_hline(y=d["accepted"].mean() * 100, line_dash="dot", line_color="grey")
    fig.update_layout(template="plotly_white", height=380, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch", key=key or f"dec_{col}")


# ------------------------------------------------------------------------------ sidebar filters
with st.sidebar:
    st.title("🎯 Filters")
    spec = st.multiselect("Degree specialization", sorted(df["degree_specialization"].unique()), default=sorted(df["degree_specialization"].unique()))
    tier = st.multiselect("Company tier", ORD["company_tier"], default=ORD["company_tier"])
    comp = st.multiselect("Competition level", ORD["competition_level"], default=ORD["competition_level"])
    gender = st.multiselect("Gender", sorted(df["gender"].unique()), default=sorted(df["gender"].unique()))
    exp_cat = st.multiselect("Experience category", ORD["experience_category"], default=ORD["experience_category"])
    st.divider()
    st.subheader("KPI assumptions")
    tech_thr = st.slider("Technical-score eligibility threshold", 30, 80, C.TECHNICAL_THRESHOLD,
                         help="Candidates below this technical score are treated as not eligible (the dataset notes a threshold "
                              "exists but does not state its value).")
    risk_thr = st.slider("High-risk if placement probability below", 0.05, 0.60, float(C.HIGH_RISK_PROB), 0.05)

f = df[df["degree_specialization"].isin(spec) & df["company_tier"].isin(tier) & df["competition_level"].isin(comp)
       & df["gender"].isin(gender) & df["experience_category"].isin(exp_cat)].copy()
if f.empty:
    st.warning("No candidates match the selected filters.")
    st.stop()

# ------------------------------------------------------------------------------ header + KPIs
st.title("Job Acceptance Prediction System")
st.caption("HR analytics - who accepts offers, what drives it, and who is at risk of dropping out.")

eligible = f[f["technical_score"] >= tech_thr]
placement_rate = f["accepted"].mean() * 100
acceptance_rate = eligible["accepted"].mean() * 100 if len(eligible) else 0.0
dropout_rate = 100 - acceptance_rate if len(eligible) else 0.0
high_risk_pct = (f["placement_probability_score"] < risk_thr).mean() * 100 if "placement_probability_score" in f else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total candidates", f"{len(f):,}", help=f"{len(f) / len(df):.0%} of the full dataset after filters")
c2.metric("Placement rate", f"{placement_rate:.1f}%", help="Placed / all candidates")
c3.metric("Job acceptance rate", f"{acceptance_rate:.1f}%", help="Placed / candidates whose technical score meets the eligibility threshold")
c4.metric("Offer dropout rate", f"{dropout_rate:.1f}%", help="Eligible candidates who were not placed (1 - acceptance rate)")
c5, c6, c7, _ = st.columns(4)
c5.metric("Avg interview score", f"{f['interview_score'].mean():.1f}", help="Mean of technical, aptitude and communication scores")
c6.metric("Avg skills match", f"{f['skills_match_percentage'].mean():.1f}%")
c7.metric("High-risk candidates", f"{high_risk_pct:.1f}%", help=f"Predicted placement probability below {risk_thr:.2f}")

tabs = st.tabs(["Overview", "Candidate performance", "Placement & acceptance", "Interview & evaluation",
                "Model insights", "Predict a candidate", "Data"])

# ------------------------------------------------------------------------------ Overview
with tabs[0]:
    a, b = st.columns(2)
    with a:
        vc = f["status"].value_counts().reset_index()
        fig = px.pie(vc, names="status", values="count", hole=0.55, color="status", color_discrete_map=COLOR_MAP, title="Outcome split")
        fig.update_layout(template="plotly_white", height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with b:
        bar_rate(f, "degree_specialization", "Placement rate by specialization")
    num_cols = [c for c in C.NUMERIC_COLUMNS if c != "age_years"] + ["interview_score", "avg_academic_pct", "ctc_gap_lpa", "accepted"]
    corr = f[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation matrix (accepted = 1 if placed)")
    fig.update_layout(height=620, template="plotly_white")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------ Candidate performance
with tabs[1]:
    a, b = st.columns(2)
    with a:
        bar_rate(f, "academic_band", "Academic band vs placement rate", ORD["academic_band"])
    with b:
        metric = st.selectbox("Score to compare by outcome", ["avg_academic_pct", "ssc_percentage", "hsc_percentage", "degree_percentage",
                                                              "technical_score", "aptitude_score", "communication_score"])
        fig = px.box(f, x="status", y=metric, color="status", color_discrete_map=COLOR_MAP, category_orders={"status": ["Not Placed", "Placed"]},
                     title=f"{metric.replace('_', ' ')} by outcome")
        fig.update_layout(template="plotly_white", height=380, showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        s = f.sample(min(3000, len(f)), random_state=1)
        fig = px.scatter(s, x="skills_match_percentage", y="interview_score", color="status", color_discrete_map=COLOR_MAP,
                         opacity=0.5, title="Skills match vs interview score (sample)")
        fig.update_layout(template="plotly_white", height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Correlation between skills match and interview score: {f['skills_match_percentage'].corr(f['interview_score']):+.2f}")
    with b:
        t = f.assign(certifications_count=f["certifications_count"].astype(str))
        bar_rate(t, "certifications_count", "Certifications vs acceptance rate", sorted(t["certifications_count"].unique(), key=int))

# ------------------------------------------------------------------------------ Placement & acceptance
with tabs[2]:
    a, b = st.columns(2)
    with a:
        bar_rate(f, "company_tier", "Acceptance rate by company tier", ORD["company_tier"])
    with b:
        bar_rate(f, "experience_category", "Experience vs placement", ORD["experience_category"])
    a, b = st.columns(2)
    with a:
        bar_rate(f, "competition_level", "Competition level vs placement", ORD["competition_level"])
    with b:
        factor = st.selectbox("Other factor", ["bond_requirement", "job_role_match", "relevant_experience", "relocation_willingness",
                                               "career_switch_willingness", "internship_experience", "layoff_history", "skills_match_level"])
        bar_rate(f, factor, f"{factor.replace('_', ' ').title()} vs placement", ORD.get(factor), key="factor")
    pv = f.pivot_table(index="company_tier", columns="competition_level", values="accepted", aggfunc="mean").reindex(
        index=[t for t in ORD["company_tier"] if t in f["company_tier"].unique()],
        columns=[c for c in ORD["competition_level"] if c in f["competition_level"].unique()]) * 100
    fig = px.imshow(pv, text_auto=".1f", color_continuous_scale=["#EAF6FF", "#9DD5F5", "#2589C7", "#153B5B"], title="Placement rate (%) - company tier x competition level", aspect="auto")
    fig.update_layout(template="plotly_white", height=360)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------ Interview & evaluation
with tabs[3]:
    score = st.selectbox("Score", ["interview_score", "technical_score", "aptitude_score", "communication_score", "skills_match_percentage"])
    a, b = st.columns(2)
    with a:
        decile_line(f, score, f"{score.replace('_', ' ').title()} vs placement probability", key="score_dec")
    with b:
        fig = px.histogram(f, x=score, color="status", color_discrete_map=COLOR_MAP, nbins=40, barmode="overlay", opacity=0.6,
                           title=f"Distribution of {score.replace('_', ' ')}")
        fig.update_layout(template="plotly_white", height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        bar_rate(f, "interview_performance", "Interview performance category vs placement", ORD["interview_performance"])
    with b:
        bar_rate(f, "skills_match_level", "Skills match level vs placement", ORD["skills_match_level"], key="sml")

# ------------------------------------------------------------------------------ Model insights
with tabs[4]:
    m = rep["metrics"]
    if not m:
        st.info("Run `python run_pipeline.py` to train the model and populate this tab.")
    else:
        fm = m["final_test_metrics"]
        st.subheader(f"{m['best_baseline']}{' (tuned)' if m['tuned'] else ''} - held-out test set ({m['test_rows']:,} candidates)")
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("ROC-AUC", "roc_auc")]):
            col.metric(label, f"{fm[key]:.3f}")
        a, b = st.columns(2)
        with a:
            comp_df = pd.DataFrame(m["baseline_comparison"])[["model", "cv_roc_auc", "test_accuracy", "test_f1", "test_roc_auc"]]
            st.markdown("**Model comparison**")
            st.dataframe(comp_df.round(4), hide_index=True, width="stretch")
        with b:
            cmx = fm["confusion_matrix"]
            arr = [[cmx["tn"], cmx["fp"]], [cmx["fn"], cmx["tp"]]]
            fig = px.imshow(arr, text_auto=True, x=["Pred rejected", "Pred accepted"], y=["Actual rejected", "Actual accepted"],
                            color_continuous_scale="Blues", title="Confusion matrix")
            fig.update_layout(template="plotly_white", height=330, margin=dict(t=50, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        imp = rep["importance"]
        if imp is not None:
            top = imp.head(15).iloc[::-1]
            fig = px.bar(top, x="importance_mean", y="feature", orientation="h", error_x="importance_std",
                         title="Feature importance (drop in test ROC-AUC when shuffled)", color_discrete_sequence=["#2589C7"])
            fig.update_layout(template="plotly_white", height=520)
            st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        with a:
            if rep["direction"] is not None:
                st.markdown("**Direction of effect (logistic regression, standardised)**")
                d = rep["direction"].head(12).copy()
                d["effect"] = np.where(d["coefficient"] > 0, "raises odds", "lowers odds")
                st.dataframe(d.round(3), hide_index=True, width="stretch")
        with b:
            if rep["fairness"] is not None:
                st.markdown("**Fairness check by gender (test set)** - gender is not a model input")
                st.dataframe(rep["fairness"].round(2), hide_index=True, width="stretch")
        st.caption("Risk bands: " + ", ".join(f"{k} {v:.1f}%" for k, v in m["risk_bands_%"].items()))
        if "placement_probability_score" in f:
            fig = px.histogram(f, x="placement_probability_score", color="status", color_discrete_map=COLOR_MAP, nbins=40,
                               barmode="overlay", opacity=0.6, title="Predicted placement probability (out-of-fold) by actual outcome")
            fig.add_vline(x=risk_thr, line_dash="dot", annotation_text="high-risk cut-off")
            fig.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------ Predict
with tabs[5]:
    if model_bundle is None:
        st.info("Train the model first: `python run_pipeline.py`.")
    else:
        st.subheader("Predict placement / acceptance probability for one candidate")
        st.caption("Defaults are dataset medians / most common values. Age and gender are not used by the model.")
        num_info = df[C.NUMERIC_COLUMNS].describe().T
        with st.form("predict"):
            cols = st.columns(3)
            vals: dict = {}
            num_fields = [c for c in C.NUMERIC_COLUMNS if c not in C.MODEL_EXCLUDED]
            for i, col in enumerate(num_fields):
                lo, hi, med = float(num_info.loc[col, "min"]), float(num_info.loc[col, "max"]), float(num_info.loc[col, "50%"])
                integer = col in C.INTEGER_COLUMNS
                step = 1 if integer else 0.5
                v = cols[i % 3].number_input(col.replace("_", " ").title(), min_value=(int(lo) if integer else lo), max_value=(int(hi) if integer else hi),
                                              value=(int(round(med)) if integer else round(med, 2)), step=step)
                vals[col] = v
            cat_fields = [c for c in C.CATEGORICAL_COLUMNS if c not in C.MODEL_EXCLUDED]
            for i, col in enumerate(cat_fields):
                opts = ORD.get(col) or sorted(df[col].dropna().unique())
                default = df[col].mode().iloc[0]
                vals[col] = cols[i % 3].selectbox(col.replace("_", " ").title(), opts, index=list(opts).index(default) if default in opts else 0)
            go_btn = st.form_submit_button("Predict", type="primary")
        if go_btn:
            row = pd.DataFrame([{**vals, "age_years": int(df["age_years"].median()), "gender": df["gender"].mode().iloc[0]}])
            row = add_features(row)
            proba = float(model_bundle["pipeline"].predict_proba(row[model_bundle["features"]])[0, 1])
            band = assign_risk_band([proba])[0]
            a, b = st.columns([1, 1])
            with a:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=proba * 100, number={"suffix": "%"},
                                             title={"text": "Probability of placement / acceptance"},
                                             gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2589C7"},
                                                    "steps": [{"range": [0, C.HIGH_RISK_PROB * 100], "color": "#F4B6B6"},
                                                              {"range": [C.HIGH_RISK_PROB * 100, C.MEDIUM_RISK_PROB * 100], "color": "#FCE8B2"},
                                                              {"range": [C.MEDIUM_RISK_PROB * 100, 100], "color": "#BFE3C0"}]}))
                fig.update_layout(height=320, margin=dict(t=60, b=10))
                st.plotly_chart(fig, width="stretch")
            with b:
                st.metric("Risk band", band)
                st.metric("Predicted outcome", "Likely to accept / be placed" if proba >= model_bundle["threshold"] else "Likely NOT to accept / be placed")
                st.caption("Decision-support only. The score reflects patterns in historical data, not a guarantee for an individual.")

# ------------------------------------------------------------------------------ Data
with tabs[6]:
    st.subheader("Filtered data")
    st.dataframe(f.head(1000), width="stretch")
    st.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_candidates.csv", mime="text/csv")
    q = rep["quality"]
    if q:
        st.subheader("Data-quality summary (from preprocessing)")
        a, b, c = st.columns(3)
        a.metric("Raw rows", f"{q['rows_raw']:,}")
        b.metric("Duplicates removed", f"{q['duplicates_removed_after_standardisation']:,}")
        c.metric("Cells imputed", f"{sum(q['missing_before_imputation'].values()):,}")
        st.json({"missing_before_imputation": q["missing_before_imputation"], "consistency_flags": q["consistency_flags"]}, expanded=False)
