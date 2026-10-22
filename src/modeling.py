"""Step 7: machine-learning modelling.

* Target: status -> Placed (job accepted) = 1, Not Placed = 0
* Imputation + scaling + encoding live INSIDE the sklearn Pipeline, so they are
  fitted on training folds only (no leakage into validation/test data).
* Demographic columns (age_years, gender) are not used as inputs; fairness is audited instead.
* 3 baseline models are compared with stratified CV, the best one is tuned
  (RandomizedSearchCV) and evaluated once on a held-out test set.
* placement_probability_score for every candidate is an out-of-fold prediction, so it is
  never produced by a model that saw that candidate's label. It is NOT a model input.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src.features import MODEL_CATEGORICAL, MODEL_FEATURES, MODEL_NUMERIC, add_features, assign_risk_band
from src.utils import banner, save_json, savefig

THRESHOLD = 0.5


# --------------------------------------------------------------------------- pipeline
def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, MODEL_NUMERIC), ("cat", cat, MODEL_CATEGORICAL)])


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def candidate_models(fast: bool) -> dict:
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=120 if fast else 300, min_samples_leaf=3,
                                                n_jobs=-1, random_state=rs),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=rs),
    }


def search_space(name: str) -> dict:
    if name == "Logistic Regression":
        return {"model__C": loguniform(1e-2, 1e2)}
    if name == "Random Forest":
        return {"model__n_estimators": [150, 250, 400], "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_leaf": [1, 3, 5, 10], "model__max_features": ["sqrt", 0.5]}
    return {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_iter": [150, 250, 400],
            "model__max_leaf_nodes": [15, 31, 63], "model__min_samples_leaf": [20, 50, 100],
            "model__l2_regularization": [0.0, 0.1, 1.0]}


# --------------------------------------------------------------------------- metrics
def evaluate(y_true, proba) -> dict:
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def fairness_by_gender(gender: pd.Series, y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"gender": gender.fillna("Unknown").values, "y": y_true.values, "p": proba})
    d["pred"] = (d["p"] >= THRESHOLD).astype(int)
    rows = []
    for g, grp in d.groupby("gender"):
        pos = grp[grp["y"] == 1]
        rows.append({"gender": g, "n": len(grp), "actual_placement_rate_%": grp["y"].mean() * 100,
                     "predicted_positive_rate_%": grp["pred"].mean() * 100,
                     "true_positive_rate_%": (pos["pred"].mean() * 100) if len(pos) else np.nan,
                     "mean_predicted_probability": grp["p"].mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plots
def plot_model_comparison(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["cv_roc_auc"], 0.4, label="5-fold CV ROC-AUC (train)", color="#4C78A8")
    ax.bar(x + 0.2, table["test_roc_auc"], 0.4, label="Test ROC-AUC", color="#F58518")
    ax.set_xticks(x, table.index, rotation=15)
    lo = max(0.5, float(table[["cv_roc_auc", "test_roc_auc"]].min().min()) - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    for i, (a, b) in enumerate(zip(table["cv_roc_auc"], table["test_roc_auc"])):
        ax.text(i - 0.2, a + 0.003, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.003, f"{b:.3f}", ha="center", fontsize=8)
    savefig(fig, "model_comparison.png")


def plot_roc(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    savefig(fig, "roc_curves.png")


def plot_confusion(cm: dict, title: str) -> None:
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], ["Rejected (0)", "Accepted (1)"])
    ax.set_yticks([0, 1], ["Rejected (0)", "Accepted (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > arr.max() / 2 else "black")
    savefig(fig, "confusion_matrix.png")


def plot_importance(imp: pd.DataFrame, top: int = 15) -> None:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#54A24B")
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title("Permutation feature importance")
    savefig(fig, "feature_importance.png")


# --------------------------------------------------------------------------- main
def main(fast: bool = False, tune: bool = True) -> None:
    banner("STEP 4 - Machine learning")
    t0 = time.time()
    C.ensure_dirs()
    if not C.ANALYTICAL_CSV.exists():
        from src import features
        features.main()

    df = add_features(pd.read_csv(C.STANDARDIZED_CSV))
    X, y = df[MODEL_FEATURES], df[C.TARGET_BIN]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    print(f"Rows: {len(df):,}  train={len(X_tr):,}  test={len(X_te):,}  accepted={y.mean():.1%}")
    print(f"Features used ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")

    cv = StratifiedKFold(n_splits=3 if fast else 5, shuffle=True, random_state=C.RANDOM_STATE)

    # ---- 1. baseline comparison -------------------------------------------------------
    rows, fitted, curves = [], {}, {}
    for name, est in candidate_models(fast).items():
        pipe = make_pipeline(est)
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["roc_auc", "f1", "accuracy"])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        m = evaluate(y_te, proba)
        fitted[name] = pipe
        fpr, tpr, _ = roc_curve(y_te, proba)
        curves[name] = (fpr, tpr, m["roc_auc"])
        rows.append({"model": name, "cv_roc_auc": scores["test_roc_auc"].mean(), "cv_roc_auc_std": scores["test_roc_auc"].std(),
                     "cv_f1": scores["test_f1"].mean(), "cv_accuracy": scores["test_accuracy"].mean(),
                     **{f"test_{k}": v for k, v in m.items() if k != "confusion_matrix"}})
        print(f"  {name:<24} CV AUC={rows[-1]['cv_roc_auc']:.4f}  test AUC={m['roc_auc']:.4f}  "
              f"acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
    table = pd.DataFrame(rows).set_index("model")
    best_name = table["cv_roc_auc"].idxmax()
    print(f"Best baseline by CV ROC-AUC: {best_name}")

    # ---- 2. tune the best model -------------------------------------------------------
    final = fitted[best_name]
    tuned_params = None
    if tune:
        search = RandomizedSearchCV(make_pipeline(clone(candidate_models(fast)[best_name])), search_space(best_name),
                                    n_iter=4 if fast else 12, cv=StratifiedKFold(3, shuffle=True, random_state=C.RANDOM_STATE),
                                    scoring="roc_auc", random_state=C.RANDOM_STATE, n_jobs=1, refit=True)
        search.fit(X_tr, y_tr)
        final, tuned_params = search.best_estimator_, {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print(f"Tuned {best_name}: best CV AUC={search.best_score_:.4f}  params={tuned_params}")

    proba_te = final.predict_proba(X_te)[:, 1]
    final_metrics = evaluate(y_te, proba_te)
    print(f"FINAL test: acc={final_metrics['accuracy']:.4f} prec={final_metrics['precision']:.4f} "
          f"recall={final_metrics['recall']:.4f} F1={final_metrics['f1']:.4f} AUC={final_metrics['roc_auc']:.4f}")

    # ---- 3. interpretability ----------------------------------------------------------
    n_imp = min(5000, len(X_te))
    idx = X_te.sample(n_imp, random_state=C.RANDOM_STATE).index
    pi = permutation_importance(final, X_te.loc[idx], y_te.loc[idx], scoring="roc_auc",
                                n_repeats=5, random_state=C.RANDOM_STATE, n_jobs=1)
    imp = (pd.DataFrame({"feature": MODEL_FEATURES, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
           .sort_values("importance_mean", ascending=False).reset_index(drop=True))
    imp.to_csv(C.FEATURE_IMPORTANCE_CSV, index=False)

    # Direction of effect from the (scaled) logistic regression baseline
    lr = fitted["Logistic Regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coef = lr.named_steps["model"].coef_[0]
    direction = (pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coefficient": coef,
                               "odds_ratio": np.exp(coef)})
                 .assign(abs_coef=lambda d: d["coefficient"].abs()).sort_values("abs_coef", ascending=False)
                 .drop(columns="abs_coef").reset_index(drop=True))
    direction.to_csv(C.FEATURE_DIRECTION_CSV, index=False)

    # ---- 4. fairness audit ------------------------------------------------------------
    fair = fairness_by_gender(df.loc[X_te.index, "gender"], y_te, proba_te)
    fair.to_csv(C.FAIRNESS_CSV, index=False)

    # ---- 5. out-of-fold probability for every candidate -------------------------------
    oof_cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    oof = cross_val_predict(clone(final), X, y, cv=oof_cv, method="predict_proba", n_jobs=1)[:, 1]
    scores = pd.Series(oof, index=df[C.ID_COL].values)
    analytical = pd.read_csv(C.ANALYTICAL_CSV)
    analytical["placement_probability_score"] = analytical[C.ID_COL].map(scores).round(4)
    analytical["risk_band"] = assign_risk_band(analytical["placement_probability_score"])
    analytical.to_csv(C.ANALYTICAL_CSV, index=False)

    # ---- 6. save artefacts ------------------------------------------------------------
    joblib.dump({"pipeline": final, "features": MODEL_FEATURES, "model_name": best_name, "threshold": THRESHOLD}, C.MODEL_PATH)
    plot_model_comparison(table)
    plot_roc(curves)
    plot_confusion(final_metrics["confusion_matrix"], f"{best_name} (tuned)" if tune else best_name)
    plot_importance(imp)

    save_json({
        "rows": len(df), "train_rows": len(X_tr), "test_rows": len(X_te),
        "accepted_rate": float(y.mean()), "features_used": MODEL_FEATURES,
        "excluded_from_model": C.MODEL_EXCLUDED, "decision_threshold": THRESHOLD,
        "cv_folds": cv.get_n_splits(), "best_baseline": best_name, "tuned": bool(tune), "tuned_params": tuned_params,
        "baseline_comparison": table.reset_index().to_dict(orient="records"),
        "final_test_metrics": final_metrics,
        "oof_auc_all_candidates": float(roc_auc_score(y, oof)),
        "risk_bands_%": analytical["risk_band"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "top_features": imp.head(10)[["feature", "importance_mean"]].to_dict(orient="records"),
        "fairness_gender": fair.to_dict(orient="records"),
    }, C.METRICS_JSON)
    print(f"\nTop features:\n{imp.head(8).to_string(index=False)}")
    print(f"\nSaved model -> {C.MODEL_PATH.relative_to(C.ROOT)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true", help="fewer folds / trees / search iterations")
    ap.add_argument("--no-tune", action="store_true", help="skip hyper-parameter search")
    a = ap.parse_args()
    main(fast=a.fast, tune=not a.no_tune)
