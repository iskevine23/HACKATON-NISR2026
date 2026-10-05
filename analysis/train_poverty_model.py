"""
Phase 2: household poverty model on EICV7 microdata.

Predicts whether a household is poor from characteristics that NISR itself uses
to model poverty (household head, size, housing, utilities, assets, financial
accounts, district). Trains an interpretable logistic regression and a gradient
boosting benchmark with survey weights, reports cross-validated performance,
compares district-level predictions with official rates, and exports the
logistic model for the app's what-if panel.

Usage
  python analysis/train_poverty_model.py --data path/to/eicv7_household.csv
  python analysis/train_poverty_model.py --demo     # synthetic data, tests the code only

IMPORTANT
  EICV7 variable names differ from the generic names below. Map them in COLUMNS
  after reading the EICV7 data dictionary. --demo output is synthetic, never
  report it as a result.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026

# Map generic names (left) to EICV7 column names (right) once you have the data dictionary.
COLUMNS = {
    "poor": "poor",                       # 1 if consumption per adult equivalent < RWF 560,127
    "weight": "weight",                   # household sampling weight
    "district": "district",
    "urban": "urban",                     # 1 urban, 0 rural
    "hh_size": "hh_size",
    "head_female": "head_female",
    "head_age": "head_age",
    "head_educ": "head_educ",             # none / primary / lower_sec / upper_sec / university
    "electricity": "electricity",         # grid or solar
    "modern_floor": "modern_floor",
    "improved_water": "improved_water",
    "phone": "phone",
    "savings_account": "savings_account",
    "momo_account": "momo_account",
    "livestock": "livestock",
}
NUMERIC = ["hh_size", "head_age"]
BINARY = ["urban", "head_female", "electricity", "modern_floor", "improved_water", "phone",
          "savings_account", "momo_account", "livestock"]
CATEGORICAL = ["head_educ", "district"]
WHATIF = ["savings_account", "momo_account", "electricity", "phone"]  # levers shown in the app


def synthetic(n=15000):
    """Synthetic households for testing the pipeline only. Not real data."""
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(ROOT / "data" / "raw" / "eicv7_district_poverty.csv")
    district = rng.choice(d.district, size=n)
    base = d.set_index("district").poverty_2024.reindex(district).to_numpy() / 100
    df = pd.DataFrame({
        "district": district, "weight": rng.uniform(50, 150, n),
        "urban": rng.binomial(1, 0.28, n), "hh_size": rng.integers(1, 10, n),
        "head_female": rng.binomial(1, 0.2, n), "head_age": rng.integers(18, 85, n),
        "head_educ": rng.choice(["none", "primary", "lower_sec", "upper_sec", "university"], n, p=[.59, .25, .04, .07, .05]),
        "electricity": rng.binomial(1, 0.72, n), "modern_floor": rng.binomial(1, 0.39, n),
        "improved_water": rng.binomial(1, 0.9, n), "phone": rng.binomial(1, 0.85, n),
        "savings_account": rng.binomial(1, 0.68, n), "momo_account": rng.binomial(1, 0.87, n),
        "livestock": rng.binomial(1, 0.5, n)})
    logit = (np.log(base / (1 - base)) + 0.12 * (df.hh_size - 4) - 0.6 * df.modern_floor
             - 0.4 * df.electricity - 0.3 * df.savings_account - 0.8 * (df.head_educ.isin(["upper_sec", "university"]))
             - 0.3 * df.urban)
    df["poor"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


def load(path):
    raw = pd.read_csv(path)
    missing = [v for v in COLUMNS.values() if v not in raw.columns]
    if missing:
        raise SystemExit(f"Map these columns in COLUMNS first: {missing}")
    return raw.rename(columns={v: k for k, v in COLUMNS.items()})[list(COLUMNS)]


def pre():
    return ColumnTransformer([("num", StandardScaler(), NUMERIC), ("bin", "passthrough", BINARY),
                              ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CATEGORICAL)])


def models():
    return {
        "logistic": Pipeline([("pre", pre()), ("m", LogisticRegression(max_iter=2000, C=1.0))]),
        "gradient_boosting": Pipeline([("pre", pre()), ("m", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300, random_state=SEED))]),
    }


def evaluate(df):
    X, y, w = df[NUMERIC + BINARY + CATEGORICAL], df.poor.to_numpy(), df.weight.to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out, oof = {}, {}
    for name, m in models().items():
        p = np.zeros(len(df))
        for tr, te in cv.split(X, y):
            m.fit(X.iloc[tr], y[tr], m__sample_weight=w[tr])
            p[te] = m.predict_proba(X.iloc[te])[:, 1]
        oof[name] = p
        out[name] = {"auc": round(roc_auc_score(y, p, sample_weight=w), 3),
                     "brier": round(brier_score_loss(y, p, sample_weight=w), 3),
                     "accuracy_at_0.5": round(float(np.average((p >= .5) == y, weights=w)), 3)}
    # District check: weighted mean predicted probability vs weighted observed rate
    dd = df.assign(pred=oof["logistic"]).groupby("district").apply(
        lambda g: pd.Series({"observed": np.average(g.poor, weights=g.weight) * 100,
                             "predicted": np.average(g.pred, weights=g.weight) * 100}), include_groups=False).round(1)
    out["district_mae_pp"] = round(float((dd.observed - dd.predicted).abs().mean()), 2)
    return out, dd


def export(df, path):
    """Fit the logistic model on all data and export what the app needs for what-if scenarios."""
    m = models()["logistic"]
    m.fit(df[NUMERIC + BINARY + CATEGORICAL], df.poor, m__sample_weight=df.weight)
    names = m.named_steps["pre"].get_feature_names_out().tolist()
    coef = dict(zip(names, m.named_steps["m"].coef_[0].round(4).tolist()))
    levers = {k: coef[f"bin__{k}"] for k in WHATIF}
    # District baseline: mean share of households with each lever, weighted
    base = df.groupby("district").apply(lambda g: pd.Series({k: np.average(g[k], weights=g.weight) for k in WHATIF}), include_groups=False).round(3)
    path.write_text(json.dumps({"intercept": round(float(m.named_steps["m"].intercept_[0]), 4),
                                "levers_log_odds": levers, "district_lever_rates": base.to_dict(orient="index"),
                                "note": "Associations from a cross-sectional model, not causal effects."}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if not a.demo and not a.data:
        ap.error("give --data PATH or --demo")
    df = synthetic() if a.demo else load(a.data)
    out_dir = ROOT / "data" / "processed" / ("demo_synthetic" if a.demo else "model")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, dd = evaluate(df)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    dd.to_csv(out_dir / "district_check.csv")
    export(df, out_dir / "model.json")
    print(("SYNTHETIC DEMO, NOT RESULTS\n" if a.demo else "") + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
