"""
Checks whether district poverty can be predicted from district finance and census
indicators. Reproduces the negative result reported in docs/METHODOLOGY.md:
with 30 districts, leave-one-out predictions are no better than the mean.

Run: python analysis/district_model_check.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "district_index.csv")
y = df.poverty_2024.to_numpy()
FEATS = ["banked", "other_formal_only", "informal_only", "excluded", "urban_share_pct"]
MODELS = {
    "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
    "random_forest": RandomForestRegressor(300, min_samples_leaf=3, random_state=2026),
    "gradient_boosting": GradientBoostingRegressor(max_depth=2, n_estimators=150, learning_rate=0.05, random_state=2026),
}
res = {"baseline_mean_mae": round(float(np.abs(y - y.mean()).mean()), 2)}
for name, m in MODELS.items():
    p = cross_val_predict(m, df[FEATS], y, cv=LeaveOneOut())
    res[name] = {"loo_r2": round(r2_score(y, p), 3), "loo_mae": round(mean_absolute_error(y, p), 2)}
(ROOT / "data" / "processed" / "district_model_check.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
