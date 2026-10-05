"""
How certain is each district's rank? (Reproducible companion to app/ntw-tools.js.)

District poverty rates (EICV7) and finance shares (FinScope 2024) are survey
estimates from about 400-500 households or adults per district. This script
re-draws every district's estimates 2,000 times within their sampling error
(effective sample size N_EFF, allowing for cluster sampling), recomputes the
priority index with the default weights each time, and reports each
district's 90% rank interval and its chance of being in the top 5.

Run:  python analysis/rank_uncertainty.py [--n-eff 250]
Out:  data/processed/rank_uncertainty.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_index import DEFAULT_WEIGHTS, INDICATORS, engineer, load

ROOT = Path(__file__).resolve().parents[1]
DRAWS, SEED = 2000, 2026


def index_scores(df, w):
    cols = [c for c, _, _ in INDICATORS]
    m = df[cols].to_numpy(float)
    rng = m.max(0) - m.min(0)
    z = np.where(rng > 0, (m - m.min(0)) / np.where(rng > 0, rng, 1), 0)
    wv = np.array([w[c] for c in cols]); wv = wv / wv.sum()
    return z @ wv * 100


def main(n_eff):
    base = engineer(load())
    rng = np.random.default_rng(SEED)
    n = len(base)
    ranks = np.zeros((DRAWS, n), int)
    fin = base[["banked", "other_formal_only", "informal_only", "excluded"]].clip(lower=0).to_numpy(float) / 100
    for d in range(DRAWS):
        df = base[["district", "population_2022"]].copy()
        p24, p17 = base.poverty_2024 / 100, base.poverty_2017_modelled / 100
        df["poverty_2024"] = rng.beta(p24 * n_eff + .5, (1 - p24) * n_eff + .5) * 100
        df["poverty_2017_modelled"] = rng.beta(p17 * n_eff + .5, (1 - p17) * n_eff + .5) * 100
        f = np.array([rng.dirichlet(row * n_eff + .5) for row in fin]) * 100
        df["banked"], df["informal_only"], df["excluded"] = f[:, 0], f[:, 2], f[:, 3]
        df["poor_people"] = df.poverty_2024 / 100 * df.population_2022
        drop = df.poverty_2017_modelled - df.poverty_2024
        df["progress_lag"] = drop.max() - drop
        df["not_formally_served"] = df.informal_only + df.excluded
        df["unbanked"] = 100 - df.banked
        s = index_scores(df, DEFAULT_WEIGHTS)
        ranks[d] = (-s).argsort().argsort() + 1

    point = (-index_scores(base, DEFAULT_WEIGHTS)).argsort().argsort() + 1
    out = pd.DataFrame({
        "district": base.district, "rank": point,
        "rank_low_90": np.percentile(ranks, 5, axis=0).astype(int),
        "rank_high_90": np.percentile(ranks, 95, axis=0).astype(int),
        "rank_median": np.median(ranks, axis=0).astype(int),
        "top5_chance_pct": ((ranks <= 5).mean(0) * 100).round(1),
    }).sort_values("rank")
    path = ROOT / "data" / "processed" / "rank_uncertainty.csv"
    out.to_csv(path, index=False)
    print(f"Effective sample size per district: {n_eff}, {DRAWS} draws\n")
    print(out.head(12).to_string(index=False))
    print(f"\nWrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eff", type=int, default=250)
    main(ap.parse_args().n_eff)
