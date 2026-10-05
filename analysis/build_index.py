"""
Ntawusigara: district priority index for poverty and financial inclusion.

Builds a transparent, adjustable priority index for Rwanda's 30 districts from
three public NISR sources, groups districts into typologies with k-means, and
tests how stable the ranking is when the index weights change.

Run:  python analysis/build_index.py
Out:  data/processed/district_index.csv, data/processed/summary.json, app/data.js
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, APP = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "app"
SEED = 2026

# Indicator definitions: column, label, direction (+1: higher value = higher priority)
INDICATORS = [
    ("poverty_2024", "Poverty rate", +1),
    ("poor_people", "Number of poor people", +1),
    ("not_formally_served", "Not formally served", +1),
    ("unbanked", "Unbanked", +1),
    ("progress_lag", "Slow poverty reduction", +1),
]
DEFAULT_WEIGHTS = {"poverty_2024": 0.35, "poor_people": 0.20, "not_formally_served": 0.20,
                   "unbanked": 0.10, "progress_lag": 0.15}


def load() -> pd.DataFrame:
    pov = pd.read_csv(RAW / "eicv7_district_poverty.csv")
    fin = pd.read_csv(RAW / "finscope2024_district_access.csv")
    pop = pd.read_csv(RAW / "census2022_district_population.csv")
    df = pop.merge(pov, on="district", validate="1:1").merge(fin, on="district", validate="1:1")
    assert len(df) == 30, "expected 30 districts"
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # People below the poverty line: 2024 rate applied to the 2022 census population.
    df["poor_people"] = (df.poverty_2024 / 100 * df.population_2022).round().astype(int)
    # Percentage-point fall in poverty since 2017 (modelled baseline).
    df["poverty_drop_pp"] = (df.poverty_2017_modelled - df.poverty_2024).round(1)
    # Slow progress = small drop. Expressed so that higher = more concerning.
    df["progress_lag"] = (df.poverty_drop_pp.max() - df.poverty_drop_pp).round(1)
    # Adults relying only on informal finance, or on nothing at all.
    df["not_formally_served"] = df.informal_only + df.excluded
    df["unbanked"] = 100 - df.banked
    return df


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    for col, _, sign in INDICATORS:
        x = df[col].astype(float)
        z = (x - x.min()) / (x.max() - x.min())
        df[f"n_{col}"] = (z if sign > 0 else 1 - z).round(4)
    return df


def score(df: pd.DataFrame, weights: dict) -> pd.Series:
    w = np.array([weights[c] for c, _, _ in INDICATORS], dtype=float)
    w = w / w.sum()
    m = df[[f"n_{c}" for c, _, _ in INDICATORS]].to_numpy()
    return pd.Series(m @ w * 100, index=df.index)


def robustness(df: pd.DataFrame, draws: int = 10000, top: int = 5) -> pd.Series:
    """Share of random weight sets (Dirichlet) under which a district ranks in the top N."""
    rng = np.random.default_rng(SEED)
    m = df[[f"n_{c}" for c, _, _ in INDICATORS]].to_numpy()
    W = rng.dirichlet(np.ones(len(INDICATORS)), size=draws)
    scores = m @ W.T                      # districts x draws
    ranks = (-scores).argsort(axis=0).argsort(axis=0)
    return pd.Series((ranks < top).mean(axis=1) * 100, index=df.index).round(1)


TYPOLOGY_RULES = [
    # (name, test on cluster centre in original units, short description)
    ("Stalled progress", lambda c: c.poverty_drop_pp < 7,
     "Poverty fell far less since 2017 than in the rest of the country"),
    ("Poor and under-served", lambda c: c.poverty_2024 >= 30 and c.not_formally_served >= 9,
     "High poverty and many adults outside formal finance"),
    ("Poor but formally reached", lambda c: c.poverty_2024 >= 30,
     "High poverty, but mobile money and SACCOs already reach most adults"),
    ("Banked urban core", lambda c: c.banked >= 40,
     "Low poverty, bank-led inclusion"),
    ("Lower poverty, finance gaps", lambda c: c.not_formally_served >= 9,
     "Below-average poverty, but pockets of informal-only or excluded adults"),
    ("Lower poverty, reached", lambda c: True,
     "Below-average poverty and broad formal reach"),
]


def typologies(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    feats = ["poverty_2024", "not_formally_served", "banked", "poverty_drop_pp"]
    X = StandardScaler().fit_transform(df[feats])
    km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
    df["cluster"] = km.labels_
    centres = df.groupby("cluster")[feats].mean()
    names, used = {}, set()
    for cl, c in centres.sort_values("poverty_2024", ascending=False).iterrows():
        for name, test, desc in TYPOLOGY_RULES:
            if test(c) and name not in used:
                names[cl] = (name, desc); used.add(name); break
        else:
            names[cl] = (f"Group {cl + 1}", "")
    df["typology"] = df.cluster.map(lambda c: names[c][0])
    df["typology_note"] = df.cluster.map(lambda c: names[c][1])
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = normalise(engineer(load()))
    df["priority_index"] = score(df, DEFAULT_WEIGHTS).round(1)
    df["priority_rank"] = df.priority_index.rank(ascending=False, method="min").astype(int)
    df["top5_robustness_pct"] = robustness(df)
    df = typologies(df)
    df = df.sort_values("priority_rank")

    rho, p = spearmanr(df.poverty_2024, df.not_formally_served)
    rho_b, p_b = spearmanr(df.poverty_2024, df.banked)
    summary = {
        "national": {"poverty_2024": 27.4, "poverty_2017_modelled": 39.8,
                     "formally_served_pct": 92, "financially_included_pct": 96},
        "total_population_2022": int(df.population_2022.sum()),
        "estimated_poor_people": int(df.poor_people.sum()),
        "default_weights": DEFAULT_WEIGHTS,
        "correlations": {
            "poverty_vs_not_formally_served": {"spearman_rho": round(rho, 2), "p": round(p, 3)},
            "poverty_vs_banked": {"spearman_rho": round(rho_b, 2), "p": round(p_b, 3)},
        },
        "top5_default": df.head(5).district.tolist(),
        "robust_top5": df[df.top5_robustness_pct >= 50].district.tolist(),
        "typology_counts": df.typology.value_counts().to_dict(),
    }

    keep = ["priority_rank", "district", "province", "population_2022", "urban_share_pct",
            "poverty_2024", "poverty_2017_modelled", "poverty_drop_pp", "poor_people",
            "banked", "other_formal_only", "informal_only", "excluded", "not_formally_served",
            "unbanked", "progress_lag", "priority_index", "top5_robustness_pct", "typology",
            "typology_note"] + [f"n_{c}" for c, _, _ in INDICATORS]
    # Optional published sampling errors, used by the rank-uncertainty simulation if present
    keep += [c for c in ("poverty_2024_se", "finscope_n") if c in df.columns]
    df[keep].to_csv(OUT / "district_index.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    shapes = json.loads((RAW / "district_shapes_svg.json").read_text())
    payload = {"districts": df[keep].to_dict(orient="records"), "summary": summary,
               "indicators": [{"key": c, "label": l} for c, l, _ in INDICATORS],
               "shapes": shapes}
    APP.mkdir(exist_ok=True)
    (APP / "data.js").write_text("window.NTW_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")

    print(df[["priority_rank", "district", "poverty_2024", "poor_people", "not_formally_served",
              "banked", "priority_index", "top5_robustness_pct", "typology"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
