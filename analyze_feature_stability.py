#!/usr/bin/env python3
"""
Feature Stability Across Years — MNQ
======================================
Generates EMA candidate features per year for MNQ, then computes
per-year correlation with trade success to identify stable vs
overfit features.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from data.ema_candidates import build_ema_candidate_dataset, EMACandidateConfig

warnings.filterwarnings("ignore")

OUT_DIR = Path("results/feature_stability")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_DATA = {
    2017: ("data/mnq_2017.csv", None, None),
    2018: ("data/mnq_2018.csv", None, None),
    2019: ("data/mnq_2019.csv", None, None),
    2024: ("data/mnq_4y.csv", "2024-01-01", "2024-12-31"),
    2025: ("data/mnq_2025.csv", None, None),
}

CONSTANT_FEATURES = {
    "f_time_minutes_since_open",
    "f_time_minutes_since_range_close",
    "f_time_minutes_to_close",
}


def generate_year(year, data_path, start, end):
    """Generate EMA candidates for one year."""
    cfg = EMACandidateConfig(
        ema_length=50,
        reward_risk=1.5,
        allow_shorts=True,
    )
    df = build_ema_candidate_dataset(
        input_path=data_path,
        cfg=cfg,
        start=start,
        end=end,
        save_path=None,
    )
    return df


def compute_correlations(df, feat_cols):
    """Compute point-biserial correlation of each feature with label_success."""
    y = df["label_success"].values
    results = {}
    for col in feat_cols:
        vals = df[col].values
        if np.std(vals) < 1e-12:
            results[col] = (0.0, 1.0)
            continue
        corr, pval = sp_stats.pointbiserialr(y, vals)
        results[col] = (corr, pval)
    return results


def main():
    print("=" * 80)
    print("  FEATURE STABILITY ANALYSIS — MNQ EMA CANDIDATES")
    print("=" * 80)

    # Generate per-year datasets
    year_dfs = {}
    for year in sorted(YEAR_DATA.keys()):
        data_path, start, end = YEAR_DATA[year]
        print(f"  Generating {year} from {data_path} ...", end=" ", flush=True)
        df = generate_year(year, data_path, start, end)
        year_dfs[year] = df
        sr = df["label_success"].mean() * 100
        print(f"{len(df)} candidates, {sr:.1f}% success rate")

    # Identify feature columns (exclude constants)
    sample = next(iter(year_dfs.values()))
    feat_cols = sorted([
        c for c in sample.columns
        if c.startswith("f_") and c not in CONSTANT_FEATURES
    ])
    print(f"\n  {len(feat_cols)} features to analyze\n")

    # Compute per-year correlations
    year_corrs = {}  # year -> {feature: (corr, pval)}
    for year, df in sorted(year_dfs.items()):
        year_corrs[year] = compute_correlations(df, feat_cols)

    years = sorted(YEAR_DATA.keys())

    # Build stability table
    rows = []
    for feat in feat_cols:
        corrs = [year_corrs[y][feat][0] for y in years]
        pvals = [year_corrs[y][feat][1] for y in years]

        mean_corr = np.mean(corrs)
        std_corr = np.std(corrs)
        median_corr = np.median(corrs)

        # Sign consistency: fraction of years with same sign as median
        if median_corr != 0:
            sign = np.sign(median_corr)
            sign_consistent = sum(1 for c in corrs if np.sign(c) == sign) / len(corrs)
        else:
            sign_consistent = 0.0

        # How many years significant at p<0.05?
        n_sig = sum(1 for p in pvals if p < 0.05)

        # Stability score: high mean |corr| + low std + consistent sign
        stability = abs(mean_corr) * sign_consistent / (std_corr + 0.01)

        rows.append({
            "feature": feat,
            "mean_corr": mean_corr,
            "median_corr": median_corr,
            "std_corr": std_corr,
            "sign_consistency": sign_consistent,
            "n_years_sig": n_sig,
            "stability_score": stability,
            **{f"corr_{y}": year_corrs[y][feat][0] for y in years},
            **{f"pval_{y}": year_corrs[y][feat][1] for y in years},
        })

    res_df = pd.DataFrame(rows).sort_values("stability_score", ascending=False)

    # Print main table
    print("=" * 115)
    print("  FEATURE STABILITY TABLE (sorted by stability score)")
    print("=" * 115)
    header = (f"  {'Feature':<32} {'Mean':>6} {'Std':>6} {'Sign%':>5} "
              f"{'#Sig':>4} {'Stab':>6} |")
    for y in years:
        header += f" {y:>7}"
    print(header)
    print("  " + "-" * 112)

    for _, row in res_df.iterrows():
        line = (f"  {row['feature']:<32} {row['mean_corr']:>+6.3f} {row['std_corr']:>6.3f} "
                f"{row['sign_consistency']*100:>4.0f}% {row['n_years_sig']:>4} "
                f"{row['stability_score']:>6.2f} |")
        for y in years:
            c = row[f"corr_{y}"]
            p = row[f"pval_{y}"]
            star = "*" if p < 0.05 else " "
            line += f" {c:>+6.3f}{star}"
        print(line)

    # Categorize features
    stable_good = res_df[
        (res_df["sign_consistency"] >= 0.8) &
        (res_df["mean_corr"].abs() >= 0.04) &
        (res_df["std_corr"] < 0.08)
    ]
    overfit = res_df[
        (res_df["sign_consistency"] < 0.6) |
        ((res_df["n_years_sig"] <= 1) & (res_df["std_corr"] > 0.06))
    ]

    print(f"\n{'='*80}")
    print(f"  CONSISTENTLY PREDICTIVE (sign consistent >= 80%, |mean r| >= 0.04, std < 0.08)")
    print(f"{'='*80}")
    if len(stable_good) > 0:
        for _, row in stable_good.iterrows():
            corr_str = ", ".join(f"{row[f'corr_{y}']:+.3f}" for y in years)
            print(f"  {row['feature']:<32} mean={row['mean_corr']:+.4f} std={row['std_corr']:.4f}  [{corr_str}]")
    else:
        print("  (none)")

    print(f"\n{'='*80}")
    print(f"  OVERFIT / UNSTABLE (sign flips or high variance)")
    print(f"{'='*80}")
    if len(overfit) > 0:
        for _, row in overfit.head(15).iterrows():
            corr_str = ", ".join(f"{row[f'corr_{y}']:+.3f}" for y in years)
            print(f"  {row['feature']:<32} mean={row['mean_corr']:+.4f} std={row['std_corr']:.4f}  [{corr_str}]")

    # Summary assessment
    max_stability = res_df["stability_score"].max()
    n_consistent = len(stable_good)
    n_overfit = len(overfit)

    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Total features:              {len(feat_cols)}")
    print(f"  Consistently predictive:     {n_consistent}")
    print(f"  Overfit / unstable:          {n_overfit}")
    print(f"  Max stability score:         {max_stability:.2f}")
    print(f"  Features significant >=3 yr: {(res_df['n_years_sig'] >= 3).sum()}")
    print(f"  Features never significant:  {(res_df['n_years_sig'] == 0).sum()}")

    # Save
    csv_path = OUT_DIR / "feature_stability.csv"
    res_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Full table: {csv_path}")

    summary = {
        "years": years,
        "n_features": len(feat_cols),
        "n_consistent": n_consistent,
        "n_overfit": n_overfit,
        "consistent_features": stable_good["feature"].tolist() if len(stable_good) > 0 else [],
        "overfit_features": overfit["feature"].tolist() if len(overfit) > 0 else [],
        "year_sample_sizes": {y: len(year_dfs[y]) for y in years},
    }
    json_path = OUT_DIR / "feature_stability_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:    {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
