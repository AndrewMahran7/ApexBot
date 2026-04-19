#!/usr/bin/env python3
"""
Feature Predictive Power Analysis
===================================
Analyzes which features in the corrected (causal-entry) EMA candidates
dataset correlate with profitable trades.

Metrics per feature:
  - Point-biserial correlation with label_success (binary)
  - Pearson correlation with label_pnl_pts (continuous)
  - Mutual information with label_success
  - T-test (win vs loss group means)
"""

import sys
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "data/ema_candidates.csv"
OUT_DIR = Path("results/feature_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} candidates from {DATA_PATH}")
    print(f"Label distribution: {df['label_success'].value_counts().to_dict()}")
    print(f"Success rate: {df['label_success'].mean()*100:.1f}%\n")

    feat_cols = [c for c in df.columns if c.startswith("f_")]

    # Drop constant features
    constant = [c for c in feat_cols if df[c].nunique() <= 1]
    if constant:
        print(f"Dropping {len(constant)} constant features: {constant}")
        feat_cols = [c for c in feat_cols if c not in constant]

    # Drop near-duplicate features (correlation > 0.99)
    corr_matrix = df[feat_cols].corr().abs()
    dupes = set()
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            if corr_matrix.iloc[i, j] > 0.99:
                dupes.add((feat_cols[i], feat_cols[j]))
    if dupes:
        print(f"\nHighly correlated pairs (r > 0.99):")
        for a, b in sorted(dupes):
            print(f"  {a} <-> {b}")

    X = df[feat_cols].values
    y_bin = df["label_success"].values
    y_cont = df["label_pnl_pts"].values

    wins = df[df["label_success"] == 1]
    losses = df[df["label_success"] == 0]

    results = []

    # 1. Point-biserial correlation with binary label
    print("\nComputing correlations...")
    for i, col in enumerate(feat_cols):
        vals = df[col].values

        # Point-biserial with success
        pb_corr, pb_pval = sp_stats.pointbiserialr(y_bin, vals)

        # Pearson with continuous PnL
        pear_corr, pear_pval = sp_stats.pearsonr(vals, y_cont)

        # T-test: mean in wins vs losses
        win_vals = wins[col].values
        loss_vals = losses[col].values
        t_stat, t_pval = sp_stats.ttest_ind(win_vals, loss_vals, equal_var=False)

        # Win/loss means
        win_mean = win_vals.mean()
        loss_mean = loss_vals.mean()

        results.append({
            "feature": col,
            "pb_corr": pb_corr,
            "pb_pval": pb_pval,
            "pear_corr": pear_corr,
            "pear_pval": pear_pval,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "win_mean": win_mean,
            "loss_mean": loss_mean,
            "mean_diff": win_mean - loss_mean,
            "mean_diff_pct": (win_mean - loss_mean) / (abs(loss_mean) + 1e-10) * 100,
        })

    # 2. Mutual information with binary label
    print("Computing mutual information (this may take a moment)...")
    X_scaled = StandardScaler().fit_transform(X)
    mi_scores = mutual_info_classif(
        X_scaled, y_bin, discrete_features=False, random_state=42, n_neighbors=5
    )
    for i, col in enumerate(feat_cols):
        results[i]["mi_score"] = mi_scores[i]

    # Build results DataFrame
    res_df = pd.DataFrame(results)

    # Composite score: average of |pb_corr| rank, |pear_corr| rank, mi rank
    res_df["abs_pb"] = res_df["pb_corr"].abs()
    res_df["abs_pear"] = res_df["pear_corr"].abs()
    res_df["rank_pb"] = res_df["abs_pb"].rank(ascending=False)
    res_df["rank_pear"] = res_df["abs_pear"].rank(ascending=False)
    res_df["rank_mi"] = res_df["mi_score"].rank(ascending=False)
    res_df["avg_rank"] = (res_df["rank_pb"] + res_df["rank_pear"] + res_df["rank_mi"]) / 3
    res_df = res_df.sort_values("avg_rank")

    # Print TOP 10
    print("\n" + "=" * 95)
    print("  TOP 10 MOST PREDICTIVE FEATURES")
    print("=" * 95)
    print(f"  {'Rank':<5} {'Feature':<35} {'PB Corr':>8} {'Pear Corr':>10} "
          f"{'MI':>7} {'T p-val':>9} {'Win Mean':>10} {'Loss Mean':>10}")
    print(f"  {'-'*5} {'-'*35} {'-'*8} {'-'*10} {'-'*7} {'-'*9} {'-'*10} {'-'*10}")

    for rank, (_, row) in enumerate(res_df.head(10).iterrows(), 1):
        sig = "***" if row["t_pval"] < 0.001 else "**" if row["t_pval"] < 0.01 else "*" if row["t_pval"] < 0.05 else ""
        print(f"  {rank:<5} {row['feature']:<35} {row['pb_corr']:>+8.4f} {row['pear_corr']:>+10.4f} "
              f"{row['mi_score']:>7.4f} {row['t_pval']:>8.1e}{sig:>1s} "
              f"{row['win_mean']:>10.4f} {row['loss_mean']:>10.4f}")

    # Print BOTTOM 10 (noise)
    print("\n" + "=" * 95)
    print("  BOTTOM 10 (NOISE / NO PREDICTIVE VALUE)")
    print("=" * 95)
    print(f"  {'Rank':<5} {'Feature':<35} {'PB Corr':>8} {'Pear Corr':>10} "
          f"{'MI':>7} {'T p-val':>9}")
    print(f"  {'-'*5} {'-'*35} {'-'*8} {'-'*10} {'-'*7} {'-'*9}")

    bottom = res_df.tail(10).iloc[::-1]
    for rank, (_, row) in enumerate(bottom.iterrows(), len(feat_cols) - 9):
        print(f"  {rank:<5} {row['feature']:<35} {row['pb_corr']:>+8.4f} {row['pear_corr']:>+10.4f} "
              f"{row['mi_score']:>7.4f} {row['t_pval']:>8.1e}")

    # Summary stats
    print("\n" + "=" * 95)
    print("  SUMMARY")
    print("=" * 95)
    sig_005 = (res_df["t_pval"] < 0.05).sum()
    sig_001 = (res_df["t_pval"] < 0.01).sum()
    sig_0001 = (res_df["t_pval"] < 0.001).sum()
    max_mi = res_df["mi_score"].max()
    max_pb = res_df["abs_pb"].max()
    print(f"  Total features analyzed: {len(feat_cols)}")
    print(f"  Significant at p<0.05:   {sig_005}")
    print(f"  Significant at p<0.01:   {sig_001}")
    print(f"  Significant at p<0.001:  {sig_0001}")
    print(f"  Max |point-biserial|:    {max_pb:.4f}")
    print(f"  Max mutual information:  {max_mi:.4f}")
    print()

    if max_pb < 0.10:
        print("  ASSESSMENT: WEAK — No feature has strong individual correlation (|r| < 0.10)")
        print("  The ML model has very little signal to work with.")
    elif max_pb < 0.20:
        print("  ASSESSMENT: MARGINAL — Some features have small but real correlations")
        print("  An ML model may extract a modest edge from feature combinations.")
    else:
        print("  ASSESSMENT: MEANINGFUL — At least one feature has notable correlation")

    # Save full results
    out_path = OUT_DIR / "feature_ranking.csv"
    save_cols = ["feature", "pb_corr", "pb_pval", "pear_corr", "pear_pval",
                 "mi_score", "t_stat", "t_pval", "win_mean", "loss_mean",
                 "mean_diff", "avg_rank"]
    res_df[save_cols].to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n  Full ranking saved to: {out_path}")

    # Save JSON summary
    summary = {
        "dataset": DATA_PATH,
        "n_samples": len(df),
        "n_features": len(feat_cols),
        "success_rate": round(df["label_success"].mean() * 100, 1),
        "constant_features_dropped": constant,
        "significant_p05": int(sig_005),
        "significant_p01": int(sig_001),
        "max_abs_pb_corr": round(float(max_pb), 4),
        "max_mi": round(float(max_mi), 4),
        "top_10": res_df.head(10)["feature"].tolist(),
        "bottom_10": res_df.tail(10)["feature"].tolist(),
    }
    json_path = OUT_DIR / "feature_analysis_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
