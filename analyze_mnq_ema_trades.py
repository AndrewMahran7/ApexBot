#!/usr/bin/env python3
"""
MNQ EMA-Only Trade Analysis
============================
Analyzes trade-level data from the no-ML EMA breakout runs.
"""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

YEARS = [2017, 2018, 2019, 2024, 2025]
DATA_DIR = Path("results/mnq_ema_noml")


def load_all_trades() -> pd.DataFrame:
    frames = []
    for y in YEARS:
        df = pd.read_csv(DATA_DIR / f"{y}_trades.csv")
        df["year"] = y
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["entry_time"] = pd.to_datetime(all_df["entry_time"], utc=True)
    all_df["exit_time"] = pd.to_datetime(all_df["exit_time"], utc=True)
    all_df["entry_hour"] = all_df["entry_time"].dt.tz_convert("America/New_York").dt.hour
    all_df["duration_min"] = (all_df["exit_time"] - all_df["entry_time"]).dt.total_seconds() / 60
    all_df["duration_bars"] = all_df["duration_min"] / 5  # 5-min bars
    all_df["win"] = all_df["net_pnl"] > 0
    return all_df


def pnl_distribution(df: pd.DataFrame):
    pnl = df["net_pnl"]
    print("=" * 70)
    print("  1. PNL DISTRIBUTION (per trade)")
    print("=" * 70)
    print(f"  Count:       {len(pnl)}")
    print(f"  Mean:        ${pnl.mean():.2f}")
    print(f"  Median:      ${pnl.median():.2f}")
    print(f"  Std:         ${pnl.std():.2f}")
    print(f"  Min:         ${pnl.min():.2f}")
    print(f"  Max:         ${pnl.max():.2f}")
    print(f"  P5:          ${pnl.quantile(0.05):.2f}")
    print(f"  P25:         ${pnl.quantile(0.25):.2f}")
    print(f"  P75:         ${pnl.quantile(0.75):.2f}")
    print(f"  P95:         ${pnl.quantile(0.95):.2f}")
    print(f"  Skew:        {pnl.skew():.3f}")

    # Histogram buckets
    bins = [-300, -100, -50, -25, -10, 0, 10, 25, 50, 100, 300, 1000]
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"${bins[i]} to ${bins[i+1]}")
    cuts = pd.cut(pnl, bins=bins, labels=labels, right=True)
    dist = cuts.value_counts().sort_index()
    print("\n  Histogram:")
    for bucket, count in dist.items():
        pct = count / len(pnl) * 100
        bar = "#" * int(pct)
        print(f"    {bucket:>16s}: {count:>4d} ({pct:>5.1f}%) {bar}")

    # By year
    print("\n  Mean/Median by year:")
    for y in YEARS:
        yp = df[df["year"] == y]["net_pnl"]
        print(f"    {y}: mean=${yp.mean():.2f}, median=${yp.median():.2f}, "
              f"std=${yp.std():.2f}")


def streaks(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  2. STREAKS")
    print("=" * 70)

    def compute_streaks(wins: list[bool]) -> tuple[int, int]:
        max_w = max_l = cur_w = cur_l = 0
        for w in wins:
            if w:
                cur_w += 1
                cur_l = 0
            else:
                cur_l += 1
                cur_w = 0
            max_w = max(max_w, cur_w)
            max_l = max(max_l, cur_l)
        return max_w, max_l

    # Overall
    wins = df["win"].tolist()
    mw, ml = compute_streaks(wins)
    print(f"  Overall: max winning streak = {mw}, max losing streak = {ml}")

    # By year
    for y in YEARS:
        w = df[df["year"] == y]["win"].tolist()
        mw, ml = compute_streaks(w)
        print(f"    {y}: max win streak = {mw}, max loss streak = {ml}")

    # Streak length distribution
    print("\n  Losing streak length distribution (all years):")
    streaks_list = []
    cur = 0
    for w in wins:
        if not w:
            cur += 1
        else:
            if cur > 0:
                streaks_list.append(cur)
            cur = 0
    if cur > 0:
        streaks_list.append(cur)
    if streaks_list:
        ctr = Counter(streaks_list)
        for length in sorted(ctr.keys()):
            print(f"    {length} losses: {ctr[length]} occurrences")


def duration_analysis(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  3. TRADE DURATION")
    print("=" * 70)
    dur = df["duration_bars"]
    print(f"  Mean bars held:   {dur.mean():.1f} ({dur.mean()*5:.0f} min)")
    print(f"  Median bars held: {dur.median():.1f} ({dur.median()*5:.0f} min)")
    print(f"  Min:              {dur.min():.0f} bars")
    print(f"  Max:              {dur.max():.0f} bars")

    # By exit reason
    print("\n  By exit reason:")
    for reason, grp in df.groupby("exit_reason"):
        d = grp["duration_bars"]
        wr = grp["win"].mean() * 100
        avg_pnl = grp["net_pnl"].mean()
        print(f"    {reason:>13s}: {len(grp):>4d} trades, "
              f"avg {d.mean():.1f} bars, WR {wr:.1f}%, avg PnL ${avg_pnl:.2f}")

    # Duration buckets
    print("\n  Duration distribution:")
    bins = [0, 2, 6, 12, 24, 48, 80, 200]
    labels = ["1-2 bars", "3-6", "7-12", "13-24", "25-48", "49-80", "80+"]
    cuts = pd.cut(dur, bins=bins, labels=labels, right=True)
    for bucket in labels:
        subset = df[cuts == bucket]
        if len(subset) == 0:
            continue
        wr = subset["win"].mean() * 100
        avg = subset["net_pnl"].mean()
        print(f"    {bucket:>10s}: {len(subset):>4d} trades, "
              f"WR {wr:.1f}%, avg PnL ${avg:.2f}")


def direction_breakdown(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  4a. LONG vs SHORT BREAKDOWN")
    print("=" * 70)
    for direction in ["long", "short"]:
        sub = df[df["direction"] == direction]
        n = len(sub)
        wins = sub["win"].sum()
        pnl = sub["net_pnl"].sum()
        avg = sub["net_pnl"].mean()
        med = sub["net_pnl"].median()
        gp = sub[sub["net_pnl"] > 0]["net_pnl"].sum()
        gl = abs(sub[sub["net_pnl"] <= 0]["net_pnl"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        avg_win = sub[sub["net_pnl"] > 0]["net_pnl"].mean() if wins > 0 else 0
        avg_loss = sub[sub["net_pnl"] <= 0]["net_pnl"].mean() if n - wins > 0 else 0
        print(f"\n  {direction.upper()}")
        print(f"    Trades:    {n}")
        print(f"    Win rate:  {wins/n*100:.1f}%")
        print(f"    Total PnL: ${pnl:.2f}")
        print(f"    Avg PnL:   ${avg:.2f}")
        print(f"    Median:    ${med:.2f}")
        print(f"    PF:        {pf:.2f}")
        print(f"    Avg win:   ${avg_win:.2f}")
        print(f"    Avg loss:  ${avg_loss:.2f}")

        # By year
        for y in YEARS:
            ys = sub[sub["year"] == y]
            yn = len(ys)
            if yn == 0:
                continue
            ywr = ys["win"].mean() * 100
            ypnl = ys["net_pnl"].sum()
            print(f"      {y}: {yn} trades, {ywr:.1f}% WR, ${ypnl:.2f}")


def time_of_day_breakdown(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  4b. TIME-OF-DAY BREAKDOWN (entry hour, ET)")
    print("=" * 70)
    print(f"  {'Hour':<6} {'Trades':>7} {'WR%':>7} {'Avg PnL':>9} {'Tot PnL':>10} "
          f"{'PF':>6} {'Avg Dur':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*9} {'-'*10} {'-'*6} {'-'*8}")

    for hour in sorted(df["entry_hour"].unique()):
        sub = df[df["entry_hour"] == hour]
        n = len(sub)
        wr = sub["win"].mean() * 100
        avg = sub["net_pnl"].mean()
        tot = sub["net_pnl"].sum()
        gp = sub[sub["net_pnl"] > 0]["net_pnl"].sum()
        gl = abs(sub[sub["net_pnl"] <= 0]["net_pnl"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        avg_dur = sub["duration_bars"].mean()
        print(f"  {hour:>4d}h  {n:>7} {wr:>6.1f}% ${avg:>8.2f} ${tot:>9.2f} "
              f"{pf:>6.2f} {avg_dur:>7.1f}b")

    # Which hours are profitable?
    print("\n  Profitable hours (positive total PnL):")
    for hour in sorted(df["entry_hour"].unique()):
        sub = df[df["entry_hour"] == hour]
        tot = sub["net_pnl"].sum()
        if tot > 0:
            wr = sub["win"].mean() * 100
            print(f"    {hour:>2d}:00 ET — {len(sub)} trades, {wr:.1f}% WR, ${tot:.2f}")


def main():
    df = load_all_trades()
    print(f"Loaded {len(df)} trades across {len(YEARS)} years\n")

    pnl_distribution(df)
    streaks(df)
    duration_analysis(df)
    direction_breakdown(df)
    time_of_day_breakdown(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
