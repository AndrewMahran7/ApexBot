"""
Analyze trade performance by time of day.

EMA candidates all enter at 09:50 ET, so we reconstruct EXIT timestamps
by replaying the same forward-bar-scan as the labeler, then group exits
into three sessions:

  Morning  : 09:50 – 11:30 ET  (fast resolution: TP/SL hit quickly)
  Midday   : 11:30 – 14:00 ET
  Close    : 14:00 – 16:00 ET  (includes forced EOD exits)

Also analyses the multi-strategy trades in trades_filtered.csv by
ENTRY hour, since those have varying entry times throughout the day.
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
import logging, json, sys
import numpy as np
import pandas as pd
from pathlib import Path

from data.loader import load_bars
from data.features import compute_features, FeatureConfig
from data.ema_candidates import (
    EMACandidateConfig,
    _identify_ema_candidates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)
log = logging.getLogger(__name__)

MES_PT = 5.0
COMMISSION_RT = 4.64        # round-trip
SLIPPAGE = 1.25             # 1 tick * $5/pt * 0.25

# ── Session boundaries (ET) ──────────────────────────────────────────
from datetime import time as dtime
MORNING_END = dtime(11, 30)
MIDDAY_END  = dtime(14, 0)

def session_label(t) -> str:
    if t < MORNING_END:
        return "Morning (9:50-11:30)"
    elif t < MIDDAY_END:
        return "Midday (11:30-14:00)"
    else:
        return "Close (14:00-16:00)"


# ═════════════════════════════════════════════════════════════════════
# Part 1:  EMA candidates — reconstruct exit times
# ═════════════════════════════════════════════════════════════════════
def reconstruct_exit_times(csv_path: str = "data/mes_4y.csv"):
    log.info("Loading bars and generating candidates ...")
    bars = load_bars(csv_path)
    cfg = EMACandidateConfig()
    feat_cfg = FeatureConfig(
        session_open=cfg.session_open,
        range_start=cfg.range_start,
        range_end=cfg.range_end,
        ema_length=cfg.ema_length,
    )
    featured = compute_features(bars, feat_cfg)
    candidates = _identify_ema_candidates(featured, cfg)
    log.info("  %d candidates found", len(candidates))

    eod_t = pd.Timestamp("1970-01-01 15:50").time()

    exit_times = []
    exit_hours = []
    pnl_list = []
    success_list = []
    exit_reasons = []
    risk_pts_list = []

    for ts, cand in candidates.iterrows():
        direction = cand["direction"]
        entry_price = cand["entry_price"]
        stop_loss = cand["stop_loss"]
        take_profit = cand["take_profit"]
        session_date = cand["session_date"]
        entry_time = cand["entry_time"]

        session_mask = (
            (featured["session_date"] == session_date)
            & (featured.index > entry_time)
            & (featured.index.time <= eod_t)
        )
        future_bars = featured.loc[session_mask]

        if future_bars.empty:
            exit_times.append(pd.NaT)
            exit_hours.append(np.nan)
            pnl_list.append(np.nan)
            success_list.append(np.nan)
            exit_reasons.append("NO_DATA")
            risk_pts_list.append(np.nan)
            continue

        exit_ts = future_bars.index[-1]   # default: last bar (EOD)
        exit_price = float(future_bars.iloc[-1]["close"])
        exit_reason = "EOD"

        for bar_ts, fb in future_bars.iterrows():
            high = float(fb["high"])
            low = float(fb["low"])
            if direction == "long":
                if low <= stop_loss:
                    exit_ts, exit_price, exit_reason = bar_ts, stop_loss, "SL"
                    break
                if high >= take_profit:
                    exit_ts, exit_price, exit_reason = bar_ts, take_profit, "TP"
                    break
            else:
                if high >= stop_loss:
                    exit_ts, exit_price, exit_reason = bar_ts, stop_loss, "SL"
                    break
                if low <= take_profit:
                    exit_ts, exit_price, exit_reason = bar_ts, take_profit, "TP"
                    break

        pnl_pts = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        pnl_dollars = pnl_pts * MES_PT - COMMISSION_RT - SLIPPAGE
        success = 1 if pnl_pts > 0 else 0

        risk_pts = abs(entry_price - stop_loss)

        exit_times.append(exit_ts)
        exit_hours.append(exit_ts.time())
        pnl_list.append(pnl_dollars)
        success_list.append(success)
        exit_reasons.append(exit_reason)
        risk_pts_list.append(risk_pts)

    candidates["exit_time"] = exit_times
    candidates["exit_hour"] = exit_hours
    candidates["pnl_dollars"] = pnl_list
    candidates["label_success"] = success_list
    candidates["exit_reason"] = exit_reasons
    candidates["risk_pts"] = risk_pts_list
    candidates = candidates.dropna(subset=["pnl_dollars"])

    # Compute minutes from entry (9:50) to exit
    entry_base = pd.Timestamp("1970-01-01 09:50")
    candidates["minutes_to_exit"] = candidates["exit_hour"].apply(
        lambda t: (pd.Timestamp(f"1970-01-01 {t}") - entry_base).total_seconds() / 60
    )

    # Assign session
    candidates["session"] = candidates["exit_hour"].apply(session_label)

    return candidates


def print_ema_analysis(df):
    log.info("")
    log.info("=" * 78)
    log.info("  EMA BREAKOUT TRADES — PERFORMANCE BY EXIT TIME SESSION")
    log.info("=" * 78)
    log.info("  All trades enter at 09:50 ET. Grouped by WHEN they exit.\n")

    sessions = ["Morning (9:50-11:30)", "Midday (11:30-14:00)", "Close (14:00-16:00)"]
    results = {}

    header = f"  {'Session':<25} {'N':>5} {'WR':>7} {'Avg PnL':>10} {'Total PnL':>12} {'Avg Mins':>9} {'TP%':>6} {'SL%':>6} {'EOD%':>6}"
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))

    for sess in sessions:
        g = df[df["session"] == sess]
        if g.empty:
            continue
        n = len(g)
        wr = g["label_success"].mean()
        avg_pnl = g["pnl_dollars"].mean()
        total_pnl = g["pnl_dollars"].sum()
        avg_min = g["minutes_to_exit"].mean()
        tp_pct = (g["exit_reason"] == "TP").mean() * 100
        sl_pct = (g["exit_reason"] == "SL").mean() * 100
        eod_pct = (g["exit_reason"] == "EOD").mean() * 100

        log.info(
            f"  {sess:<25} {n:>5} {wr:>6.1%} ${avg_pnl:>9.2f} ${total_pnl:>11.2f} {avg_min:>8.0f}m {tp_pct:>5.1f} {sl_pct:>5.1f} {eod_pct:>5.1f}"
        )
        results[sess] = {
            "n_trades": n,
            "win_rate": round(wr, 4),
            "avg_pnl": round(avg_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_minutes_to_exit": round(avg_min, 1),
            "tp_pct": round(tp_pct, 1),
            "sl_pct": round(sl_pct, 1),
            "eod_pct": round(eod_pct, 1),
        }

    # ── Finer granularity: hourly buckets ──
    log.info("")
    log.info("  HOURLY BREAKDOWN (ET exit hour):")
    log.info(f"  {'Hour':>6} {'N':>5} {'WR':>7} {'Avg PnL':>10} {'Total':>10} {'TP%':>6} {'SL%':>6}")
    log.info("  " + "-" * 55)

    df["exit_hour_int"] = df["exit_hour"].apply(lambda t: t.hour)
    hourly = {}
    for hr in sorted(df["exit_hour_int"].unique()):
        g = df[df["exit_hour_int"] == hr]
        n = len(g)
        wr = g["label_success"].mean()
        avg_pnl = g["pnl_dollars"].mean()
        total = g["pnl_dollars"].sum()
        tp_pct = (g["exit_reason"] == "TP").mean() * 100
        sl_pct = (g["exit_reason"] == "SL").mean() * 100
        log.info(
            f"  {hr:>5}h {n:>5} {wr:>6.1%} ${avg_pnl:>9.2f} ${total:>9.2f} {tp_pct:>5.1f} {sl_pct:>5.1f}"
        )
        hourly[f"{hr}h"] = {
            "n": n, "wr": round(wr, 4),
            "avg_pnl": round(avg_pnl, 2), "total": round(total, 2),
        }

    # ── Per-year × session ──
    log.info("")
    log.info("  PER-YEAR × SESSION:")
    df["year"] = pd.to_datetime(df["exit_time"]).dt.year

    log.info(f"  {'Year':<6} {'Morning':>18} {'Midday':>18} {'Close':>18}")
    log.info("  " + "-" * 62)

    yearly = {}
    for yr in sorted(df["year"].unique()):
        parts = []
        yr_data = {}
        for sess in sessions:
            g = df[(df["year"] == yr) & (df["session"] == sess)]
            if g.empty:
                parts.append(f"{'--':>18}")
                continue
            wr = g["label_success"].mean()
            total = g["pnl_dollars"].sum()
            parts.append(f"{wr:.0%} ${total:>8.0f}")
            yr_data[sess] = {"wr": round(wr, 4), "total": round(total, 2), "n": len(g)}
        log.info(f"  {yr:<6} " + " ".join(parts))
        yearly[str(yr)] = yr_data

    # ── Win rate by exit speed (quartiles) ──
    log.info("")
    log.info("  WIN RATE BY EXIT SPEED (minutes from entry to exit):")
    df["speed_q"] = pd.qcut(df["minutes_to_exit"], 4, labels=["Fast", "Medium", "Slow", "Very Slow"])
    for label in ["Fast", "Medium", "Slow", "Very Slow"]:
        g = df[df["speed_q"] == label]
        rng = f"{g['minutes_to_exit'].min():.0f}-{g['minutes_to_exit'].max():.0f}m"
        log.info(
            f"    {label:<12} ({rng:>12}): N={len(g):>4}  WR={g['label_success'].mean():.1%}  "
            f"Avg PnL=${g['pnl_dollars'].mean():.2f}"
        )

    return {"sessions": results, "hourly": hourly, "yearly": yearly}


# ═════════════════════════════════════════════════════════════════════
# Part 2:  Multi-strategy trades — by ENTRY hour
# ═════════════════════════════════════════════════════════════════════
def analyze_multi_strategy():
    path = Path("results/trades_filtered.csv")
    if not path.exists():
        log.info("No multi-strategy trades file found — skipping Part 2.")
        return {}

    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    # Convert UTC → ET (UTC-5 — matches original data offset)
    df["entry_et"] = df["entry_time"] - pd.Timedelta(hours=5)
    df["entry_hour"] = df["entry_et"].dt.hour

    # Assign session by entry hour
    def entry_session(h):
        if h < 12:
            return "Morning (9:30-12:00)"
        elif h < 14:
            return "Midday (12:00-14:00)"
        else:
            return "Close (14:00-16:00)"

    df["session"] = df["entry_hour"].apply(entry_session)
    df["win"] = (df["net_pnl"] > 0).astype(int)

    log.info("")
    log.info("=" * 78)
    log.info("  MULTI-STRATEGY TRADES — PERFORMANCE BY ENTRY HOUR")
    log.info("=" * 78)
    log.info(f"  {len(df)} trades from trades_filtered.csv\n")

    sessions = ["Morning (9:30-12:00)", "Midday (12:00-14:00)", "Close (14:00-16:00)"]
    ms_results = {}

    header = f"  {'Session':<25} {'N':>5} {'WR':>7} {'Avg PnL':>10} {'Total PnL':>12}"
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))

    for sess in sessions:
        g = df[df["session"] == sess]
        if g.empty:
            continue
        n = len(g)
        wr = g["win"].mean()
        avg = g["net_pnl"].mean()
        total = g["net_pnl"].sum()
        log.info(f"  {sess:<25} {n:>5} {wr:>6.1%} ${avg:>9.2f} ${total:>11.2f}")
        ms_results[sess] = {
            "n": n, "wr": round(wr, 4),
            "avg_pnl": round(avg, 2), "total": round(total, 2),
        }

    # Hourly
    log.info("")
    log.info(f"  {'Hour':>6} {'N':>5} {'WR':>7} {'Avg PnL':>10} {'Total':>10}")
    log.info("  " + "-" * 45)
    for hr in sorted(df["entry_hour"].unique()):
        g = df[df["entry_hour"] == hr]
        log.info(
            f"  {hr:>5}h {len(g):>5} {g['win'].mean():>6.1%} ${g['net_pnl'].mean():>9.2f} ${g['net_pnl'].sum():>9.2f}"
        )

    # By strategy type
    if "strategy_type" in df.columns:
        log.info("")
        log.info("  BY STRATEGY × SESSION:")
        log.info(f"  {'Strategy':<22} {'Session':<25} {'N':>4} {'WR':>6} {'Avg PnL':>9}")
        log.info("  " + "-" * 70)
        for strat in sorted(df["strategy_type"].unique()):
            for sess in sessions:
                g = df[(df["strategy_type"] == strat) & (df["session"] == sess)]
                if g.empty:
                    continue
                log.info(
                    f"  {strat:<22} {sess:<25} {len(g):>4} {g['win'].mean():>5.1%} ${g['net_pnl'].mean():>8.2f}"
                )

    return ms_results


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("=" * 78)
    log.info("  TRADE PERFORMANCE BY HOUR OF DAY")
    log.info("=" * 78)

    # Part 1 — EMA breakout by exit time
    ema_df = reconstruct_exit_times()
    ema_results = print_ema_analysis(ema_df)

    # Part 2 — Multi-strategy by entry time
    ms_results = analyze_multi_strategy()

    # Save all
    out_dir = Path("results/hour_of_day")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = {"ema_breakout_by_exit_session": ema_results, "multi_strategy_by_entry_session": ms_results}
    out_path = out_dir / "hour_analysis.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    log.info(f"\nSaved to {out_path}")
