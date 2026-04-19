"""
FINAL SYSTEM EVALUATION — EMA + Reduced-16 LogReg ML

Step 1: Retrain the reduced-16 LogReg model and save it.
Step 2: Run EMA-only vs EMA+ML(reduced) across 2017, 2018, 2019, 2024, 2025
        on MNQ (primary) and MES (secondary).
Step 3: Compute full metrics and classify system robustness.

Classification criteria:
  ROBUST      — Profitable ≥4/5 years on primary instrument, no catastrophic year
  ACCEPTABLE  — Profitable ≥3/5 years, or EMA-only profitable + ML adds value
  FRAGILE     — Profitable <3/5 years, large variance, sign flips with small changes
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
import json, logging, pickle, time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config.settings import BacktestConfig, StrategyConfig, INSTRUMENT_REGISTRY
from data.loader import load_bars
from backtest.engine import BacktestEngine
from strategy.hybrid_ema_ml import HybridEMAMLStrategy, HybridEMAMLConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/final_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reduced-16 features (from train_reduced_model.py, ranked by combined score)
REDUCED_FEATURES = [
    "f_risk_points", "f_regime_vol_trend", "f_ema_distance_pct",
    "f_price_ret_12bar", "f_vol_relative", "f_price_ema_dist",
    "f_price_ema_dist_pct", "f_ema_distance", "f_range_high",
    "f_regime_trend_direction", "f_price_ema", "f_range_size_vs_atr",
    "f_price_ema_slope", "f_direction_long", "f_range_vs_atr",
    "f_range_low",
]

MODEL_PATH = "models/ema_reduced16_logreg.pkl"

# ── Year → data file mappings ────────────────────────────────────────
MNQ_YEARS = {
    2017: ("data/mnq_2017.csv", None, None),
    2018: ("data/mnq_2018.csv", None, None),
    2019: ("data/mnq_2019.csv", None, None),
    2024: ("data/mnq_4y.csv", "2024-01-01", "2024-12-31"),
    2025: ("data/mnq_2025.csv", None, None),
}

MES_YEARS = {
    2017: ("data/mes_2017.csv", None, None),
    2018: ("data/mes_2018.csv", None, None),
    2019: ("data/mes_2019.csv", None, None),
    2024: ("data/mes_4y.csv", "2024-01-01", "2024-12-31"),
    2025: ("data/mes_2025.csv", None, None),
}

ML_THRESHOLD = 0.40   # best threshold from threshold sweep analysis


# ═════════════════════════════════════════════════════════════════════
# Step 1: Train and save reduced-16 LogReg
# ═════════════════════════════════════════════════════════════════════
def train_and_save_model():
    log.info("=" * 78)
    log.info("  STEP 1: TRAINING REDUCED-16 LOGREG MODEL")
    log.info("=" * 78)

    df = pd.read_csv("data/ema_candidates.csv", parse_dates=["session_date"])
    df["year"] = df["session_date"].dt.year

    train_df = df[df["year"] <= 2023].copy()
    test_df = df[df["year"] == 2024].copy()

    log.info(f"  Train: {len(train_df)} samples (2021-2023), Test: {len(test_df)} (2024)")
    log.info(f"  Features: {len(REDUCED_FEATURES)}")

    X_train = train_df[REDUCED_FEATURES].values
    y_train = train_df["label_success"].values
    X_test = test_df[REDUCED_FEATURES].values
    y_test = test_df["label_success"].values

    # Build sklearn pipeline (same format as existing model)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")),
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import roc_auc_score
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob_test)
    train_auc = roc_auc_score(y_train, y_prob_train)

    log.info(f"  Test AUC:  {test_auc:.4f}")
    log.info(f"  Train AUC: {train_auc:.4f}")
    log.info(f"  Overfit gap: {train_auc - test_auc:+.4f}")

    # Compute train medians for missing-value imputation
    train_medians = {col: float(train_df[col].median()) for col in REDUCED_FEATURES}

    # Save in same format as existing model
    model_dict = {
        "model": pipeline,
        "feature_columns": REDUCED_FEATURES,
        "model_type": "logistic_regression_reduced16",
        "train_medians": train_medians,
        "split_dates": {
            "train": {"start": "2021-01-04", "end": "2023-12-29"},
            "test": {"start": "2024-01-02", "end": "2024-12-30"},
        },
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_dict, f)

    log.info(f"  Saved to {MODEL_PATH}")

    return {
        "test_auc": round(test_auc, 4),
        "train_auc": round(train_auc, 4),
        "overfit_gap": round(train_auc - test_auc, 4),
        "n_features": len(REDUCED_FEATURES),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }


# ═════════════════════════════════════════════════════════════════════
# Step 2: Run backtests
# ═════════════════════════════════════════════════════════════════════
def run_single_backtest(instrument_name, data_file, start, end, ml_threshold, model_path):
    instrument = INSTRUMENT_REGISTRY[instrument_name]

    cfg = HybridEMAMLConfig(
        allow_shorts=True,
        ml_threshold=ml_threshold,
        model_path=model_path,
    )
    strategy = HybridEMAMLStrategy(cfg)

    strat_cfg = StrategyConfig(shorts_enabled=True, ema_enabled=True, ema_length=50)
    bt_cfg = BacktestConfig(
        slippage_ticks=1,
        commission_per_side=2.32,
        initial_capital=5000.0,
    )

    bars = load_bars(data_file, timezone=strat_cfg.timezone, start=start, end=end)
    engine = BacktestEngine(instrument, strat_cfg, bt_cfg, strategy=strategy)
    result = engine.run(bars)

    trades = result.trades
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wins": 0, "win_rate": 0, "total_pnl": 0,
                "avg_pnl": 0, "profit_factor": 0, "max_dd": 0}

    wins = sum(1 for t in trades if t.net_pnl > 0)
    total_pnl = sum(t.net_pnl for t in trades)
    avg_pnl = total_pnl / n
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    equity = 5000.0
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t.net_pnl
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    return {
        "trades": n,
        "wins": wins,
        "win_rate": round(wins / n * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "profit_factor": round(pf, 3),
        "max_dd": round(max_dd, 2),
    }


def run_instrument(instrument_name, year_config):
    log.info(f"\n{'=' * 78}")
    log.info(f"  {instrument_name}: EMA-ONLY vs EMA+ML(Reduced-16)")
    log.info(f"  ML threshold={ML_THRESHOLD} | Model: {MODEL_PATH}")
    log.info("=" * 78)

    rows = []
    for year in sorted(year_config.keys()):
        data_file, start, end = year_config[year]
        log.info(f"\n  --- {year} ---")

        # EMA-only (threshold=0.0 accepts everything)
        log.info(f"    EMA-only ...")
        ema = run_single_backtest(instrument_name, data_file, start, end,
                                  ml_threshold=0.0, model_path=MODEL_PATH)
        log.info(f"      {ema['trades']} trades, WR {ema['win_rate']}%, "
                 f"PnL ${ema['total_pnl']:.2f}, PF {ema['profit_factor']:.3f}")

        # EMA+ML
        log.info(f"    EMA+ML ...")
        ml = run_single_backtest(instrument_name, data_file, start, end,
                                 ml_threshold=ML_THRESHOLD, model_path=MODEL_PATH)
        log.info(f"      {ml['trades']} trades, WR {ml['win_rate']}%, "
                 f"PnL ${ml['total_pnl']:.2f}, PF {ml['profit_factor']:.3f}")

        rows.append({
            "year": year,
            "ema": ema,
            "ml": ml,
            "pnl_delta": round(ml["total_pnl"] - ema["total_pnl"], 2),
            "wr_delta": round(ml["win_rate"] - ema["win_rate"], 1),
        })

    return rows


def print_comparison(instrument_name, rows):
    log.info(f"\n{'=' * 110}")
    log.info(f"  {instrument_name}: FULL METRICS")
    log.info(f"{'=' * 110}")

    hdr = (f"  {'Year':<6} | {'EMA Trd':>7} {'WR':>6} {'PnL':>10} {'PF':>6} "
           f"{'MaxDD':>8} | {'ML Trd':>7} {'WR':>6} {'PnL':>10} {'PF':>6} "
           f"{'MaxDD':>8} | {'dPnL':>9} {'dWR':>5}")
    log.info(hdr)
    log.info("  " + "-" * 106)

    for r in rows:
        e, m = r["ema"], r["ml"]
        log.info(
            f"  {r['year']:<6} | {e['trades']:>7} {e['win_rate']:>5.1f}% "
            f"${e['total_pnl']:>9.2f} {e['profit_factor']:>5.3f} "
            f"${e['max_dd']:>7.2f} | {m['trades']:>7} {m['win_rate']:>5.1f}% "
            f"${m['total_pnl']:>9.2f} {m['profit_factor']:>5.3f} "
            f"${m['max_dd']:>7.2f} | ${r['pnl_delta']:>8.2f} {r['wr_delta']:>+4.1f}%"
        )

    # Totals
    t_ema_trades = sum(r["ema"]["trades"] for r in rows)
    t_ml_trades = sum(r["ml"]["trades"] for r in rows)
    t_ema_pnl = sum(r["ema"]["total_pnl"] for r in rows)
    t_ml_pnl = sum(r["ml"]["total_pnl"] for r in rows)
    t_ema_wins = sum(r["ema"]["wins"] for r in rows)
    t_ml_wins = sum(r["ml"]["wins"] for r in rows)
    t_ema_wr = t_ema_wins / t_ema_trades * 100 if t_ema_trades else 0
    t_ml_wr = t_ml_wins / t_ml_trades * 100 if t_ml_trades else 0

    log.info("  " + "-" * 106)
    log.info(
        f"  {'TOTAL':<6} | {t_ema_trades:>7} {t_ema_wr:>5.1f}% "
        f"${t_ema_pnl:>9.2f} {'':>5} {'':>8} | "
        f"{t_ml_trades:>7} {t_ml_wr:>5.1f}% "
        f"${t_ml_pnl:>9.2f} {'':>5} {'':>8} | "
        f"${t_ml_pnl - t_ema_pnl:>8.2f} {t_ml_wr - t_ema_wr:>+4.1f}%"
    )

    return {
        "ema_total_pnl": round(t_ema_pnl, 2),
        "ml_total_pnl": round(t_ml_pnl, 2),
        "ema_total_trades": t_ema_trades,
        "ml_total_trades": t_ml_trades,
        "ema_wr": round(t_ema_wr, 1),
        "ml_wr": round(t_ml_wr, 1),
    }


# ═════════════════════════════════════════════════════════════════════
# Step 3: Classify robustness
# ═════════════════════════════════════════════════════════════════════
def classify_robustness(mnq_rows, mes_rows, mnq_summary, mes_summary):
    log.info(f"\n{'=' * 78}")
    log.info("  ROBUSTNESS CLASSIFICATION")
    log.info("=" * 78)

    # Criteria based on MNQ (primary instrument)
    ema_profitable_years = sum(1 for r in mnq_rows if r["ema"]["total_pnl"] > 0)
    ml_profitable_years = sum(1 for r in mnq_rows if r["ml"]["total_pnl"] > 0)
    ml_adds_value = mnq_summary["ml_total_pnl"] > mnq_summary["ema_total_pnl"]
    ema_positive_total = mnq_summary["ema_total_pnl"] > 0
    ml_positive_total = mnq_summary["ml_total_pnl"] > 0

    # Max single-year loss
    ema_max_loss = min(r["ema"]["total_pnl"] for r in mnq_rows)
    ml_max_loss = min(r["ml"]["total_pnl"] for r in mnq_rows)

    # Variance of annual PnL
    ema_annual_pnls = [r["ema"]["total_pnl"] for r in mnq_rows]
    ml_annual_pnls = [r["ml"]["total_pnl"] for r in mnq_rows]
    ema_std = np.std(ema_annual_pnls)
    ml_std = np.std(ml_annual_pnls)

    # MES cross-check
    mes_ema_profitable = sum(1 for r in mes_rows if r["ema"]["total_pnl"] > 0)

    log.info(f"\n  MNQ (Primary Instrument):")
    log.info(f"    EMA-only profitable years: {ema_profitable_years}/5")
    log.info(f"    EMA+ML profitable years:   {ml_profitable_years}/5")
    log.info(f"    EMA-only total PnL:        ${mnq_summary['ema_total_pnl']:,.2f}")
    log.info(f"    EMA+ML total PnL:          ${mnq_summary['ml_total_pnl']:,.2f}")
    log.info(f"    ML adds value:             {ml_adds_value}")
    log.info(f"    EMA worst year:            ${ema_max_loss:,.2f}")
    log.info(f"    ML worst year:             ${ml_max_loss:,.2f}")
    log.info(f"    EMA annual PnL std:        ${ema_std:,.2f}")
    log.info(f"    ML annual PnL std:         ${ml_std:,.2f}")
    log.info(f"\n  MES (Cross-check):")
    log.info(f"    EMA-only profitable years: {mes_ema_profitable}/5")
    log.info(f"    MES total EMA PnL:         ${mes_summary['ema_total_pnl']:,.2f}")

    # Classification logic
    log.info(f"\n  APPLYING CRITERIA:")

    score = 0
    reasons = []

    # Profitable years
    if ml_profitable_years >= 4:
        score += 3
        reasons.append(f"+3: ML profitable {ml_profitable_years}/5 years")
    elif ml_profitable_years >= 3:
        score += 2
        reasons.append(f"+2: ML profitable {ml_profitable_years}/5 years")
    elif ml_profitable_years >= 2:
        score += 1
        reasons.append(f"+1: ML profitable {ml_profitable_years}/5 years")
    else:
        score -= 2
        reasons.append(f"-2: ML profitable only {ml_profitable_years}/5 years")

    # Total PnL positive
    if ml_positive_total:
        score += 2
        reasons.append(f"+2: ML total PnL positive (${mnq_summary['ml_total_pnl']:,.2f})")
    else:
        score -= 2
        reasons.append(f"-2: ML total PnL negative (${mnq_summary['ml_total_pnl']:,.2f})")

    # EMA baseline
    if ema_profitable_years >= 3:
        score += 1
        reasons.append(f"+1: EMA-only profitable {ema_profitable_years}/5 years")
    else:
        score -= 1
        reasons.append(f"-1: EMA-only profitable only {ema_profitable_years}/5 years")

    # ML adds value
    if ml_adds_value:
        score += 1
        reasons.append("+1: ML improves over EMA-only")
    else:
        score -= 1
        reasons.append("-1: ML does NOT improve over EMA-only")

    # Worst year severity (relative to starting capital $5,000)
    worst_pct = ml_max_loss / 5000 * 100
    if ml_max_loss > -500:
        score += 1
        reasons.append(f"+1: Worst year contained (${ml_max_loss:,.2f} = {worst_pct:.1f}%)")
    elif ml_max_loss > -2000:
        reasons.append(f" 0: Worst ML year = ${ml_max_loss:,.2f} ({worst_pct:.1f}%)")
    else:
        score -= 2
        reasons.append(f"-2: Catastrophic worst year (${ml_max_loss:,.2f} = {worst_pct:.1f}%)")

    # Cross-instrument check
    if mes_ema_profitable >= 3:
        score += 1
        reasons.append(f"+1: MES cross-check: {mes_ema_profitable}/5 profitable years")
    elif mes_ema_profitable >= 2:
        reasons.append(f" 0: MES cross-check: {mes_ema_profitable}/5 profitable years")
    else:
        score -= 1
        reasons.append(f"-1: MES weak: only {mes_ema_profitable}/5 profitable")

    for r in reasons:
        log.info(f"    {r}")

    log.info(f"\n    TOTAL SCORE: {score}")

    if score >= 6:
        classification = "ROBUST"
    elif score >= 3:
        classification = "ACCEPTABLE"
    else:
        classification = "FRAGILE"

    log.info(f"\n    ┌─────────────────────────────────────┐")
    log.info(f"    │  CLASSIFICATION:  {classification:<18} │")
    log.info(f"    └─────────────────────────────────────┘")

    # Honest caveats
    log.info(f"\n  CAVEATS:")
    log.info(f"    - ML model trained on MES 2021-2023, tested on MES 2024")
    log.info(f"    - 2017-2019 and MNQ are TRUE out-of-sample (different instrument/period)")
    log.info(f"    - 2025 is forward OOS (future data relative to training)")
    log.info(f"    - Sample sizes are small (~250 trades/year)")
    log.info(f"    - Transaction costs modeled but real slippage may vary")
    log.info(f"    - No position sizing optimization applied")

    return {
        "classification": classification,
        "score": score,
        "reasons": reasons,
        "ml_profitable_years": ml_profitable_years,
        "ema_profitable_years": ema_profitable_years,
    }


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    # Step 1
    model_metrics = train_and_save_model()

    # Step 2
    mnq_rows = run_instrument("MNQ", MNQ_YEARS)
    mnq_summary = print_comparison("MNQ", mnq_rows)

    mes_rows = run_instrument("MES", MES_YEARS)
    mes_summary = print_comparison("MES", mes_rows)

    # Step 3
    classification = classify_robustness(mnq_rows, mes_rows, mnq_summary, mes_summary)

    elapsed = time.time() - t0
    log.info(f"\n  Total runtime: {elapsed:.0f}s")

    # Save everything
    output = {
        "model": model_metrics,
        "ml_threshold": ML_THRESHOLD,
        "mnq": {
            "summary": mnq_summary,
            "years": [{
                "year": r["year"],
                "ema": r["ema"],
                "ml": r["ml"],
                "pnl_delta": r["pnl_delta"],
                "wr_delta": r["wr_delta"],
            } for r in mnq_rows],
        },
        "mes": {
            "summary": mes_summary,
            "years": [{
                "year": r["year"],
                "ema": r["ema"],
                "ml": r["ml"],
                "pnl_delta": r["pnl_delta"],
                "wr_delta": r["wr_delta"],
            } for r in mes_rows],
        },
        "classification": classification,
    }

    out_path = OUT_DIR / "final_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
