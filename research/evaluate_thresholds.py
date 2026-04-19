"""
Threshold sweep on baseline logistic regression model.
Uses time-based split (train 2021-2023, test 2024).
Model is retrained identically — NO hyperparameter changes.
Evaluates trading metrics at thresholds 0.30 to 0.90.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/baseline_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_VALUE = 5.0        # MES: $5/point
COMMISSION  = 2.32 * 2   # round-trip
SLIPPAGE_PTS = 0.25      # 1 tick

# ── Load & split ─────────────────────────────────────────────────────
df = pd.read_csv("data/ema_candidates.csv", parse_dates=["session_date"])
df["year"] = df["session_date"].dt.year

feat_cols = [c for c in df.columns if c.startswith("f_")]
constant = [c for c in feat_cols if df[c].nunique() <= 1]
feat_cols = [c for c in feat_cols if c not in constant]

train_df = df[df["year"] <= 2023].copy()
test_df  = df[df["year"] == 2024].copy()

X_train = train_df[feat_cols].values
y_train = train_df["label_success"].values
X_test  = test_df[feat_cols].values
y_test  = test_df["label_success"].values
pnl_pts = test_df["label_pnl_pts"].values

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train identical model ────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
model.fit(X_train_sc, y_train)
y_prob = model.predict_proba(X_test_sc)[:, 1]

log.info(f"Test set: {len(y_test)} candidates, base win rate {y_test.mean():.3f}")
log.info(f"Prob distribution: mean={y_prob.mean():.3f}, std={y_prob.std():.3f}, "
         f"min={y_prob.min():.3f}, max={y_prob.max():.3f}")

# ── Compute PnL in dollars ───────────────────────────────────────────
pnl_dollars = pnl_pts * POINT_VALUE - SLIPPAGE_PTS * POINT_VALUE - COMMISSION

log.info(f"\nAll trades (no filter): {len(pnl_dollars)} trades, "
         f"win rate {y_test.mean():.3f}, "
         f"avg PnL ${pnl_dollars.mean():.2f}, "
         f"total PnL ${pnl_dollars.sum():.2f}")

# ── Threshold sweep ──────────────────────────────────────────────────
thresholds = np.arange(0.30, 0.91, 0.05)
rows = []

log.info(f"\n{'Thresh':>7} {'Trades':>7} {'WinRate':>8} {'AvgPnL':>9} "
         f"{'TotalPnL':>10} {'PF':>6} {'MaxDD':>8} {'Rejected':>9}")
log.info("-" * 75)

for thresh in thresholds:
    mask = y_prob >= thresh
    n = mask.sum()

    if n == 0:
        rows.append({
            "threshold": round(float(thresh), 2),
            "n_trades": 0, "win_rate": 0, "avg_pnl": 0,
            "total_pnl": 0, "profit_factor": 0,
            "max_drawdown": 0, "pct_rejected": 100.0,
        })
        log.info(f"{thresh:7.2f} {'—no trades—':>60}")
        continue

    sel_pnl = pnl_dollars[mask]
    sel_win = y_test[mask]
    wr = sel_win.mean()
    avg = sel_pnl.mean()
    total = sel_pnl.sum()

    # Profit factor
    gross_profit = sel_pnl[sel_pnl > 0].sum()
    gross_loss   = abs(sel_pnl[sel_pnl < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    equity = np.cumsum(sel_pnl)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = dd.max()

    pct_rejected = (1 - n / len(y_test)) * 100

    rows.append({
        "threshold": round(float(thresh), 2),
        "n_trades": int(n),
        "win_rate": round(float(wr), 4),
        "avg_pnl": round(float(avg), 2),
        "total_pnl": round(float(total), 2),
        "profit_factor": round(float(pf), 3),
        "max_drawdown": round(float(max_dd), 2),
        "pct_rejected": round(float(pct_rejected), 1),
    })

    log.info(f"{thresh:7.2f} {n:7d} {wr:8.3f} ${avg:8.2f} ${total:9.2f} "
             f"{pf:6.2f} ${max_dd:7.2f} {pct_rejected:8.1f}%")

# ── Rejected trade analysis ─────────────────────────────────────────
log.info(f"\n{'='*60}")
log.info("REJECTED TRADES ANALYSIS (what the filter removes)")
log.info(f"{'='*60}")
log.info(f"{'Thresh':>7} {'Kept':>6} {'RejN':>6} {'RejWR':>7} {'RejAvg':>9} {'RejTotal':>10}")
log.info("-" * 55)

for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    kept = y_prob >= thresh
    rejected = ~kept
    n_rej = rejected.sum()
    if n_rej == 0:
        continue
    rej_pnl = pnl_dollars[rejected]
    rej_wr = y_test[rejected].mean()
    log.info(f"{thresh:7.2f} {kept.sum():6d} {n_rej:6d} {rej_wr:7.3f} "
             f"${rej_pnl.mean():8.2f} ${rej_pnl.sum():9.2f}")

# ── Comparison: filtered vs unfiltered per-trade quality ─────────────
log.info(f"\n{'='*60}")
log.info("QUALITY COMPARISON vs NO FILTER")
log.info(f"{'='*60}")
base_avg = pnl_dollars.mean()
base_wr  = y_test.mean()
for thresh in thresholds:
    mask = y_prob >= thresh
    n = mask.sum()
    if n < 5:
        continue
    sel_pnl = pnl_dollars[mask]
    sel_wr = y_test[mask].mean()
    wr_delta = sel_wr - base_wr
    avg_delta = sel_pnl.mean() - base_avg
    log.info(f"  thresh {thresh:.2f}: WR {sel_wr:.3f} ({wr_delta:+.3f}), "
             f"avg ${sel_pnl.mean():.2f} ({avg_delta:+.2f}), "
             f"n={n}")

# ── Save ─────────────────────────────────────────────────────────────
out_path = OUT_DIR / "threshold_sweep.json"
with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)

sweep_df = pd.DataFrame(rows)
csv_path = OUT_DIR / "threshold_sweep.csv"
sweep_df.to_csv(csv_path, index=False)

log.info(f"\nSaved to {out_path} and {csv_path}")
