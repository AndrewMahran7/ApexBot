# AdaptiveRegimeStrategy Historical Validation (2017-2025)

Generated: 2026-04-18T11:08:00.779918

## Strategy Config
Pure defaults — no optimization, no ML, no parameter changes.

## Execution Model
- Stop-order entry at opening range breakout level
- Entry price = OR high/low + buffer (pre-determined before bar)
- All features (EMA, ATR, volume) use standard rolling windows
- One trade per day per symbol
- No ML model involved

## Data Coverage
- Years tested: [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
- 2020: NOT AVAILABLE (no data file)
- 2021/2023/2024: extracted from 4-year aggregate files
- Symbols: MES, MNQ

## Per-Symbol Results

```
+------+--------+--------+----------+-----------+-----------+---------------+----------+--------------+--------+
| Year | Symbol | Trades | Win Rate | Avg Trade | Total PnL | Profit Factor |   Max DD | Final Equity | Sharpe |
+------+--------+--------+----------+-----------+-----------+---------------+----------+--------------+--------+
| 2017 | MES    |     61 |    49.2% |     $0.19 |    $11.75 |          1.02 |  $403.75 |    $25011.75 |   0.05 |
| 2017 | MNQ    |    146 |    41.1% |    $-3.81 |  $-556.50 |          0.81 |  $847.25 |    $24443.50 |  -0.91 |
| 2018 | MES    |     70 |    41.4% |    $-6.73 |  $-471.25 |          0.79 |  $900.00 |    $24528.75 |  -0.73 |
| 2018 | MNQ    |    154 |    37.7% |   $-14.43 | $-2222.50 |          0.71 | $2399.50 |    $22777.50 |  -1.60 |
| 2019 | MES    |     79 |    51.9% |     $8.25 |   $652.00 |          1.45 |  $232.75 |    $25652.00 |   1.19 |
| 2019 | MNQ    |    138 |    42.0% |     $0.72 |   $100.00 |          1.02 |  $657.75 |    $25097.25 |   0.11 |
| 2021 | MES    |     81 |    53.1% |     $6.21 |   $503.00 |          1.18 |  $553.00 |    $25503.00 |   0.60 |
| 2021 | MNQ    |    181 |    54.7% |    $37.45 |  $6779.00 |          1.57 | $1084.75 |    $31779.00 |   2.35 |
| 2022 | MES    |     95 |    55.8% |    $28.04 |  $2663.75 |          1.46 |  $666.50 |    $27663.75 |   1.47 |
| 2022 | MNQ    |    196 |    41.3% |    $-1.26 |  $-246.00 |          0.99 | $3000.75 |    $24754.00 |   0.02 |
| 2023 | MES    |     61 |    44.3% |     $4.29 |   $261.75 |          1.11 |  $349.25 |    $25261.75 |   0.34 |
| 2023 | MNQ    |    158 |    44.3% |     $8.11 |  $1282.00 |          1.11 | $1653.25 |    $26279.25 |   0.54 |
| 2024 | MES    |     57 |    42.1% |   $-11.14 |  $-635.25 |          0.79 |  $884.50 |    $24364.75 |  -0.63 |
| 2024 | MNQ    |    162 |    43.8% |    $-0.50 |   $-80.50 |          0.99 | $1693.25 |    $24919.50 |   0.02 |
| 2025 | MES    |     86 |    46.5% |    $15.17 |  $1304.25 |          1.24 |  $768.50 |    $26304.25 |   0.75 |
| 2025 | MNQ    |    182 |    48.9% |    $43.32 |  $7885.00 |          1.41 | $2897.00 |    $32879.50 |   1.77 |
+------+--------+--------+----------+-----------+-----------+---------------+----------+--------------+--------+
```

## Annual Combined Summary

```
+------+--------------+-----------+-------------+--------------+---------------------------------------------------------+
| Year | Total Trades | Total PnL | Best Symbol | Worst Symbol | Market Character                                        |
+------+--------------+-----------+-------------+--------------+---------------------------------------------------------+
| 2017 |          207 |  $-544.75 | MES         | MNQ          | Low vol, grind higher, difficult breakout year          |
| 2018 |          224 | $-2693.75 | MES         | MNQ          | Volatile, regime-shift year (Feb vol-spike, Q4 selloff) |
| 2019 |          217 |   $752.00 | MES         | MNQ          | Trending bull year after Q4-2018 selloff                |
| 2021 |          262 |  $7282.00 | MNQ         | MES          | Strong trend, expansion, low-vol breakout-friendly      |
| 2022 |          291 |  $2417.75 | MES         | MNQ          | Bear market, volatile, frequent reversals               |
| 2023 |          219 |  $1543.75 | MNQ         | MES          | Mixed — choppy early, stronger trend later              |
| 2024 |          219 |  $-715.75 | MNQ         | MES          | Mixed — broadening, rotational                          |
| 2025 |          268 |  $9189.25 | MNQ         | MES          | Strong trend, recent out-of-sample year                 |
+------+--------------+-----------+-------------+--------------+---------------------------------------------------------+
```

## Regime-Aware Interpretation

### 2017 — Low vol, grind higher, difficult breakout year
- **Losing year** ($-545). Strategy struggled in this regime.

### 2018 — Volatile, regime-shift year (Feb vol-spike, Q4 selloff)
- **Losing year** ($-2694). Strategy struggled in this regime.

### 2019 — Trending bull year after Q4-2018 selloff
- **Profitable** ($752). Strategy appears well-suited to this regime.

### 2021 — Strong trend, expansion, low-vol breakout-friendly
- **Profitable** ($7282). Strategy appears well-suited to this regime.

### 2022 — Bear market, volatile, frequent reversals
- **Profitable** ($2418). Strategy appears well-suited to this regime.

### 2023 — Mixed — choppy early, stronger trend later
- **Profitable** ($1544). Strategy appears well-suited to this regime.

### 2024 — Mixed — broadening, rotational
- **Losing year** ($-716). Strategy struggled in this regime.

### 2025 — Strong trend, recent out-of-sample year
- **Profitable** ($9189). Strategy appears well-suited to this regime.

## Final Classification

**ACCEPTABLE**

Basis: 5/8 years profitable, cumulative $17230, worst DD $3001

### Criteria Used
- Profitable years: 5/8 (62%)
- Cumulative PnL: $17230.50
- Total trades: 1907
- Worst single-year drawdown: $3000.75
- Average annual PnL: $2153.81

### Classification Scale
- ROBUST: ≥75% profitable years, positive cumulative, worst DD < $5000
- ACCEPTABLE: ≥50% profitable years, cumulative > -$2000
- FRAGILE: Below acceptable thresholds

## Caveats
- 2020 data is missing — a critical volatility year is not tested
- 2021/2023/2024 extracted from 4-year aggregate files (same data, filtered by date)
- MNQ has $2/point (vs MES $5/point) — PnL magnitudes differ
- No transaction cost optimization — uses $2.25/side + 1 tick slippage
- Features include current bar data in EMA/ATR (standard practice, ~2% weight)