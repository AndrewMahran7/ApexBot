# AdaptiveRegimeStrategy Historical Validation (2017-2025)

Generated: 2026-04-18T11:44:32.596213

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
| 2017 | MNQ    |    145 |    44.1% |    $-1.09 |  $-158.00 |          0.94 |  $764.75 |    $24842.00 |  -0.26 |
| 2018 | MES    |     70 |    41.4% |    $-6.73 |  $-471.25 |          0.79 |  $900.00 |    $24528.75 |  -0.73 |
| 2018 | MNQ    |    154 |    39.6% |   $-10.39 | $-1599.50 |          0.79 | $1776.50 |    $23400.50 |  -1.13 |
| 2019 | MES    |     79 |    51.9% |     $8.25 |   $652.00 |          1.45 |  $232.75 |    $25652.00 |   1.19 |
| 2019 | MNQ    |    136 |    42.6% |     $2.07 |   $281.50 |          1.07 |  $609.75 |    $25278.75 |   0.29 |
| 2021 | MES    |     81 |    53.1% |     $6.21 |   $503.00 |          1.18 |  $553.00 |    $25503.00 |   0.60 |
| 2021 | MNQ    |    181 |    54.7% |    $37.45 |  $6778.00 |          1.57 | $1084.75 |    $31778.00 |   2.35 |
| 2022 | MES    |     95 |    55.8% |    $28.04 |  $2663.75 |          1.46 |  $666.50 |    $27663.75 |   1.47 |
| 2022 | MNQ    |    196 |    41.8% |     $2.42 |   $474.00 |          1.02 | $3000.75 |    $25474.00 |   0.18 |
| 2023 | MES    |     61 |    44.3% |     $4.29 |   $261.75 |          1.11 |  $349.25 |    $25261.75 |   0.34 |
| 2023 | MNQ    |    158 |    44.9% |    $10.46 |  $1653.00 |          1.14 | $1500.25 |    $26650.25 |   0.68 |
| 2024 | MES    |     57 |    42.1% |   $-11.14 |  $-635.25 |          0.79 |  $884.50 |    $24364.75 |  -0.63 |
| 2024 | MNQ    |    162 |    44.4% |     $1.91 |   $310.00 |          1.02 | $1488.75 |    $25310.00 |   0.15 |
| 2025 | MES    |     86 |    46.5% |    $15.17 |  $1304.25 |          1.24 |  $768.50 |    $26304.25 |   0.75 |
| 2025 | MNQ    |    182 |    50.0% |    $46.80 |  $8518.00 |          1.46 | $2683.50 |    $33512.50 |   1.90 |
+------+--------+--------+----------+-----------+-----------+---------------+----------+--------------+--------+
```

## Annual Combined Summary

```
+------+--------------+-----------+-------------+--------------+---------------------------------------------------------+
| Year | Total Trades | Total PnL | Best Symbol | Worst Symbol | Market Character                                        |
+------+--------------+-----------+-------------+--------------+---------------------------------------------------------+
| 2017 |          206 |  $-146.25 | MES         | MNQ          | Low vol, grind higher, difficult breakout year          |
| 2018 |          224 | $-2070.75 | MES         | MNQ          | Volatile, regime-shift year (Feb vol-spike, Q4 selloff) |
| 2019 |          215 |   $933.50 | MES         | MNQ          | Trending bull year after Q4-2018 selloff                |
| 2021 |          262 |  $7281.00 | MNQ         | MES          | Strong trend, expansion, low-vol breakout-friendly      |
| 2022 |          291 |  $3137.75 | MES         | MNQ          | Bear market, volatile, frequent reversals               |
| 2023 |          219 |  $1914.75 | MNQ         | MES          | Mixed — choppy early, stronger trend later              |
| 2024 |          219 |  $-325.25 | MNQ         | MES          | Mixed — broadening, rotational                          |
| 2025 |          268 |  $9822.25 | MNQ         | MES          | Strong trend, recent out-of-sample year                 |
+------+--------------+-----------+-------------+--------------+---------------------------------------------------------+
```

## Regime-Aware Interpretation

### 2017 — Low vol, grind higher, difficult breakout year
- **Marginally negative** ($-146). Slight drag, not catastrophic.

### 2018 — Volatile, regime-shift year (Feb vol-spike, Q4 selloff)
- **Losing year** ($-2071). Strategy struggled in this regime.

### 2019 — Trending bull year after Q4-2018 selloff
- **Profitable** ($934). Strategy appears well-suited to this regime.

### 2021 — Strong trend, expansion, low-vol breakout-friendly
- **Profitable** ($7281). Strategy appears well-suited to this regime.

### 2022 — Bear market, volatile, frequent reversals
- **Profitable** ($3138). Strategy appears well-suited to this regime.

### 2023 — Mixed — choppy early, stronger trend later
- **Profitable** ($1915). Strategy appears well-suited to this regime.

### 2024 — Mixed — broadening, rotational
- **Marginally negative** ($-325). Slight drag, not catastrophic.

### 2025 — Strong trend, recent out-of-sample year
- **Profitable** ($9822). Strategy appears well-suited to this regime.

## Final Classification

**ACCEPTABLE**

Basis: 5/8 years profitable, cumulative $20547, worst DD $3001

### Criteria Used
- Profitable years: 5/8 (62%)
- Cumulative PnL: $20547.00
- Total trades: 1904
- Worst single-year drawdown: $3000.75
- Average annual PnL: $2568.38

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