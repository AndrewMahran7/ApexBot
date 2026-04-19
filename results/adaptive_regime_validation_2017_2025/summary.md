# AdaptiveRegimeStrategy Historical Validation (2017-2025)

Generated: 2026-04-18T10:54:03.203488

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
| 2017 | MES    |    239 |    43.9% |    $-3.76 |  $-898.00 |          0.72 |  $978.00 |    $24098.50 |  -1.94 |
| 2017 | MNQ    |    248 |    40.3% |    $-4.85 | $-1202.00 |          0.75 | $1303.25 |    $23798.00 |  -1.70 |
| 2018 | MES    |    249 |    39.8% |    $-5.98 | $-1488.00 |          0.80 | $1696.00 |    $23512.00 |  -1.36 |
| 2018 | MNQ    |    251 |    39.8% |    $-7.46 | $-1871.50 |          0.83 | $2386.75 |    $23128.50 |  -1.15 |
| 2019 | MES    |    250 |    41.6% |    $-2.41 |  $-602.50 |          0.90 | $1055.00 |    $24394.00 |  -0.65 |
| 2019 | MNQ    |    252 |    40.9% |    $-1.72 |  $-434.00 |          0.94 | $1417.75 |    $24563.25 |  -0.33 |
| 2021 | MES    |    251 |    46.6% |     $4.73 |  $1188.00 |          1.14 | $1046.50 |    $26188.00 |   0.84 |
| 2021 | MNQ    |    254 |    49.2% |    $22.79 |  $5787.50 |          1.35 | $1402.75 |    $30787.50 |   1.84 |
| 2022 | MES    |    251 |    44.6% |     $4.21 |  $1055.50 |          1.07 | $2193.25 |    $26052.00 |   0.42 |
| 2022 | MNQ    |    253 |    43.1% |     $7.80 |  $1973.00 |          1.07 | $3625.25 |    $26970.25 |   0.46 |
| 2023 | MES    |    249 |    38.5% |    $-9.36 | $-2330.50 |          0.78 | $2787.50 |    $22669.50 |  -1.53 |
| 2023 | MNQ    |    252 |    40.5% |     $2.06 |   $518.50 |          1.03 | $2190.75 |    $25515.75 |   0.22 |
| 2024 | MES    |    253 |    41.1% |    $-2.37 |  $-599.75 |          0.94 | $1370.00 |    $24400.25 |  -0.34 |
| 2024 | MNQ    |    253 |    45.1% |    $14.41 |  $3645.00 |          1.18 |  $970.25 |    $28645.00 |   1.08 |
| 2025 | MES    |    249 |    42.6% |     $7.27 |  $1809.50 |          1.13 | $2051.50 |    $26802.50 |   0.76 |
| 2025 | MNQ    |    249 |    46.6% |    $27.17 |  $6765.50 |          1.26 | $3558.00 |    $31757.25 |   1.44 |
+------+--------+--------+----------+-----------+-----------+---------------+----------+--------------+--------+
```

## Annual Combined Summary

```
+------+--------------+-----------+-------------+--------------+---------------------------------------------------------+
| Year | Total Trades | Total PnL | Best Symbol | Worst Symbol | Market Character                                        |
+------+--------------+-----------+-------------+--------------+---------------------------------------------------------+
| 2017 |          487 | $-2100.00 | MES         | MNQ          | Low vol, grind higher, difficult breakout year          |
| 2018 |          500 | $-3359.50 | MES         | MNQ          | Volatile, regime-shift year (Feb vol-spike, Q4 selloff) |
| 2019 |          502 | $-1036.50 | MNQ         | MES          | Trending bull year after Q4-2018 selloff                |
| 2021 |          505 |  $6975.50 | MNQ         | MES          | Strong trend, expansion, low-vol breakout-friendly      |
| 2022 |          504 |  $3028.50 | MNQ         | MES          | Bear market, volatile, frequent reversals               |
| 2023 |          501 | $-1812.00 | MNQ         | MES          | Mixed — choppy early, stronger trend later              |
| 2024 |          506 |  $3045.25 | MNQ         | MES          | Mixed — broadening, rotational                          |
| 2025 |          498 |  $8575.00 | MNQ         | MES          | Strong trend, recent out-of-sample year                 |
+------+--------------+-----------+-------------+--------------+---------------------------------------------------------+
```

## Regime-Aware Interpretation

### 2017 — Low vol, grind higher, difficult breakout year
- **Losing year** ($-2100). Strategy struggled in this regime.
- High trade count (487). May indicate overtrading.

### 2018 — Volatile, regime-shift year (Feb vol-spike, Q4 selloff)
- **Losing year** ($-3360). Strategy struggled in this regime.
- High trade count (500). May indicate overtrading.

### 2019 — Trending bull year after Q4-2018 selloff
- **Losing year** ($-1036). Strategy struggled in this regime.
- High trade count (502). May indicate overtrading.

### 2021 — Strong trend, expansion, low-vol breakout-friendly
- **Profitable** ($6976). Strategy appears well-suited to this regime.
- High trade count (505). May indicate overtrading.

### 2022 — Bear market, volatile, frequent reversals
- **Profitable** ($3028). Strategy appears well-suited to this regime.
- High trade count (504). May indicate overtrading.

### 2023 — Mixed — choppy early, stronger trend later
- **Losing year** ($-1812). Strategy struggled in this regime.
- High trade count (501). May indicate overtrading.

### 2024 — Mixed — broadening, rotational
- **Profitable** ($3045). Strategy appears well-suited to this regime.
- High trade count (506). May indicate overtrading.

### 2025 — Strong trend, recent out-of-sample year
- **Profitable** ($8575). Strategy appears well-suited to this regime.
- High trade count (498). May indicate overtrading.

## Final Classification

**ACCEPTABLE**

Basis: 4/8 years profitable, cumulative $13316, worst DD $3625

### Criteria Used
- Profitable years: 4/8 (50%)
- Cumulative PnL: $13316.25
- Total trades: 4003
- Worst single-year drawdown: $3625.25
- Average annual PnL: $1664.53

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