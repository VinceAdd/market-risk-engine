# risk_engine

A minimal, self-contained market risk library for portfolio VaR, CVaR, risk attribution, and model validation. Built for clarity and reproducibility — every number is traceable to a formula in the code.

---

## Project structure

```
risk_engine/
├── __init__.py          Public API — everything the caller needs is exported here
├── volatility.py        EWMA conditional volatility (feeds FHS models)
├── risk_models.py       VaR, CVaR, FHS variants, covariance attribution
├── rolling_var.py       Rolling out-of-sample forecast series for backtesting
└── modelvalidation.py   Statistical backtests and Basel Traffic Light

portfolio_demo.py        End-to-end demo: CSV inputs → full risk report
data/inbound/
├── positions.csv        Snapshot holdings (one date, one row per security)
├── prices.csv           Historical daily prices
└── fx_rates.csv         Historical daily FX rates to report currency
```

---

## Quick start

```bash
# generate sample data and run the full report
python portfolio_demo.py --sample

# run on your own CSV files
python portfolio_demo.py
  --positions data/inbound/positions.csv
  --prices    data/inbound/prices.csv
  --fx        data/inbound/fx_rates.csv
```

**Dependencies:** *requirements.txt*

---

## Input file format

### positions.csv
One row per security. Must contain a single snapshot date, the date the positions were recorded.

| Column | Description |
|---|---|
| Date | Snapshot date (YYYY-MM-DD) |
| Desk | Desk name |
| Book | Book name |
| Strategy | Strategy name |
| Security | Ticker / identifier |
| Quantity | Number of units held |
| Price_Currency | Currency of the security price |

### prices.csv
Historical daily prices, one row per security per date.

| Column | Description |
|---|---|
| Date | Trading date |
| Security | Ticker (must match positions.csv) |
| Price | Close price in local currency |
| Price_Currency | Currency of the price |

### fx_rates.csv
Historical daily FX rates, one row per currency per date.

| Column | Description |
|---|---|
| Date | Trading date |
| Currency | Currency code (e.g. EUR, JPY) |
| FX_to_Report_Ccy | Rate to convert 1 unit of this currency to the report currency |

EUR is included with `FX_to_Report_Ccy = 1.0`. Any currency absent from this file is assumed to already be in report currency.

---

## Computation pipeline

```
positions + prices + fx
        │
        ▼
MV_i(t) = Quantity_i × Price_i(t) × FX_i(t)      per-security market value
        │
        ▼
NAV(t) = Σ MV_i(t)                                 portfolio net asset value
        │
        ▼
r(t) = log(NAV(t) / NAV(t-1))                      daily log-returns
        │
        ▼
risk_engine                                          all measures below
```

---

## Risk measures

Value at risk measures use **1-day 99%** confidence unless stated otherwise.\
Conditional Value at risk measures use **1-day 97.5%** confidence unless stated otherwise.

### Volatility — `ewma_volatility`

EWMA conditional volatility with the RiskMetrics λ = 0.94 convention.

```
σ²_t = λ · σ²_{t-1} + (1 − λ) · r²_{t-1}
```

- No distribution assumption
- Effective memory ≈ 1/(1−λ) ≈ 17 days
- Feeds the FHS-VaR and FHS-CVaR models
- **Known limitation:** λ is a convention, not estimated from data. No mean reversion. Symmetric response to up/down shocks.

### Historical VaR — `var_historical`

The 1st percentile of the empirical return distribution. No distribution assumption. Limited to the tail actually observed in the sample window.

```
VaR = −Q_{1%}(r₁, ..., r_T)
```

### FHS-VaR — `var_fhs`

Filtered Historical Simulation. Standardises returns by EWMA volatility before taking the empirical quantile, then rescales by the current (most recent) volatility estimate.

```
z_t  = r_t / σ_t                        standardise
z_q  = Q_{1%}(z₁, ..., z_T)             empirical quantile of residuals
VaR  = −z_q × σ_T                       rescale by current vol
```

Captures the current volatility regime without assuming a distribution. When FHS-VaR > Historical VaR, current vol is elevated relative to the historical average.

### CVaR (Expected Shortfall) — `cvar`

Mean loss on the days that exceed the VaR threshold. Coherent risk measure mandated under Basel IV / FRTB at 97.5%.

```
CVaR = −E[r | r < −VaR]
```

CVaR ≥ VaR always. The gap between them indicates tail heaviness.

### FHS-CVaR — `cvar_fhs`

Same as FHS-VaR but uses the mean of the standardised tail rather than the quantile, then rescales by current vol.

### Risk Attribution — `var_parametric_cov`

Decomposes portfolio VaR into per-security contributions using the covariance matrix. Component VaR sums exactly to portfolio VaR.

```
component_VaR_i = w_i × (∂VaR / ∂w_i)
```

**This is not the primary risk number.** It is a decomposition tool only. It uses a Normal assumption scoped to the attribution step. The covariance matrix is estimated with Ledoit-Wolf shrinkage to reduce estimation noise.

---

## Backtesting

Rolling out-of-sample forecasts are generated by `rolling_var` and `rolling_cvar`. Each forecast at time t uses only data up to t−1. The first `window` entries are NaN. Default window: 250 trading days.

Four tests are run by `backtest_summary`. Read them in order:

### 1. Binomial test (frequency)

**Question:** Is the total count of breaches consistent with the model's confidence level?

Under H0, the number of breaches X ~ Binomial(T, 1%). The p-value is the exact probability P(X ≥ x | H0), with no asymptotic approximation. One-tailed upper test: flags too many breaches only.

| p-value | Interpretation                                        |
|---|-------------------------------------------------------|
| < 0.05 | REJECT : too many breaches, model underestimates risk |
| ≥ 0.05 | ok : breach count consistent with model               |

### 2. Christoffersen test (independence)

**Question:** Are breaches independent, or do they cluster in crises?

A model can pass the frequency test (right annual count) but still be dangerous if all failures arrive in the same two-week crisis window. Christoffersen counts the four possible day-to-day transitions (calm to calm, calm to breach, breach to calm, breach to breach) and tests whether P(breach | breach yesterday) equals P(breach | no breach yesterday).

The test decomposes into three LR statistics, each asymptotically chi-squared:

| Statistic | Question                     | df |
|---|------------------------------|---|
| LR_UC | Frequency (same as binomial) | 1 |
| LR_IND | Independence, no clustering  | 1 |
| LR_CC = LR_UC + LR_IND | Both at once                 | 2 |

If `p_independence < 0.05`, breaches are clustering. The most common cause is a volatility model that is too slow (EWMA λ too high, or window too long), it underestimates risk precisely during stress periods.

### 3. CVaR exceedance test — tail severity

**Question:** On the days we were hurt, was the CVaR forecast large enough?

On breach days only, tests whether mean(actual loss − CVaR forecast) = 0 via a one-sample t-test.

```
ratio = mean(actual loss on breach days) / mean(CVaR forecast on breach days)
```

| ratio | Interpretation |
|---|---|
| > 1.0 | Model underestimates tail severity |
| ≈ 1.0 | Well calibrated |
| < 1.0 | Model is conservative |

**Important limitation:** At 99% VaR over 250 days you expect only 2–3 breach days. The t-statistic is mean(excess) / (std(excess) / √n), and with n = 2–3 the standard error is so large that the test has almost no power. Treat the ratio as a qualitative signal. The test becomes statistically meaningful only after a long time period.

### 4. Basel Traffic Light (regulatory output)

**Not a statistical test.** Counts exceptions over the most recent 250 trading days and assigns a regulatory zone. Thresholds are conventions from Basel (1996), not statistically derived.\
The base multiplier of 3 is established in the Amendment to the Capital Accord to Incorporate Market Risks (Basel Committee, January 1996). (BCBS24, Section B.4. QUANTITATIVE STANDARDS (j))
Available at: https://www.bis.org/publ/bcbs22.pdf, specifically Table 2. (BCBS22)

| Zone | Exceptions | Capital multiplier |
|---|---|---|
| Green | 0–4 | 3.00 |
| Yellow | 5–9 | 3.40–3.85 |
| Red | 10+ | 4.00 |

A model can pass the binomial test at 5% significance and still land in Yellow. The two scales are not aligned.

### Decision framework

```
Binomial rejects?        → recalibrate the model
Christoffersen rejects?  → volatility model is too slow (increase lambda sensitivity or shorten window)
Basel Red / Yellow?      → regulatory output
CVaR ratio > 1.2?        → to consider with caution
All pass?                → model is performing as expected
```

---

## FX decomposition

The report decomposes total risk into equity and FX components using frozen FX rates at the snapshot date:

```
r_total  = r_equity + r_fx         (exact log decomposition)
FX add-on = VaR_total − VaR_equity
```

A negative FX add-on means the FX exposure is diversifying the portfolio — equity and FX moves are partially offsetting each other.

---

## Sample output

```
[NAV] 505 days |      486,247 ->      419,152  (report ccy)

[Weights from snapshot]
 7203 JP Equity        21.1%
 AAPL US Equity        27.6%
 ASML NA Equity        24.2%
 MSFT US Equity        27.0%

============================================================
 PORTFOLIO RISK REPORT (504 daily observations)
============================================================

[Volatility - EWMA lambda=0.94]
 Current : 0.96%/day  (15.2% ann.)
 Average : 0.90%/day  (14.2% ann.)

[1-day 99% Risk Measures - equity + FX]
                            VaR      CVaR
 ----------------------------------------
 Historical              2.331%    2.566%
 FHS (EWMA)              2.371%    2.744%  <- current vol elevated
 ----------------------------------------
 Historical: full return history, no vol adjustment
 FHS: standardised by EWMA vol, rescaled by today's sigma

[Risk Attribution - Covariance Model  (Normal, for decomposition only)]
 Ledoit-Wolf shrinkage delta : 0.39  (0=none, 1=full identity)
 Portfolio VaR (covariance)  : 2.055%  cf. Historical VaR: 2.331%

 By Security:
  7203 JP Equity        +0.296%  (14.4%)  [APAC / JP_Auto]
  AAPL US Equity        +0.643%  (31.3%)  [Americas / US_Tech]
  ASML NA Equity        +0.531%  (25.8%)  [EMEA / EU_Semis]
  MSFT US Equity        +0.585%  (28.5%)  [Americas / US_Tech]

 By Book:
  JP_Auto               +0.296%  (14.4%)
  EU_Semis              +0.531%  (25.8%)
  US_Tech               +1.228%  (59.8%)

 By Desk:
  APAC                  +0.296%  (14.4%)
  EMEA                  +0.531%  (25.8%)
  Americas              +1.228%  (59.8%)

[Backtest — rolling 250-day window @ 99%]
============================================================
  BACKTEST SUMMARY
============================================================
  Observations  : 254
  Exception rate: 0.79%  (expected 1.00%)
  Binomial [ok] 2/254 exceptions (0.79% vs expected 1.00%  [2.5 expected]  p=0.7224)
  Christoffersen  IND:ok (p=0.8583)  CC:ok (p=0.9245)
  CVaR exceedance [insufficient data — 2 exceptions] ratio=1.095
  Basel [Green] 2 exceptions over 250 days  capital multiplier x3.00
============================================================

[FX Risk Decomposition @ 99%]
                                     VaR     CVaR
 -----------------------------------------------
 Total (equity + FX)              2.331%   2.566%
 Equity (FX frozen)               2.394%   2.530%
 FX (equity frozen)               0.380%   0.417%
 -----------------------------------------------
 FX add-on  (total - equity)    -0.063%
```

---

## Design decisions

**Why Historical VaR as the primary number, not parametric?**
Historical VaR makes no distribution assumption. It uses the actual empirical tail of the portfolio's return history. The parametric covariance model is used only for attribution.

**Why Ledoit-Wolf shrinkage?**
The sample covariance may be noisy. Shrinkage reduces estimation error analytically without cross-validation.

**Why EWMA rather than GARCH for volatility?**
EWMA is a single-parameter model that is computationally trivial and directly comparable to RiskMetrics systems at other firms. It has no mean reversion and responds symmetrically to positive and negative shocks, both known limitations. GARCH would address these at the cost of estimation complexity. For a single-series portfolio-level vol estimate feeding FHS, EWMA is a reasonable and honest choice.

**Why log-returns?**
Log-returns are time-additive. The FX decomposition r_total = r_equity + r_fx holds exactly only in log space. Arithmetic returns would introduce a small cross-term that breaks the clean additive decomposition.

---

## References

- RiskMetrics Technical Document, J.P. Morgan (1996)
- Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix", *Journal of Portfolio Management*
- Christoffersen (1998), "Evaluating Interval Forecasts", *International Economic Review*
- McNeil & Frey (2000), "Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series", *Journal of Empirical Finance*
- Basel Committee on Banking Supervision (1996), "Supervisory Framework for the Use of Backtesting"
