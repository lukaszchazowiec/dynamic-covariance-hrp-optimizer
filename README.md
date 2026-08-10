# Dynamic Portfolio Allocation via DCC-GARCH & Hierarchical Risk Parity (HRP)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A small quantitative finance project combining **Dynamic Conditional Correlation (DCC-GARCH)** with **Hierarchical Risk Parity (HRP)** portfolio allocation, built as a self-study project while learning quantitative finance and risk management.

---

## Overview

Classical Mean-Variance Optimization (Markowitz) is known to be unstable in practice — it's very sensitive to estimation error in expected returns, and it requires inverting a covariance matrix, which becomes numerically fragile when assets are highly correlated.

This project explores an alternative, two-stage approach:

1. **DCC-GARCH(1,1):** models how volatility and correlation between assets change over time, instead of assuming a single, static covariance matrix for the whole sample.
2. **Hierarchical Risk Parity (HRP):** allocates capital using hierarchical clustering directly on the correlation structure, **without inverting the covariance matrix** — which is the main source of instability in classical mean-variance optimization.

The resulting HRP-DCC strategy is benchmarked against three alternatives — Equal-Weight (1/N), static HRP (Ledoit-Wolf shrinkage), and Minimum-Variance — on a 16-ETF universe spanning US equities, fixed income, and alternatives, using a walk-forward backtest.

**Note on results:** the DCC-GARCH-based HRP strategy underperformed all three benchmarks on risk-adjusted metrics in this backtest — see `notebooks/diagnostics.ipynb` for the full analysis.

---

## Methodology

### 1. Univariate GARCH(1,1) filtering

Each asset's returns $r_{i,t}$ are filtered through an individual GARCH(1,1) model (Student-t distributed, to account for fat tails) to obtain a time-varying conditional volatility $\sigma_{i,t}$ and a standardized residual:

$$e_{i,t} = \frac{r_{i,t}}{\sigma_{i,t}}$$

This separates "how much an asset moved" (volatility) from "in which direction, relative to its own scale" (standardized shock) — the second stage only models the latter.

### 2. Scalar DCC(1,1) correlation modeling

The standardized residuals feed into a Dynamic Conditional Correlation model. The pseudo-covariance matrix $Q_t$ evolves as:

$$Q_t = (1 - \alpha - \beta)\bar{Q} + \alpha (e_{t-1} e_{t-1}^T) + \beta Q_{t-1}$$

and is rescaled into a proper correlation matrix:

$$R_t = \left(\mathrm{diag}(Q_t)\right)^{-1/2} Q_t \left(\mathrm{diag}(Q_t)\right)^{-1/2}$$

Parameters $\alpha$ (reaction to new shocks) and $\beta$ (persistence of past correlation) are estimated via Quasi-Maximum Likelihood Estimation (QMLE), with a multi-start optimizer to guard against convergence to degenerate solutions (see the diagnostics notebook for why this was needed).

### 3. HRP allocation

Given a covariance matrix (here, the DCC-implied one), HRP allocates weights in three steps:

1. **Distance metric** — convert the correlation matrix into a distance matrix, $D_t = \sqrt{\tfrac{1}{2}(1 - R_t)}$.
2. **Hierarchical clustering** — group assets into a dendrogram (Ward linkage) based on this distance.
3. **Quasi-diagonalization & recursive bisection** — reorder assets by cluster similarity, then recursively split the tree, allocating more weight to lower-variance sub-clusters. No matrix inversion is needed anywhere in this process.

---

## Repository Structure

```text
.
├── data/                       # Cached price data (parquet)
├── notebooks/
│   └── diagnostics.ipynb       # Full diagnostic walkthrough: data checks, GARCH/DCC
│                                # diagnostics, HRP structure, backtest results & conclusions
├── src/
│   ├── data_loader.py          # Downloads/caches prices, computes log returns, normality tests
│   ├── dcc_garch.py            # Univariate GARCH filtering + DCC(1,1) QMLE estimation
│   ├── portfolio.py            # HRP allocation + benchmark strategies (EW, static HRP, Min-Var)
│   ├── backtest.py             # Walk-forward rolling backtest
│   ├── evaluation.py           # Performance metrics (Sharpe, Calmar, max drawdown, ...)
│   └── plots.py                # Correlation heatmaps, rolling correlation, dendrograms
├── requirements.txt
└── README.md
```

---

## Asset Universe

16 ETFs across four categories, chosen to give the covariance/correlation models something genuinely heterogeneous to work with:

- **US Equities (broad):** SPY, QQQ, IWM
- **US Equities (sectors):** XLF, XLE, XLV, XLU, XLK
- **Fixed Income:** AGG, TLT, IEF, HYG, LQD
- **Alternatives / Volatility:** GLD, TIP, SVXY

Sample period: 2011-10-10 to 2026-01-01 (start date constrained by SVXY's inception).

---

## What's in the diagnostics notebook

Rather than just presenting final numbers, `notebooks/diagnostics.ipynb` walks through the reasoning at each stage:

1. **Data checks** — return distributions, normality tests, static correlation.
2. **GARCH diagnostics** — parameter sanity checks, ACF of squared residuals, a look at SVXY's near-IGARCH behavior.
3. **DCC diagnostics** — persistence of the fitted correlation, comparison against a simple rolling-window correlation to sanity-check the model's smoothing behavior.
4. **DCC stability diagnostics** — an investigation into optimizer instability found during backtesting (some training windows converged to degenerate α+β≈0 solutions), how it was fixed (multi-start optimization + fallback), and confirmation that the fix didn't change the final results — meaning the underperformance found below is real, not a bug.
5. **Portfolio structure** — dendrogram interpretation and a comparison of weights across all four strategies on a single day.
6. **Backtest results** — performance metrics, equity curves, and weight evolution over the full sample.
7. **Conclusions** — a summary of findings and honest limitations.

---

## Running the project

```bash
pip install -r requirements.txt

# Run the full pipeline from src/, in order:
python src/data_loader.py    # downloads & caches price data
python src/dcc_garch.py      # sanity-checks the GARCH/DCC pipeline
python src/portfolio.py      # sanity-checks HRP weights vs. benchmarks
python src/backtest.py       # runs the full walk-forward backtest, saves results as CSV
python src/evaluation.py     # generates the metrics table and equity curve plots
```

Or open `notebooks/diagnostics.ipynb` for the full, step-by-step walkthrough with commentary.

---

## Known limitations

- **No transaction costs** are modeled — given HRP-DCC's relatively high turnover between rebalances, including costs would likely widen the performance gap versus the benchmarks further.
- **DCC estimation** uses Nelder-Mead, which can be unstable on short training windows (see diagnostics notebook); a gradient-based optimizer with a parameter transformation would likely be more robust.
- **No statistical significance testing** on the difference in Sharpe ratios between strategies — the reported gap could partly reflect sampling variation rather than a persistent, structural difference.

---

## Context

This project was built as a self-study project to get hands-on experience with dynamic covariance modeling and modern portfolio construction techniques beyond classical mean-variance optimization, as part of learning quantitative finance and risk management. It is a learning project, not a production system.