# Dynamic Portfolio Allocation via DCC-GARCH & Hierarchical Risk Parity (HRP)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade Quantitative Finance framework combining **Dynamic Conditional Correlation (DCC-GARCH)** time-series econometrics with **Hierarchical Risk Parity (HRP)** machine learning portfolio allocation.

---

## 📌 Executive Summary

Classical Mean-Variance Optimization (Markowitz) suffers from severe instability due to noise in expected return estimates and inversion of ill-conditioned covariance matrices. 

This project solves both issues through a two-stage hybrid approach:
1. **DCC-GARCH(1,1):** Models time-varying, non-linear conditional correlations between assets, capturing volatility clustering and market regime shifts.
2. **Hierarchical Risk Parity (HRP):** Allocates portfolio capital using machine learning clustering (dendrograms) directly on DCC-derived correlation matrices **without requiring matrix inversion**.

---

## ⚙️ Mathematical Framework

### 1. Stage 1: Univariate GARCH(1,1) Filtering
Raw asset returns $r_{i,t}$ are filtered to extract conditional volatilities $\sigma_{i,t}$ and standardized residuals $e_{i,t}$:
$$e_{i,t} = \frac{r_{i,t} - \mu_i}{\sigma_{i,t}}$$

### 2. Stage 2: Scalar DCC(1,1) Correlation Modeling
The pseudo-covariance matrix $Q_t$ updates dynamically via:
$$Q_t = (1 - \alpha - \beta)\bar{Q} + \alpha (e_{t-1} e_{t-1}^T) + \beta Q_{t-1}$$
Dynamic correlation matrix $R_t$ is obtained by diagonal scaling:
$$R_t = (\text{diag}(Q_t))^{-1/2} Q_t (\text{diag}(Q_t))^{-1/2}$$
Parameters $\alpha$ and $\beta$ are estimated via **Quasi-Maximum Likelihood Estimation (QMLE)**.

### 3. Stage 3: HRP Allocation
1. **Distance Metric:** Converts $R_t$ to distance matrix $D_t = \sqrt{\frac{1}{2}(1 - R_t)}$.
2. **Clustering:** Builds hierarchical trees (dendrograms) using single-linkage agglomerative clustering.
3. **Quasi-Diagonalization & Recursive Bisection:** Sorts assets by cluster similarity and recursively allocates weights based on intra-cluster variance.

---

## 📂 Repository Structure

```text
.
├── data/                   # Historical asset price datasets
├── notebooks/              # Research & visualization notebooks
├── src/                    # Core source code
│   ├── garch.py            # Univariate GARCH filtering
│   ├── dcc.py              # DCC-GARCH QMLE optimization & correlation engine
│   ├── hrp.py              # Hierarchical Risk Parity weighting module
│   └── backtest.py         # Rolling-window backtesting framework
├── tests/                  # Unit tests (pytest)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation