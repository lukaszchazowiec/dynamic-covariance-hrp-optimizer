import os
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats

# Define your 16 ETF universe exactly as planned
ETF_UNIVERSE = [
    # US Equities - Broad
    "SPY", "QQQ", "IWM",
    # US Equities - Sectors
    "XLF", "XLE", "XLV", "XLU", "XLK",
    # Fixed Income
    "AGG", "TLT", "IEF", "HYG", "LQD",
    # Alternatives / Volatility
    "GLD", "TIP", "SVXY"
]

TICKERS = ETF_UNIVERSE
START_DATE = "2011-10-10" # SVXY started didnt exist before 2011
END_DATE = "2026-01-01"
CACHE_PATH = "../data/prices.parquet"

def fetch_prices(force_download=False):
    # SPEED FIX: Use read_parquet (no parse_dates needed, data types are locked in)
    if os.path.exists(CACHE_PATH) and not force_download:
        print(f"Loading data from cache: {CACHE_PATH}")
        prices = pd.read_parquet(CACHE_PATH)
        return prices

    print(f"Downloading {len(TICKERS)} tickers from Yahoo Finance...")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    prices = raw["Close"]
    prices = prices.dropna()

    os.makedirs("../data", exist_ok=True)
    prices.to_parquet(CACHE_PATH)
    print(f"Saved to {CACHE_PATH}")

    return prices


def compute_returns(prices):
    # Compute logarithmic returns, I will refer to this simply as "returns"
    returns = np.log(prices/prices.shift(1)).dropna()

    return returns



def describe_returns(returns):
    summary = returns.describe()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    pct_high_skew = (skewness.abs() > 0.5).mean() * 100
    pct_high_kurtosis = (kurtosis.abs() > 0).mean() * 100   # normal_kurtosis == 0

    skew_kurt = pd.DataFrame([skewness, kurtosis], index=["skewness", "kurtosis"])
    print(summary.T)
    print(pct_high_skew, "% of returns are highly skewed")
    print(pct_high_kurtosis, "% of returns are highly kurtosic")


def jarque_bera(returns):
    print("=== JB Test Summary ===")
    jb_results = []

    for asset in returns.columns:
        jb_stat, jb_p = stats.jarque_bera(returns[asset])
        jb_results.append(jb_p)


    jb_series = pd.Series(jb_results, index=returns.columns)
    pct_rejected = (jb_series < 0.05).mean() * 100

    print(f"{pct_rejected:.1f}% of the ETFs completely reject the assumption of normality (p < 0.05).")



# Quick check

if __name__ == "__main__":

    df = fetch_prices()
    returns = compute_returns(df)

    describe_returns(returns)
    print()
    jarque_bera(returns)




"""
if __name__ == "__main__":
    df = fetch_prices()
    returns = compute_returns(df)

    print("=== returns preview ===")
    print("\nfirst 5 rows:")
    print(returns.head())
    print("\nlast 5 rows:")
    print(returns.tail())
"""


