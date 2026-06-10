import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from data_loader import fetch_prices, compute_returns



def plot_paired_correlation(returns):
    plt.close("all")
    correlation = returns.corr()
    sns.heatmap(correlation, annot=False, cmap="coolwarm", vmin=-1, vmax=1)

    plt.title("Correlation between stock returns")
    plt.tight_layout()
    plt.show()


def plot_rolling_correlation(returns, window=60):
    plt.close("all")
    avg_corr = []

    for i in range(window, len(returns)):
        corr_matrix = returns.iloc[i-window:i].corr()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        avg_corr.append(upper.stack().mean())


    plt.figure(figsize=(12, 6))
    plt.title("Rolling average correlation between stock returns")
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.grid(True)
    plt.plot(avg_corr)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.show()


# Quick check

if __name__ == "__main__":

    prices = fetch_prices()
    returns = compute_returns(prices)

    plot_paired_correlation(returns)
    plot_rolling_correlation(returns)