import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import scipy.cluster.hierarchy as sch

from data_loader import fetch_prices, compute_returns
from dcc_garch import fit_dcc, get_dynamic_covariance, fit_univariate_garch
from portfolio import hrp_weights


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


def plot_dynamic_correlations(dcc_results, returns_df, asset_i, asset_j):

    asset_i_idx = returns_df.columns.get_loc(asset_i)
    asset_j_idx = returns_df.columns.get_loc(asset_j)

    pairwise_corr = dcc_results['R'][:, asset_i_idx, asset_j_idx]

    plt.figure(figsize=(12, 6))
    plt.title(f"Dynamic correlations between {asset_i} and {asset_j}")
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    plt.plot(returns_df.index, pairwise_corr, label="DCC-GARCH Correlation")
    plt.axvspan('2020-02-15', '2020-04-30', color='red', alpha=0.15, label='COVID-19 Market Panic')

    plt.legend(loc="lower left")

    plt.show()




def plot_dendogram(linkage_matrix, asset_names=None):

    plt.figure(figsize=(12, 6))
    plt.title("Hierarchical Clustering Dendogram")
    plt.xlabel("ETF's")
    plt.ylabel("Hierarchical Distance (Ward)")
    plt.grid(True)
    plt.tight_layout()

    sch.dendrogram(linkage_matrix, labels=asset_names, orientation="top", leaf_rotation=45, leaf_font_size=8)

    plt.show()




# Quick check

if __name__ == "__main__":

    prices = fetch_prices()
    returns = compute_returns(prices)

    plot_paired_correlation(returns)
    plot_rolling_correlation(returns)

    std_resid, vols = fit_univariate_garch(returns)

    dcc_results = fit_dcc(std_resid.values)

    plot_dynamic_correlations(dcc_results, returns, 'SPY', 'TLT')

    # Plot the dendogram

    # 1. Pobieramy pełną historię kowariancji 3D
    cov_history_3D = get_dynamic_covariance(vols, dcc_results)

    # 2. Wyciągamy macierz dla ostatniego dnia (tak jak robiliśmy to wcześniej)
    single_day_cov = cov_history_3D[-1, :, :]

    # 3. Rozpakowujemy dwie rzeczy z funkcji hrp_weights
    final_weights, linkage_matrix = hrp_weights(single_day_cov)

    # 4. Rysujemy dendrogram
    plot_dendogram(linkage_matrix, asset_names=list(returns.columns))