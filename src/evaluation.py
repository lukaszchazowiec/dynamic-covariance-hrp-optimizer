import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#plt.style.use('whitegrid')

import yfinance as yf

from data_loader import compute_returns

def sharpe_ratio(returns, rf_series):
    daily_rf = rf_series / 252.0
    excess_ret = returns - daily_rf
    mean_excess = excess_ret.mean()
    std_excess = excess_ret.std()
    if std_excess == 0:
        return 0.0

    sharpe = (mean_excess / std_excess) * np.sqrt(252.0)
    return sharpe


def max_drawdown(cumulative_returns):
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max

    return drawdown.min()


def calmar_ratio(annualized_returns, max_dd):
    abs_drawdown = abs(max_dd)
    if abs_drawdown == 0:
        return np.inf

    calmar = annualized_returns / abs_drawdown
    return calmar


def metrics_table(results_df, rf_series):
    metrics = {}
    years = len(results_df) / 252.0

    for col in results_df.columns:
        rets = results_df[col]
        cum_rets = (1.0 + rets).cumprod()

        # Total Returns, Annualized Returns (CAGR), Annualized Volatility
        total_return = cum_rets.iloc[-1] - 1.0
        annualized_return = (cum_rets.iloc[-1]) ** (1.0/years) - 1.0
        annualized_vol = rets.std() * np.sqrt(252.0)

        # Sharpe Ratio, Max Drawdown, Calmar Ratio
        s_ratio = sharpe_ratio(rets, rf_series)
        m_dd = max_drawdown(cum_rets)
        c_ratio = calmar_ratio(annualized_return, m_dd)

        metrics[col] = {
            'Total Return': total_return,
            'CAGR (Ann Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': s_ratio,
            'Max Drawdown': m_dd,
            'Calmar Ratio': c_ratio
        }

    df = pd.DataFrame(metrics)
    return df


if __name__ == "__main__":
    # 1. Wczytanie danych z backtestu
    results_df = pd.read_csv("backtest_returns.csv", index_col=0, parse_dates=True)
    weights_df = pd.read_csv("hrp_weights_history.csv", index_col=0, parse_dates=True)

    # 2. Szybkie pobranie i dopasowanie T-Bills
    print("Pobieram stopy wolne od ryzyka...")
    tbills = yf.download("^IRX", start=results_df.index[0], end=results_df.index[-1])['Close'].squeeze()
    rf_series = tbills.reindex(results_df.index, method='ffill').bfill().ffill() / 100.0

    # 3. Obliczenie i wyświetlenie tabeli metryk
    summary_df = metrics_table(results_df, rf_series)
    print("\n--- ZBIORCZY RAPORT INWESTYCYJNY ---")
    print(summary_df.round(4))  # Zaokrąglenie do 4 miejsc po przecinku dla czystego widoku

    # 4. Wykres stóp zwrotu (NAV)
    plt.figure(figsize=(10, 5))
    nav_df = (1.0 + results_df).cumprod()
    plt.plot(nav_df)
    plt.legend(nav_df.columns, loc="upper left")
    plt.title("Skumulowane stopy zwrotu (NAV)")
    plt.tight_layout()
    plt.savefig("equity_curves.png")
    plt.close()

    # 5. Wykres wag portfela HRP
    plt.figure(figsize=(10, 5))
    plt.stackplot(weights_df.index, weights_df.values.T, labels=weights_df.columns)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize='x-small')
    plt.title("Ewolucja wag portfela HRP")
    plt.tight_layout()
    plt.savefig("hrp_weights_evolution.png")
    plt.close()

    print("\nRaport wyświetlony. Wykresy zostały zapisane jako pliki .png!")