import numpy as np
import pandas as pd

# 1. Importy danych i modeli portfolio
from data_loader import fetch_prices, compute_returns
from portfolio import benchmark_equal_weights, benchmark_static_hrp, benchmark_min_variance, hrp_weights

# 2. DODAJEMY IMPORTY Z TWOJEGO MODUŁU DCC-GARCH
from dcc_garch import fit_univariate_garch, fit_dcc, get_dynamic_covariance


def rolling_backtest(window_size=252, rebal_freq=21):
    prices = fetch_prices()
    returns = compute_returns(prices)

    total_days = len(returns)

    # Cztery listy na codzienne wyniki (dokładamy nasz główny model HRP_DCC)
    hrp_dcc_returns = []
    ew_returns = []
    static_hrp_returns = []
    min_var_returns = []

    # Zmienne na aktualne wagi
    current_w_hrp = None
    current_w_ew = None
    current_w_static = None
    current_w_min_var = None

    print(f"Rozpoczynam pełny backtest Walk-Forward. Łączna liczba dni: {total_days}")

    for t in range(window_size, total_days):

        # OKRESOWY REBALANS (Co 21 dni)
        if (t - window_size) % rebal_freq == 0:
            train_returns = returns.iloc[t - window_size: t]

            # --- SEKCJA DYNAMICZNEJ KOWARIANCJI (DCC-GARCH) ---
            # Odpalamy Twoje funkcje z modułu DCC-GARCH
            std_resid, vols = fit_univariate_garch(train_returns)
            dcc_results = fit_dcc(std_resid.values)
            cov_3d = get_dynamic_covariance(vols, dcc_results)

            # Wyciągamy dynamiczną macierz kowariancji z ostatniego dnia okna treningowego
            last_cov = cov_3d[-1]

            # --- WYLICZANIE WAG DLA WSZYSTKICH MODELI ---

            # Główny model: HRP oparty na dynamicznej kowariancji DCC-GARCH
            # (Ignorujemy macierz powiązań '_', interesują nas tylko wagi)
            current_w_hrp, _ = hrp_weights(last_cov)

            # Benchmark 1: Equal Weight
            current_w_ew = benchmark_equal_weights(train_returns.values)

            # Benchmark 2: Static HRP (oparty na Ledoit-Wolf wewnątrz funkcji)
            current_w_static = benchmark_static_hrp(train_returns)

            # Benchmark 3: Minimum Variance (podmieniamy historical_cov na naszą nową last_cov!)
            current_w_min_var = benchmark_min_variance(last_cov)

            print(f"Backtest at time {t}")

        # CODZIENNE LICZENIE ZYSKÓW / STRAT (Dla 4 portfeli)
        day_returns = returns.iloc[t].values

        port_hrp = np.sum(current_w_hrp * day_returns)
        port_ew = np.sum(current_w_ew * day_returns)
        port_static = np.sum(current_w_static * day_returns)
        port_min_var = np.sum(current_w_min_var * day_returns)

        hrp_dcc_returns.append(port_hrp)
        ew_returns.append(port_ew)
        static_hrp_returns.append(port_static)
        min_var_returns.append(port_min_var)

    return hrp_dcc_returns, ew_returns, static_hrp_returns, min_var_returns


if __name__ == "__main__":
    # 1. Odpalamy backtest i odbieramy komplet wyników
    hrp, ew, static, min_var = rolling_backtest()

    # 2. Dopasowujemy daty (odcinamy pierwsze 252 dni okna treningowego)
    prices = fetch_prices()
    returns = compute_returns(prices)
    backtest_dates = returns.index[252:]

    # 3. Tworzymy tabelę zbiorczą DataFrame
    results_df = pd.DataFrame({
        'HRP_DCC': hrp,
        'Equal_Weight': ew,
        'Static_HRP': static,
        'Min_Variance': min_var
    }, index=backtest_dates)

    # 4. MATEMATYKA PODSUMOWUJĄCA:
    print("\n" + "=" * 60)
    print("        RAPORT KOŃCOWY STRATEGII INWESTYCYJNYCH")
    print("=" * 60)

    # Skumulowana stopa zwrotu
    equity_curves = (1.0 + results_df).cumprod()
    final_wealth = equity_curves.iloc[-1]

    # Roczny zysk (Annualized Return) i Roczna zmienność (Annualized Volatility)
    ann_returns = results_df.mean() * 252
    ann_vols = results_df.std() * np.sqrt(252)
    sharpe_ratios = ann_returns / ann_vols

    # Maksymalne Obsunięcie Kapitału (Max Drawdown)
    # Sprawdzamy, jak głęboko kapitał spadał od swoich szczytów
    running_max = equity_curves.cummax()
    drawdowns = (equity_curves - running_max) / running_max
    max_drawdowns = drawdowns.min()

    # Wyświetlamy piękną tabelę w konsoli
    summary_table = pd.DataFrame({
        'Zysk Skumulowany %': (final_wealth - 1.0) * 100,
        'Zysk Roczny % (Ann)': ann_returns * 100,
        'Zmienność Roczna %': ann_vols * 100,
        'Wskaźnik Sharpe\'a': sharpe_ratios,
        'Max Drawdown %': max_drawdowns * 100
    })

    # --- TA LINIA ROZWIĄZUJE PROBLEM: ---
    # Zmuszamy Pandas, żeby wyświetlał wszystkie kolumny w konsoli bez ucinania
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)  # Rozszerzamy szerokość linii tekstowej
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    print(summary_table)
    print("=" * 60)