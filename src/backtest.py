import numpy as np
import pandas as pd

# 1. Importy danych i modeli portfolio
from data_loader import fetch_prices, compute_returns
from portfolio import benchmark_equal_weights, benchmark_static_hrp, benchmark_min_variance, hrp_weights

# 2. Importy funkcji z modułu DCC-GARCH
from dcc_garch import fit_univariate_garch, fit_dcc, get_dynamic_covariance


def rolling_backtest(window_size=252, rebal_freq=21):
    prices = fetch_prices()
    returns = compute_returns(prices)

    total_days = len(returns)

    # Listy na codzienne stopy zwrotu strategii
    hrp_dcc_returns = []
    ew_returns = []
    static_hrp_returns = []
    min_var_returns = []

    # NOWOŚĆ: Lista na historię wag głównego modelu HRP (DCC-GARCH)
    hrp_weights_history = []

    # Zmienne na aktualne wagi strategii
    current_w_hrp = None
    current_w_ew = None
    current_w_static = None
    current_w_min_var = None

    print(f"Rozpoczynam pełny backtest Walk-Forward. Łączna liczba dni: {total_days}")

    for t in range(window_size, total_days):

        # OKRESOWY REBALANS (Co rebal_freq dni, np. co miesiąc inwestycyjny - 21 dni)
        if (t - window_size) % rebal_freq == 0:
            train_returns = returns.iloc[t - window_size: t]

            # --- SEKCJA DYNAMICZNEJ KOWARIANCJI (DCC-GARCH) ---
            std_resid, vols = fit_univariate_garch(train_returns)
            dcc_results = fit_dcc(std_resid.values)
            cov_3d = get_dynamic_covariance(vols, dcc_results)

            # Wyciągamy dynamiczną macierz kowariancji z ostatniego dnia okna treningowego
            last_cov = cov_3d[-1]

            # --- WYLICZANIE WAG DLA WSZYSTKICH MODELI ---

            # Główny model: HRP oparty na dynamicznej macierzy kowariancji DCC-GARCH
            current_w_hrp, _ = hrp_weights(last_cov)

            # Benchmark 1: Equal Weight (1/N)
            current_w_ew = benchmark_equal_weights(train_returns.values)

            # Benchmark 2: Static HRP (oparty na skurczonej macierzy Ledoit-Wolf wewnątrz funkcji)
            current_w_static = benchmark_static_hrp(train_returns)

            # Benchmark 3: Minimum Variance (klasyczny Markowitz oparty na last_cov)
            current_w_min_var = benchmark_min_variance(last_cov)

            print(f"Backtest progress: day {t}/{total_days}")

        # CODZIENNE LICZENIE ZYSKÓW / STRAT (Dla wszystkich 4 portfeli)
        day_returns = returns.iloc[t].values

        port_hrp = np.sum(current_w_hrp * day_returns)
        port_ew = np.sum(current_w_ew * day_returns)
        port_static = np.sum(current_w_static * day_returns)
        port_min_var = np.sum(current_w_min_var * day_returns)

        # Zapisujemy dzienne stopy zwrotu
        hrp_dcc_returns.append(port_hrp)
        ew_returns.append(port_ew)
        static_hrp_returns.append(port_static)
        min_var_returns.append(port_min_var)

        # NOWOŚĆ: Zapisujemy wagi z danego dnia do historii alokacji HRP
        hrp_weights_history.append(current_w_hrp)

    return hrp_dcc_returns, ew_returns, static_hrp_returns, min_var_returns, hrp_weights_history


if __name__ == "__main__":
    window_size = 252
    # 1. Odpalamy zmodyfikowany backtest i odbieramy komplet 5 wyników
    hrp, ew, static, min_var, hrp_weights_hist = rolling_backtest()

    # 2. Dopasowujemy daty (odcinamy pierwsze 252 dni stanowiące pierwsze okno treningowe)
    prices = fetch_prices()
    returns = compute_returns(prices)
    backtest_dates = returns.index[window_size:]
    asset_names = returns.columns

    # 3. Tworzymy tabelę zbiorczą stóp zwrotu (DataFrame)
    results_df = pd.DataFrame({
        'HRP_DCC': hrp,
        'Equal_Weight': ew,
        'Static_HRP': static,
        'Min_Variance': min_var
    }, index=backtest_dates)

    # 4. Tworzymy tabelę zbiorczą historii wag dla modelu HRP
    weights_df = pd.DataFrame(hrp_weights_hist, index=backtest_dates, columns=asset_names)

    # 5. Zapisujemy wyniki do plików CSV w folderze projektu (opcjonalne, ale bardzo ułatwia pracę)
    results_df.to_csv("backtest_returns.csv")
    weights_df.to_csv("hrp_weights_history.csv")

    print("\n" + "=" * 60)
    print("Backtest zakończony powodzeniem!")
    print(f"Zapisano stopy zwrotu (wymiary: {results_df.shape}) oraz historię wag (wymiary: {weights_df.shape}).")
    print("Dane są gotowe do wstrzyknięcia do modułu src/evaluation.py")
    print("=" * 60)