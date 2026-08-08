import numpy as np
import pandas as pd
from data_loader import fetch_prices, compute_returns
from arch import arch_model
from scipy.optimize import minimize


def fit_univariate_garch(returns):
    std_resid_df = pd.DataFrame(index=returns.index)
    vols_df = pd.DataFrame(index=returns.index)

    for asset in returns.columns:
        y = returns[asset].dropna()

        # 1. Checking the current variance in the return window
        current_variance = np.var(y)

        # 2. The 'arch' library uses optimizers that struggle with very small values, so I rescale the variance to be close to 1.0
        if current_variance > 0:
            rescale_factor = 1.0 / np.sqrt(current_variance)
        else:
            rescale_factor = 1.0

        # Scaled entry data
        y_scaled = y * rescale_factor

        # 3. Fitting the model on scaled data (GARCH(1,1))
        model = arch_model(y_scaled, p=1, q=1, dist='t')
        res = model.fit(disp="off", show_warning=False)

        # 4. Getting the results and returning to the original scale
        std_resid_df[asset] = pd.Series(res.std_resid, index=y.index)
        vols_df[asset] = pd.Series(res.conditional_volatility / rescale_factor, index=y.index)

    std_resid_df = std_resid_df.dropna()
    vols_df = vols_df.dropna()

    return std_resid_df, vols_df


def fit_dcc(std_resid: np.ndarray) -> dict:
    """
    Dwuetapowa estymacja skalarnego modelu DCC(1,1) metodą Quasi-Maximum Likelihood (QMLE).

    Parameters
    ----------
    std_resid : np.ndarray, kształt (T, N)
        Ustandaryzowane reszty u_t = e_t / sqrt(h_t) z jednowymiarowych modeli GARCH(1,1).
        Każda kolumna ma średnią ≈ 0 i wariancję ≈ 1.

    Returns
    -------
    dict z kluczami:
        'alpha'  : estymator rekonstrukcji szoku (dławik reakcji korelacji na nowe szoki)
        'beta'   : estymator pamięci (bezwładność historycznej korelacji)
        'Q_bar'  : macierz bezwarunkowej kowariancji reszt ustandaryzowanych (punkt kotwiczenia)
        'R'      : szereg czasowy macierzy korelacji warunkowych o wymiarze (T, N, N)
        'loglik' : maksymalizowana wartość funkcji log-wiarygodności (Log-Likelihood)
    """
    T, N = std_resid.shape

    # =========================================================================
    # KROK 1: Obliczenie macierzy kotwiczącej Q_bar (Długoterminowa kowariancja)
    # =========================================================================
    # Q_bar reprezentuje długoterminową średnią macierz pseudokowariancji reszt.
    # W warunkach spokoju rynkowego korelacje powracają do tej wartości (mean-reversion).
    # Iloczyn (N, T) @ (T, N) z podziałem przez T daje estymator bezwarunkowej macierzy (N, N).
    Q_bar = std_resid.T @ std_resid / T  # Kształt: (N, N)

    # =========================================================================
    # KROK 2: Definicja funkcji celu dla optymalizatora (Negative Log-Likelihood)
    # =========================================================================
    def _dcc_loglik(params):
        alpha, beta = params

        # DECYZJA 1: Warunek stacjonarności i dodatniej określoności.
        # W finansach suma alpha + beta MUSI być mniejsza od 1, aby proces nie eksplodował.
        # Parametry muszą być dodatnie. Jeśli optimizer przetestuje zabroniony obszar,
        # zwracamy sztucznie dużą karę (1e10), zmuszając go do wycofania się.
        if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
            return 1e10

        Q = Q_bar.copy()  # Inicjalizacja macierzy Q_0 wartością bezwarunkową Q_bar
        ll = 0.0  # Zmienna kumulująca wartość funkcji log-wiarygodności

        # Pętla krocząca po historii giełdowej (dzień po dniu)
        for t in range(T):
            e = std_resid[t]  # Wektor ustandaryzowanych reszt w dniu t (wymiar N,)

            # DECYZJA 2: Przekształcenie pseudokowariancji Q_t w czystą macierz korelacji R_t.
            # Ponieważ elementy na przekątnej Q_t nie są idealnie równe 1.0, skalujemy Q_t
            # przez iloczyn odchyleń standardowych z jej diagonali: R = Q / (d * d^T).
            d = np.sqrt(np.diag(Q))  # Odchylenia standardowe (wymiar N,)
            R = Q / np.outer(d, d)  # Skalowanie macierzowe -> macierz korelacji (N, N)

            # DECYZJA 3: Zabezpieczenie numeryczne przed osobliwością (Invertibility).
            # Zapobiegamy sytuacji, w której korelacja osiagnie dokładnie 1.0 lub -1.0,
            # co uniemożliwiłoby odwrócenie macierzy R (awaria linalg.inv).
            np.clip(R, -0.9999, 0.9999, out=R)
            np.fill_diagonal(R, 1.0)  # Gwarancja idealnych jedynek na przekątnej

            # DECYZJA 4: Liczenie wyznacznika odporne na błędy precyzji (Underflow).
            # Wyznacznik macierzy korelacji bywa bliski 0. Metoda slogdet wyciąga od razu
            # logarytm wyznacznika (log_det), unikając błędów zaokrągleń float64.
            sign, log_det = np.linalg.slogdet(R)
            if sign <= 0:  # Macierz MUSI być dodatnio określona
                return 1e10

            # DECYZJA 5: Obliczenie dziennego wkładu do Log-Likelihood.
            # Wzór: -0.5 * [ ln|R_t| + e_t^T * R_t^{-1} * e_t - e_t^T * e_t ]
            # Wyrażenie (e @ R_inv @ e) to odległość Mahalanobisa — kara za błąd prognozy korelacji.
            # Człon (e @ e) to stała normalizująca względem modelu bazowego (IID).
            # Odejmujemy ten wkład, budując łączną wartość log-wiarygodności.
            R_inv = np.linalg.inv(R)
            ll -= 0.5 * (log_det + e @ R_inv @ e - e @ e)

            # DECYZJA 6: Aktualizacja rekurencyjna Q_t na kolejny dzień (DCC Recursion).
            # Nowa wartość Q to ważona kombinacja trzech elementów:
            # 1. Średniej długoterminowej: (1 - alpha - beta) * Q_bar
            # 2. Wczorajszego szoku rynkowego: alpha * (e * e^T) [iloczyn zewnętrzny]
            # 3. Pamięci z poprzedniego dnia: beta * Q_{t-1}
            Q = (1 - alpha - beta) * Q_bar \
                + alpha * np.outer(e, e) \
                + beta * Q

        # DECYZJA 7: Zmiana znaku (Konwersja problemu).
        # Chcemy MAKSYMALIZOWAĆ wiarygodność (ll). Ponieważ scipy.optimize.minimize
        # potrafi wyłącznie MINIMALIZOWAĆ funkcje, zwracamy ujemne Log-Likelihood (-ll).
        return -ll

        # =========================================================================

    # KROK 3: Numeryczna optymalizacja parametrów (alpha, beta)
    # =========================================================================
    # DECYZJA 8: Wybór algorytmu Nelder-Mead i punktu startowego.
    # Metoda Nelder-Mead jest bezgradientowa i bardzo stabilna dla problemów z ograniczeniami.
    # Punkt x0=[0.05, 0.90] reprezentuje typowe stany rynkowe (mały szok, wysoka pamięć).
    result = minimize(
        _dcc_loglik,
        x0=[0.05, 0.90],
        method='Nelder-Mead',
        options={'maxiter': 500, 'maxfev': 500, 'xatol': 1e-4, 'fatol': 1e-4}
    )

    if not result.success:
        print("Uwaga: optymalizacja DCC się nie zbiegła:", result.message)

    alpha_hat, beta_hat = result.x

    # =========================================================================
    # KROK 4: Odtworzenie pełnego szeregu czasowego macierzy R_t dla zoptymalizowanych parametrów
    # =========================================================================
    # Mając już optymalne parametry (alpha_hat, beta_hat), przechodzimy po pętli po raz ostatni,
    # aby zapisać i zwrócić macierze R_t dla każdego dnia. Te macierze zasilą bezpośrednio HRP.
    R_series = np.zeros((T, N, N))
    Q = Q_bar.copy()

    for t in range(T):
        e = std_resid[t]
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        np.clip(R, -0.9999, 0.9999, out=R)
        np.fill_diagonal(R, 1.0)

        R_series[t] = R  # Zapisujemy wyliczoną macierz korelacji dla dnia t

        Q = (1 - alpha_hat - beta_hat) * Q_bar \
            + alpha_hat * np.outer(e, e) \
            + beta_hat * Q

    # Zwracamy slownik ze wszystkimi wynikami.
    # Odwracamy znak result.fun (-result.fun), aby przywrócić pierwotną, dodatnią/prawdziwą wartość Log-Likelihood.
    return {
        'alpha': alpha_hat,
        'beta': beta_hat,
        'Q_bar': Q_bar,
        'R': R_series,
        'loglik': -result.fun,
    }


def get_dynamic_covariance(vols_df, fit_dcc_results):
    # dynamic_cor_matrix (sigma) = D * R * D

    T, N = vols_df.shape
    sigma_series = np.zeros((T, N, N))
    vols_array = vols_df.to_numpy()

    for t in range(T):
        R_t = fit_dcc_results['R'][t]
        vols_t = vols_array[t]
        D_t = np.diag(vols_t)

        sigma_series[t] = D_t @ R_t @ D_t


    return sigma_series


if __name__ == "__main__":
    prices  = fetch_prices()
    returns = compute_returns(prices)

    Z, cond_vol = fit_univariate_garch(returns)
    dcc_result = fit_dcc(Z.to_numpy())
    Sigma = get_dynamic_covariance(cond_vol, dcc_result)

    print(f"Sigma shape: {Sigma.shape}")
    print(f"Sigma[0] diagonal: {np.diag(Sigma[0])}")
    print(f"Sigma symmetric: {np.allclose(Sigma[0], Sigma[0].T)}")