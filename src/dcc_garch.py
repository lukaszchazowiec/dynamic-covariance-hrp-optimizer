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

def compute_R_series(std_resid: np.ndarray, Q_bar: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Przelicza szereg czasowy macierzy korelacji R_t dla podanych (już ustalonych)
    parametrów alpha, beta. Wydzielone z fit_dcc, żeby dało się przeliczyć R_t
    także dla parametrów innych niż te znalezione przez optymalizator
    (np. przy fallbacku na parametry z poprzedniego okna w backteście).
    """
    T, N = std_resid.shape
    R_series = np.zeros((T, N, N))
    Q = Q_bar.copy()

    for t in range(T):
        e = std_resid[t]
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        np.clip(R, -0.9999, 0.9999, out=R)
        np.fill_diagonal(R, 1.0)

        R_series[t] = R

        Q = (1 - alpha - beta) * Q_bar \
            + alpha * np.outer(e, e) \
            + beta * Q

    return R_series

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
        'success': czy optymalizacja się zbiegła (bool)
        'n_restarts_tried': ile punktów startowych zostało wypróbowanych
    """
    T, N = std_resid.shape

    # =========================================================================
    # KROK 1: Obliczenie macierzy kotwiczącej Q_bar (Długoterminowa kowariancja)
    # =========================================================================
    Q_bar = std_resid.T @ std_resid / T  # Kształt: (N, N)

    # =========================================================================
    # KROK 2: Definicja funkcji celu dla optymalizatora (Negative Log-Likelihood)
    # =========================================================================
    def _dcc_loglik(params):
        alpha, beta = params

        if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
            return 1e10

        Q = Q_bar.copy()
        ll = 0.0

        for t in range(T):
            e = std_resid[t]

            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)

            np.clip(R, -0.9999, 0.9999, out=R)
            np.fill_diagonal(R, 1.0)

            try:
                sign, log_det = np.linalg.slogdet(R)
                if sign <= 0:
                    return 1e10
                R_inv = np.linalg.inv(R)
            except np.linalg.LinAlgError:
                # Macierz osobliwa / nieodwracalna -> kara zamiast crasha
                return 1e10

            ll -= 0.5 * (log_det + e @ R_inv @ e - e @ e)

            Q = (1 - alpha - beta) * Q_bar \
                + alpha * np.outer(e, e) \
                + beta * Q

        if not np.isfinite(ll):
            return 1e10

        return -ll

    # =========================================================================
    # KROK 3: Numeryczna optymalizacja z wieloma punktami startowymi (multi-start)
    # =========================================================================
    # DECYZJA: Nelder-Mead bywa niestabilny i potrafi utknąć w złym rozwiązaniu
    # zależnie od punktu startowego. Próbujemy kilku różnych punktów startowych
    # i wybieramy ten, który dał najniższą wartość funkcji celu (najlepsze dopasowanie).
    starting_points = [
        [0.05, 0.90],
        [0.01, 0.95],
        [0.10, 0.80],
        [0.03, 0.70],
        [0.15, 0.60],
    ]

    best_result = None

    for x0 in starting_points:
        result = minimize(
            _dcc_loglik,
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'maxfev': 500, 'xatol': 1e-4, 'fatol': 1e-4}
        )

        # Odrzucamy wyniki, które utknęły w obszarze kary (brak sensownego dopasowania)
        if not np.isfinite(result.fun) or result.fun >= 1e9:
            continue

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    # Jeśli WSZYSTKIE starty zawiodły, spróbujmy jeszcze raz z domyślnym punktem
    # i przyjmijmy go jako wynik z jawnym oznaczeniem niepowodzenia, zamiast
    # ciszej awarii.
    if best_result is None:
        best_result = minimize(
            _dcc_loglik,
            x0=[0.05, 0.90],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'maxfev': 1000, 'xatol': 1e-4, 'fatol': 1e-4}
        )

    alpha_hat, beta_hat = best_result.x
    converged = bool(best_result.success) and (best_result.fun < 1e9)

    if not converged:
        print(f"Warning: DCC optimization did not converge cleanly. "
              f"alpha={alpha_hat:.4f}, beta={beta_hat:.4f}, "
              f"message={getattr(best_result, 'message', 'n/a')}")

    # =========================================================================
    # KROK 4: Odtworzenie pełnego szeregu czasowego macierzy R_t
    # =========================================================================
    R_series = compute_R_series(std_resid, Q_bar, alpha_hat, beta_hat)

    return {
        'alpha': alpha_hat,
        'beta': beta_hat,
        'Q_bar': Q_bar,
        'R': R_series,
        'loglik': -best_result.fun,
        'success': converged,
        'n_restarts_tried': len(starting_points),
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