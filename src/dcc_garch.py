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

        # 1. Sprawdzamy aktualną wariancję tego konkretnego aktywa w oknie
        current_variance = np.var(y)

        # 2. Automatycznie dobieramy mnożnik tak, aby wariancja była bliska 1.0
        #    (Biblioteka arch kocha wariancję w tym przedziale)
        if current_variance > 0:
            rescale_factor = 1.0 / np.sqrt(current_variance)
        else:
            rescale_factor = 1.0

        # Skalujemy dane wejściowe
        y_scaled = y * rescale_factor

        # 3. Dopasowujemy model na idealnie wyskalowanych danych
        model = arch_model(y_scaled, p=1, q=1, dist='t')
        # Wyłączamy pokazywanie ostrzeżeń, bo nasza skala jest już idealna
        res = model.fit(disp="off", show_warning=False)

        # 4. Wyciągamy wyniki i wracamy do oryginalnej skali backtestu
        # Reszty standaryzowane (std_resid) są bezwymiarowe, więc skala sama się skraca
        std_resid_df[asset] = pd.Series(res.std_resid, index=y.index)

        # Wolatylność (vols) MUSI zostać podzielona przez nasz mnożnik,
        # aby wróciła do oryginalnego rzędu wielkości w Twoim backteście!
        vols_df[asset] = pd.Series(res.conditional_volatility / rescale_factor, index=y.index)

    std_resid_df = std_resid_df.dropna()
    vols_df = vols_df.dropna()

    return std_resid_df, vols_df




def fit_dcc(std_resid: np.ndarray) -> dict:
    """
    Fit scalar DCC(1,1) model via quasi-MLE.

    Parameters
    ----------
    std_resid : np.ndarray, shape (T, N)
        Standardized residuals from per-asset GARCH fits.
        Each column should have mean ≈ 0 and variance ≈ 1.

    Returns
    -------
    dict with keys:
        'alpha'  : scalar DCC alpha
        'beta'   : scalar DCC beta
        'Q_bar'  : (N, N) unconditional covariance of std_resid
        'R'      : (T, N, N) time-varying correlation matrices
        'loglik' : maximized log-likelihood value
    """
    T, N = std_resid.shape

    # Step 1: unconditional covariance (Q-bar)
    Q_bar = std_resid.T @ std_resid / T   # (N, N)

    def _dcc_loglik(params):
        alpha, beta = params

        # Stationarity constraint enforced via penalty
        if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
            return 1e10

        Q = Q_bar.copy()          # initialise Q at unconditional value
        ll = 0.0                  # log-likelihood

        for t in range(T):
            e = std_resid[t]      # (N,) vector

            # Correlation matrix from Qt
            d = np.sqrt(np.diag(Q))          # (N,)
            R = Q / np.outer(d, d)            # (N, N)

            # Clamp for numerical safety
            np.clip(R, -0.9999, 0.9999, out=R)
            np.fill_diagonal(R, 1.0)

            # Quasi log-likelihood contribution:
            # ll += -0.5 * (log|R| + e' R^{-1} e)
            sign, log_det = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e10

            R_inv = np.linalg.inv(R)
            ll -= 0.5 * (log_det + e @ R_inv @ e)

            # Recurse Q
            Q = (1 - alpha - beta) * Q_bar \
                + alpha * np.outer(e, e) \
                + beta * Q

        return -ll   # we minimise, so return negative

    # Step 2: optimise over (alpha, beta)
    result = minimize(
        _dcc_loglik,
        x0=[0.05, 0.90],           # sensible starting point
        method='Nelder-Mead',
        options={'maxiter': 500, 'maxfev': 500, 'xatol': 1e-4, 'fatol': 1e-4}
    )

    alpha_hat, beta_hat = result.x

    # Step 3: recover the full R series at optimal params
    R_series = np.zeros((T, N, N))
    Q = Q_bar.copy()

    for t in range(T):
        e = std_resid[t]
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        np.clip(R, -0.9999, 0.9999, out=R)
        np.fill_diagonal(R, 1.0)
        R_series[t] = R
        Q = (1 - alpha_hat - beta_hat) * Q_bar \
            + alpha_hat * np.outer(e, e) \
            + beta_hat * Q

    return {
        'alpha':  alpha_hat,
        'beta':   beta_hat,
        'Q_bar':  Q_bar,
        'R':      R_series,
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



"""
if __name__ == "__main__":
    prices = fetch_prices()
    returns = compute_returns(prices)
    Z = fit_univariate_garch(returns)
    dcc_result = fit_dcc(Z.to_numpy())  # <-- add .to_numpy()
    print(f"α={dcc_result['alpha']:.4f}  β={dcc_result['beta']:.4f}")
    print(f"α+β={dcc_result['alpha'] + dcc_result['beta']:.4f}  (must be <1)")
    print(f"R[0] diagonal: {np.diag(dcc_result['R'][0])}")  # should be all 1s
 """



if __name__ == "__main__":
    prices  = fetch_prices()
    returns = compute_returns(prices)

    Z, cond_vol = fit_univariate_garch(returns)
    dcc_result = fit_dcc(Z.to_numpy())
    Sigma = get_dynamic_covariance(cond_vol, dcc_result)

    print(f"Sigma shape: {Sigma.shape}")
    print(f"Sigma[0] diagonal: {np.diag(Sigma[0])}")
    print(f"Sigma symmetric: {np.allclose(Sigma[0], Sigma[0].T)}")

"""
if __name__ == "__main__":
    from data_loader import fetch_prices, compute_returns

    # Moduł 1: Pobieranie i zwroty
    prices = fetch_prices()
    returns = compute_returns(prices)

    # Moduł 2, Krok 1: GARCH (wyciągamy rezidua I zmienności!)
    print("Dopasowywanie modeli GARCH...")
    std_resid, vols = fit_univariate_garch(returns)

    # Moduł 2, Krok 2: DCC (karminy go reziduami w postaci tablicy NumPy)
    print("Dopasowywanie modelu DCC...")
    dcc_results = fit_dcc(std_resid.values)

    # Moduł 2, Krok 3: Połączenie wszystkiego w Sigmę!
    print("Obliczanie dynamicznej kowariancji...")
    final_covariance = get_dynamic_covariance(vols, dcc_results)

    # Szybki test poprawności:
    print("\nSukces!")
    print(f"Kształt końcowej tablicy kowariancji: {final_covariance.shape}")
    # Powinno wypisać np. (3600, 16, 16) - czyli dla każdego dnia masz osobną macierz ryzyka!
"""