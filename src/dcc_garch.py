import numpy as np
import pandas as pd
from data_loader import fetch_prices, compute_returns
from arch import arch_model
from scipy.optimize import minimize


def fit_univariate_garch(returns):
    std_resid_df = pd.DataFrame(index=returns.index)

    for asset in returns.columns:
        model = arch_model(returns[asset]*1000, p=1, q=1, dist='t')
        res = model.fit(disp="off")

        std_resid_df[asset] = pd.Series(res.std_resid, index=returns.index[-len(res.std_resid):])

    std_resid_df = std_resid_df.dropna()

    return std_resid_df




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
        ll = 0.0

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
        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
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



if __name__ == "__main__":
    prices = fetch_prices()
    returns = compute_returns(prices)
    Z = fit_univariate_garch(returns)
    dcc_result = fit_dcc(Z.to_numpy())  # <-- add .to_numpy()
    print(f"α={dcc_result['alpha']:.4f}  β={dcc_result['beta']:.4f}")
    print(f"α+β={dcc_result['alpha'] + dcc_result['beta']:.4f}  (must be <1)")
    print(f"R[0] diagonal: {np.diag(dcc_result['R'][0])}")  # should be all 1s