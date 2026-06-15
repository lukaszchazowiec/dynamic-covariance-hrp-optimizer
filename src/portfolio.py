import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from data_loader import fetch_prices, compute_returns
from dcc_garch import fit_dcc, get_dynamic_covariance, fit_univariate_garch


def get_cluster_var(cov_matrix, cluster_indices):
    """
    Oblicza wariancję jednego zamkniętego klastra (koszyka aktywów).
    """
    # 1. Wycinamy mały fragment macierzy kowariancji tylko dla aktywów z tego klastra
    cov_sub = cov_matrix[np.ix_(cluster_indices, cluster_indices)]

    # 2. Liczymy wagi inverse-variance wewnątrz tego klastra
    inv_vars = 1.0 / np.diag(cov_sub)
    w = inv_vars / np.sum(inv_vars)

    # 3. Mnożenie macierzowe: w^T * Cov * w
    cluster_var = np.dot(w, np.dot(cov_sub, w))
    return cluster_var


def hrp_weights(cov_matrix):
    cov_matrix = np.asarray(cov_matrix)

    # --- TUTAJ BYŁ BRAKUJĄCY BEZPIECZNIK ---
    # Jeśli dostajemy macierz 3D, automatycznie wycinamy ostatni dzień
    if cov_matrix.ndim == 3:
        cov_matrix = cov_matrix[-1]

    # Strażnicy formatu danych
    if cov_matrix.ndim != 2:
        raise ValueError(
            f"cov_matrix must be a 2D matrix or a 3D series. Got shape {cov_matrix.shape}."
        )

    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError(
            f"cov_matrix must be square. Got shape {cov_matrix.shape}."
        )
    # ---------------------------------------
    # STEP 1: Calculating the distance matrix

    # We need volatilities, so we take the square root of the diagonal (because we have a covariance matrix)
    # We only need the diagonal, because that's where the variance is stored
    vols = np.sqrt(np.diag(cov_matrix))

    if np.any(vols <= 0):
        raise ValueError(
            "All assets must have positive variance. "
            "At least one diagonal covariance entry is zero or negative."
        )

    # To get the correlation matrix, we use the standard formula:
    # p(X,Y) = cov(X,Y) / (vol(X) * vol(Y))
    corr_matrix = cov_matrix / np.outer(vols, vols)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    # Now we calculate the distance matrix from the De Prado formula:
    # d(i,j) = (1 - p(i,j)) / 2
    distance_matrix = np.sqrt((1.0-corr_matrix) / 2.0)

    # Wymuszamy idealną symetrię numeryczną: (M + M.T) / 2
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

    np.fill_diagonal(distance_matrix, 0.0)


    # STEP 2: Calculating the Hierarchical Clustering via the Ward linkage

    from scipy.spatial.distance import squareform
    condensed_distance = squareform(distance_matrix)

    linkage_matrix = sch.linkage(condensed_distance, method='ward')


    # STEP 3: Tree ordering and Quasi-Diagonalization

    root_note = sch.to_tree(linkage_matrix, rd=False)

    def get_quasi_diag_order(node):
        if node.is_leaf():
            return [node.get_id()]
        else:
            return get_quasi_diag_order(node.get_left()) + get_quasi_diag_order(node.get_right())

    sort_indices = get_quasi_diag_order(root_note)


    # STEP 4: Recursive Bisection

    weights = pd.Series(1.0, index=sort_indices)
    clusters_queue = [sort_indices]

    while len(clusters_queue) > 0:
        current_cluster = clusters_queue.pop(0)

        if len(current_cluster) <= 1:
            continue


        # 1. Bisection
        midpoint = len(current_cluster) // 2
        left_cluster = current_cluster[:midpoint]
        right_cluster = current_cluster[midpoint:]

        # 2: Risk
        left_cluster_var = get_cluster_var(cov_matrix, left_cluster)
        right_cluster_var = get_cluster_var(cov_matrix, right_cluster)

        # 3: Weights (Alpha)
        alpha = 1.0 - left_cluster_var / (left_cluster_var + right_cluster_var)

        # 4: Update the weights of the current cluster
        weights.loc[left_cluster] *= alpha
        weights.loc[right_cluster] *= (1.0 - alpha)

        # 5: Add the new clusters to the queue
        clusters_queue.append(left_cluster)
        clusters_queue.append(right_cluster)

    return weights.sort_index().values, linkage_matrix


#=====================================================
# BENCHMARKS


# Benchmark 1: EQUAL-WEIGHTS PORTFOLIO

def benchmark_equal_weights(cov_matrix):
    N = cov_matrix.shape[-1] # not using len(), because if we have 3D cov_matrix, we want to use the last dimension
    weights = np.ones(N) / N

    return weights


# Benchmark 2: STATIC HRP PORTFOLIO

def benchmark_static_hrp(returns):
    from sklearn.covariance import LedoitWolf
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf()
    lw.fit(returns)
    shrunk_cov = lw.covariance_

    weights, _ = hrp_weights(shrunk_cov)
    return weights


# Benchmark 3: MINIMUM-VARIANCE PORTFOLIO

def benchmark_min_variance(cov_matrix):
    import scipy.optimize as sco
    cov_matrix = np.asarray(cov_matrix)

    # --- BEZPIECZNIK 3D ---
    if cov_matrix.ndim == 3:
        cov_matrix = cov_matrix[-1]
    # ----------------------

    N = cov_matrix.shape[-1]

    # min(wT * Cov * w)

    function = lambda w, c: np.dot(w.T, np.dot(c, w))

    x0 = np.zeros(N)
    x0[0] = 1.0  # Cały kapitał na starcie w AGG
    x0 = benchmark_equal_weights(cov_matrix)

    constraints = {
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1.0,
    }

    bounds = [(0.0, 1.0) for _ in range(N)]

    res = sco.minimize(function, x0, args=(cov_matrix,), method='SLSQP',
                       constraints=constraints, bounds=bounds,
                       options={'ftol': 1e-9, 'maxiter': 1000})

    return res.x[:N]



# Quick check


if __name__ == "__main__":
    prices = fetch_prices()
    returns = compute_returns(prices)

    std_resid, vols = fit_univariate_garch(returns)
    dcc_results = fit_dcc(std_resid.values)
    cov_history_3D = get_dynamic_covariance(vols, dcc_results)

    last_day_cov = cov_history_3D[-1]

    print("Obliczam wagi portfela HRP (DCC-GARCH)...")
    w_hrp, _ = hrp_weights(last_day_cov)

    print("Obliczam wagi portfela Equal-Weight...")
    w_ew = benchmark_equal_weights(last_day_cov)

    print("Obliczam wagi portfela Static HRP (Ledoit-Wolf)...")
    w_static = benchmark_static_hrp(returns)

    print("Obliczam wagi portfela Minimum-Variance (Markowitz)...")
    w_min_var = benchmark_min_variance(last_day_cov)

    # 5. Tworzenie czytelnej tabeli porównawczej w Pandas
    comparison_df = pd.DataFrame({
        'HRP (DCC)': w_hrp * 100,
        'Equal-Weight': w_ew * 100,
        'Static HRP (LW)': w_static * 100,
        'Min-Variance': w_min_var * 100
    }, index=returns.columns)

    # 6. Wyświetlanie wyników w konsoli
    print("\n" + "=" * 65)
    print("         PORÓWNANIE WAG PORTFELI W PROCENTACH (%)")
    print("=" * 65)
    print(comparison_df.round(2).to_string())
    print("=" * 65)

    # 7. Szybki test sumowania (Kontrola jakości: każda kolumna musi dać 100%)
    print("\nKontrola sumowania kapitału (czy każdy model ulokował 100%?):")
    print(f"  HRP (DCC):      {w_hrp.sum() * 100:.2f}%")
    print(f"  Equal-Weight:   {w_ew.sum() * 100:.2f}%")
    print(f"  Static HRP:     {w_static.sum() * 100:.2f}%")
    print(f"  Min-Variance:   {w_min_var.sum() * 100:.2f}%")
    print("=" * 65)


"""
if __name__ == "__main__":
    prices = fetch_prices()
    returns = compute_returns(prices)

    Z, cond_vol = fit_univariate_garch(returns)
    dcc_result = fit_dcc(Z.to_numpy())

    # 1. Pobieramy pełną historię kowariancji 3D
    cov_history_3D = get_dynamic_covariance(cond_vol, dcc_result)

    # 2. Wrzucamy ją do HRP (nasz bezpiecznik sam wytnie ostatni dzień!)
    final_weights = hrp_weights(cov_history_3D)

    # 3. Profesjonalne testy weryfikacyjne w konsoli
    print("\n" + "=" * 50)
    print("      SUKCES! WAGI PORTFELA HRP WYLICZONE")
    print("=" * 50)

    print("\nOstateczny wektor wag (NumPy Array):")
    print(final_weights)

    # Sprawdzenie sumy (musi być idealnie 1.0, czyli 100%)
    weights_sum = np.sum(final_weights)
    print(f"\nSuma wszystkich wag: {weights_sum:.6f} (Oczekiwano: 1.000000)")

    # Sprawdzenie czy nie ma pozycji ujemnych (HRP nie pozwala na shorty)
    has_negative = np.any(final_weights < 0)
    print(f"Czy w portfelu są pozycje krótkie (ujemne)? {has_negative} (Oczekiwano: False)")

    # Ładny podgląd wag dla poszczególnych indeksów aktywów
    print("\nSzczegółowa alokacja kapitału:")
    for idx, weight in enumerate(final_weights):
        print(f"  Aktywo nr {idx:2d}: {weight * 100:6.2f}%")
    print("=" * 50)
"""