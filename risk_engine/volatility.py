"""
volatility.py
=============
EWMA: Estimates the volatility of the portfolio and feeds the FHS-VaR model.

Pros
----
- No distribution assumption
- Reactivity to volatility regimes
- One parameter only: lambda = 0.94
- Computationally trivial — a single pass through the data
- Directly comparable to RiskMetrics-based systems at other firms

Cons
----
- Lambda 0.94 is a convention, not an estimate. One should estimate it, or add sensitivity testing using multiple lambdas)
- No mean reversion towards a long-run volatility level compared to a GARCH model.
- Symmetric shocks up and down on the volatility estimate. Strong assumption in the equity market.
- Single decay rate for all timescales.
"""

import numpy as np

def ewma_volatility(returns, lam = 0.94, initial_var = None):
    """
    Exponentially Weighted Moving Average (EWMA) conditional volatility.

    Variance recursion:
        σ²_t = λ · σ²_{t-1} + (1 − λ) · r²_{t-1}

    Parameters
    ----------
    returns     : array-like, shape (T,) : log-return series
    lam         : float, default 0.94 : decay factor, 0 < λ < 1.
                  RiskMetrics daily convention. Higher = smoother, slower.
    initial_var : float or None : seed variance σ²_0.
                  Defaults to the full-sample variance of the series.

    Returns
    -------
    sigma : np.ndarray, shape (T,) : per-day conditional volatility (std dev).

    Notes
    -----
    Seed: the full-sample variance is used as σ²_0. This is a known simplification, it uses data from the full series to initialise the recursion (forward-looking).
    In practice the seed decays to negligible weight (λ^T) within 50-60 days regardless of its value, so for windows of 250 days or more the choice of seed has no material impact.
    λ is not estimated from data, 0.94 is a calibrated convention. Effective memory ≈ 1/(1−λ) observations. At λ=0.94: ~17 days.

    Examples
    --------
    vol = ewma_volatility(r)
    vol.shape
    np.all(vol > 0)
    """

    r = np.asarray(returns, dtype=float)
    T = len(r)

    if not 0 < lam < 1:
        raise ValueError(f"lam must be in (0, 1), got {lam}")

    sigma2    = np.empty(T)
    sigma2[0] = initial_var if initial_var is not None else np.var(r, ddof=1)

    for t in range(1, T):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2

    return np.sqrt(sigma2)