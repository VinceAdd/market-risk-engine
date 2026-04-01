"""
rolling_var.py
==============
Rolling out-of-sample VaR and CVaR forecast series for backtesting.

Each forecast at time t uses only data up to t-1 to avoid forward-looking bias.
The first `window` entries are NaN (insufficient history to estimate).

Two methods — consistent across both rolling_var and rolling_cvar
-----------------------------------------------------------------
historical : empirical quantile
fhs        : EWMA-filtered historical simulation
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .risk_models import var_historical, var_fhs, cvar, cvar_fhs
from .volatility  import ewma_volatility

def rolling_var(returns: ArrayLike, window: int = 250, confidence: float = 0.99,
                method: Literal["historical", "fhs"] = "historical", ) -> np.ndarray:
    """
    Rolling out-of-sample VaR forecast series.

    Parameters
    ----------
    returns    : array-like, shape (T,)
    window     : int, default 250 — lookback period in days
    confidence : float, default 0.99
    method     : "historical" or "fhs"
                 historical — empirical quantile, no distribution assumption
                 fhs        — EWMA-filtered, regime-aware (see var_fhs)

    Returns
    -------
    var_series : np.ndarray, shape (T,)
        var_series[t] = VaR forecast made at t-1 using returns[t-window:t].
        First `window` entries are NaN.

    Examples
    --------
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    v = rolling_var(r, window=100)
    np.isnan(v[:100]).all()
    np.all(v[100:] > 0)
    """

    r = np.asarray(returns, dtype=float)
    T = len(r)

    if window >= T:
        raise ValueError(f"window ({window}) must be less than series length ({T})")
    if method not in ("historical", "fhs"):
        raise ValueError(f"method must be 'historical' or 'fhs', got '{method}'")

    var_series = np.full(T, np.nan)

    for t in range(window, T):
        w = r[t - window : t]
        try:
            if method == "historical":
                var_series[t] = var_historical(w, confidence)
            else:                                        # fhs
                var_series[t] = var_fhs(w, ewma_volatility(w), confidence)
        except Exception:
            pass
    return var_series


def rolling_cvar(returns: ArrayLike, window: int = 250, confidence: float = 0.975,
                 method: Literal["historical", "fhs"] = "historical", ) -> np.ndarray:
    """
    Rolling out-of-sample CVaR forecast series.

    Mirrors rolling_var exactly — same window, same method options,
    same out-of-sample discipline. Pair with rolling_var for backtesting:
    use the same method for both so VaR and CVaR come from the same model.

    Parameters
    ----------
    returns    : array-like, shape (T,)
    window     : int, default 250 — lookback period in days
    confidence : float, default 0.99
    method     : "historical" or "fhs"
                 historical — empirical tail mean (see cvar)
                 fhs        — EWMA-filtered tail mean (see cvar_fhs)

    Returns
    -------
    cvar_series : np.ndarray, shape (T,)
        cvar_series[t] = CVaR forecast made at t-1 using returns[t-window:t].
        First `window` entries are NaN.

    Examples
    --------
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    cv = rolling_cvar(r, window=100)
    np.isnan(cv[:100]).all()
    np.all(cv[100:] > 0)
    """

    r = np.asarray(returns, dtype=float)
    T = len(r)

    if window >= T:
        raise ValueError(f"window ({window}) must be less than series length ({T})")
    if method not in ("historical", "fhs"):
        raise ValueError(f"method must be 'historical' or 'fhs', got '{method}'")

    cvar_series = np.full(T, np.nan)

    for t in range(window, T):
        w = r[t - window : t]
        try:
            if method == "historical":
                cvar_series[t] = cvar(w, confidence)
            else:                                        # fhs
                cvar_series[t] = cvar_fhs(w, ewma_volatility(w), confidence)
        except Exception:
            pass
    return cvar_series
