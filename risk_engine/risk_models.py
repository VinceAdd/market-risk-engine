"""
risk_models.py
==============
Risk measures: Historical VaR, FHS-VaR, CVaR, Covariance VaR (for VaR allocation).

--------------
var_historical : primary risk number, no distribution assumption, limited to the actual empirical tail of portfolio returns.
var_fhs : regime-aware VaR, standardises by EWMA volatility so current vol conditions are reflected in the estimate. It combines EWMA (volatility) with empirical tail (shape).
cvar : average loss over a defined quantile. It is considered a coherent risk measure (Basel IV / FRTB mandate).
A coherent risk measure satisfies the 4 properties:
    - monotonicity: if one portfolio always results in worse outcomes than another, its risk should be assessed as higher.
    - subadditivity: the risk of two portfolios together cannot get any worse than adding the two risks separately: this is the diversification principle
    - positive homogeneity: scaling a portfolio by a positive factor should scale its risk by the same factor.
    - translation invariance: adding a risk-free asset to a portfolio should reduce the portfolio's risk by the amount of the risk-free asset.
var_parametric_cov : useful for risk driver decomposition at security-level, through the covariance matrix. Not the primary risk measure.
"""

from __future__ import annotations # used for Type hints (recent python versions)

import numpy as np
from scipy import stats
from numpy.typing import ArrayLike


def var_historical(returns: ArrayLike, confidence: float = 0.99, ) -> float:
    """
    Historical Simulation (HS) Value at Risk.
    The x% empirical quantile of the return distribution.

    Parameters
    ----------
    returns : array-like, shape (T,) : portfolio log-returns
    confidence : float, default 0.99

    Returns
    -------
    var : float, positive VaR estimate

    Examples
    --------
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    var_historical(r) > 0
    # Closed-form: HS VaR = -Q_{1-confidence}(returns)
    abs(var_historical(r) - (-np.quantile(r, 0.01))) < 1e-10
    """

    r = np.asarray(returns, dtype=float)
    if len(r) < 30:
        raise ValueError(f"Need at least 30 observations, got {len(r)}")
    return float(-np.quantile(r, 1 - confidence))


def var_fhs(returns: ArrayLike, sigma: ArrayLike, confidence: float = 0.99, ) -> float:
    """
    Filtered Historical Simulation (FHS) VaR.

    Standardises returns by EWMA volatility, takes the empirical quantile of the residuals, then rescales by current (last) volatility estimate (sigma).

    Steps
    -----
    1. z_t = r_t / sigma_t                standardise by conditional vol
    2. z_q = Q_{1-confidence}(z_1..z_T)   empirical quantile of residuals
    3. VaR  = -z_q * sigma_T              rescale by current vol

    This considers the current volatility regime into the historical (distribution-free) tail estimate. Increases responsiveness to volalitity.

    Parameters
    ----------
    returns : array-like, shape (T,), portfolio log-returns
    sigma : array-like, shape (T,), EWMA conditional volatility
    confidence : float, default 0.99

    Returns
    -------
    var : float, positive FHS VaR estimate

    Examples
    --------
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    sigma = ewma_volatility(r)
    var_fhs(r, sigma) > 0
    """

    r = np.asarray(returns, dtype=float)
    s = np.asarray(sigma,   dtype=float)

    if r.shape != s.shape:
        raise ValueError("returns and sigma must have the same shape")
    if np.any(s <= 0):
        raise ValueError("sigma must be strictly positive")

    z_t = r / s                                  # standardised residuals
    z_q = np.quantile(z_t, 1 - confidence)      # empirical quantile
    return float(-z_q * s[-1])                  # rescale by current vol


def cvar_fhs(returns: ArrayLike, sigma: ArrayLike, confidence: float = 0.975, ) -> float:
    """
    Filtered Historical Simulation CVaR (Expected Shortfall).

    Similar to var_fhs with the same standardisation, volatility scaling, tail mean instead of tail quantile. Regime-aware: reflects current vol via s[-1].

    Steps
    -----
    1. z_t  = r_t / s_t                   standardise by conditional vol
    2. z_q  = Q_{1-confidence}(z_1..z_T)        VaR threshold in residual space
    3. tail = z_t[z_t < z_q]              residuals beyond the threshold
    4. CVaR = -mean(tail) * s_T           rescale by current vol

    Parameters
    ----------
    returns    : array-like, shape (T,), portfolio log-returns
    sigma      : array-like, shape (T,), EWMA conditional volatility
    confidence : float, default 0.99

    Returns
    -------
    cvar_val : float, positive FHS-CVaR estimate

    Examples
    --------
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    sig = ewma_volatility(r)
    cvar_fhs(r, sig) >= var_fhs(r, sig)
    """
    r = np.asarray(returns, dtype=float)
    s = np.asarray(sigma,   dtype=float)

    if r.shape != s.shape:
        raise ValueError("returns and sigma must have the same shape")
    if np.any(s <= 0):
        raise ValueError("sigma must be strictly positive")

    z_t = r / s                              # standardised residuals
    z_q = np.quantile(z_t, 1 - confidence)   # VaR threshold in residual space
    tail = z_t[z_t < z_q]                     # tail of standardised residuals

    if len(tail) == 0:
        raise ValueError("No observations in the tail. Reduce confidence or increase sample size.")

    return float(-tail.mean() * s[-1])          # rescale by current vol


def cvar(returns: ArrayLike, confidence: float = 0.975, ) -> float:
    """
    Conditional VaR (CVaR): Mean loss where the loss exceeds the VaR threshold.
    Mandated under Basel IV / FRTB at the 97.5% confidence level.

    Parameters
    ----------
    returns : array-like, shape (T,), portfolio log-returns
    confidence : float, default 0.99

    Returns
    -------
    cvar_val : float, positive CVaR estimate

    Examples
    --------
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    cvar(r) >= var_historical(r)
    """
    r = np.asarray(returns, dtype=float)
    threshold = np.quantile(r, 1 - confidence)
    tail = r[r < threshold]

    if len(tail) == 0:
        raise ValueError("No observations in the tail. Reduce confidence or increase sample size.")

    return float(-tail.mean())


def _ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix."

    Replaces the noisy sample covariance with a weighted average between
    the sample and the most stable possible structure: every asset has the
    same variance, no correlations. The weight (delta) is chosen analytically
    to minimise estimation error given the sample size.

        Sigma* = (1 - delta) * S  +  delta * mu * I

    mu = np.trace(S) / N  is the mean asset variance, it scales the identity
    so the target matches the average variance in the data.

    Example:
        S = [[0.0004, 0.0002],    np.trace(S) = 0.0004 + 0.0009 = 0.0013
             [0.0002, 0.0009]]    mu = 0.0013 / 2 = 0.00065
                                  target = [[0.00065, 0.0],
                                            [0.0,     0.00065]]

    Delta — the analytical formula
    --------------------------------
    Ledoit and Wolf derived delta by minimising the expected squared error
    between Sigma* and the true (unknown) covariance. The result:

              ((T-2)/T) * ||S||²_F  +  tr(S)²
    delta* = ────────────────────────────────────
              (T+2) * ( ||S||²_F  -  tr(S)²/N )

    where:
      ||S||²_F = np.trace(S @ S), squared Frobenius norm (sum of squared elements)
      tr(S) = np.trace(S), sum of diagonal variances
      T = observations,
      N = assets

    T and N effects:
      More observations (larger T) > larger denominator > smaller delta > trust S more.
      More assets (larger N) > more to estimate with the same data > larger delta > shrink more.
      Rule of thumb: T/N < 20 means shrinkage has a meaningful impact.

    Closed-form Delta pro: No cross-validation needed, given the same data, always the same delta.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Demeaned return matrix, T observations, N assets.
        Each column has mean zero: X = R - R.mean(axis=0).
        Not the raw return matrix R.

    Returns
    -------
    shrunk_cov : np.ndarray, shape (N, N), shrunk covariance matrix
    delta: float, shrinkage intensity in [0, 1].
        0 = pure sample covariance, 1 = pure scaled identity.
    """

    T, N  = X.shape
    S     = X.T @ X / (T - 1)   # sample covariance, ddof=1
    I     = np.eye(N)            # identity matrix

    mu    = np.trace(S) / N      # mean asset variance — scales the target
    tr_S  = np.trace(S)          # sum of diagonal variances
    frob2 = np.trace(S @ S)      # ||S||²_F — squared Frobenius norm

    num   = ((T - 2) / T) * frob2 + tr_S ** 2
    denom = (T + 2) * (frob2 - tr_S ** 2 / N)
    delta = float(np.clip(num / denom if denom != 0 else 1.0, 0.0, 1.0))

    return (1 - delta) * S + delta * mu * I, delta


def var_parametric_cov(asset_returns: ArrayLike, weights: ArrayLike, confidence: float = 0.99, shrink: bool = True, ) -> dict:
    """
    Parametric VaR via the full covariance matrix (attribution only).

    Uses the Normal assumption and current snapshot weights to decompose total portfolio VaR into per-security contributions (component VaR).

    This is NOT the primary risk number.
    This function answers: "which position is driving my risk?"

    Component VaR is additive, sum(component_var) == portfolio VaR exactly

    Parameters
    ----------
    asset_returns : array-like, shape (T, N), per-security log-returns
    weights : array-like, shape (N,), current portfolio weights
    confidence : float, default 0.99
    shrink : bool, default True, apply Ledoit-Wolf shrinkage. Recommended when T/N < 20.

    Returns
    -------
    dict with keys:
        var : float, portfolio VaR (positive)
        sigma_port : float, portfolio daily volatility
        shrinkage : float, Ledoit-Wolf delta (0=none, 1=full)
        component_var : ndarray (N), per-security VaR contributions

    Examples
    --------
    R = np.random.default_rng(0).multivariate_normal([0,0], np.eye(2)*0.0001, 500)  # np.eye(2)= [[1,0],[0,1]]
    w = np.array([0.6, 0.4])
    res = var_parametric_cov(R, w)
    res['var'] > 0
    abs(res['component_var'].sum() - res['var']) < 1e-10
    """
    R = np.asarray(asset_returns, dtype=float)
    w = np.asarray(weights,       dtype=float)

    if R.ndim != 2:
        raise ValueError("asset_returns must be 2-D (T, N)")
    if len(w) != R.shape[1]:
        raise ValueError(f"weights length {len(w)} != number of assets {R.shape[1]}")

    w = w / w.sum()               # normalise, weights must sum to 1

    T, N = R.shape
    mu = R.mean(axis=0)
    X = R - mu                    # demean before covariance estimation

    cov, delta = _ledoit_wolf(X) if shrink else (X.T @ X / (T - 1), 0.0)

    mu_port = float(w @ mu)  # portfolio mean return
    sigma_port = float(np.sqrt(w @ cov @ w))  # portfolio volatility
    z = stats.norm.ppf(1 - confidence)  # Normal quantile
    var_1d = -(mu_port + z * sigma_port)  # VaR = -(mean + z * vol)

    # Risk Attribution
    # Step 1: Beta of each asset against the portfolio:
    #   beta_i = Cov(asset_i, portfolio) / Var(portfolio)
    cov_w = cov @ w
    beta = cov_w / (sigma_port ** 2)

    # Step 2: Marginal VaR: how much does portfolio VaR change if weight_i increases by 1?
    #   marginal_VaR_i = -z * sigma_port * beta_i
    marginal_var = -z * sigma_port * beta

    # Step 3 — Component VaR: scale by actual weight to get each asset's contribution.
    #   component_VaR_i = w_i * marginal_VaR_i
    #   mean_comp_i     = -mu_i * w_i             — mean return also reduces VaR
    vol_comp = w * marginal_var  # volatility contribution per asset
    mean_comp = -mu * w  # mean return contribution per asset
    component_var = vol_comp + mean_comp  # total per asset: sums exactly to var_1d

    return {
        "var": float(var_1d),
        "sigma_port": sigma_port,
        "shrinkage": delta,
        "component_var": component_var,
    }
