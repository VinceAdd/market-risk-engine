"""
model_validation.py
==================
VaR and CVaR model validation via statistical backtesting.

Four tests, four distinct questions
-------------------------------------
binomial_test        : Is the frequency of VaR breaches correct?
                       Exact binomial test — no asymptotic approximation.

christoffersen_test  : Are breaches independent (not clustered in crises)?
                       A model can have the right frequency but always fail
                       during crises. Christoffersen detects this.

cvar_exceedance_test : When we breach, does the model correctly size the loss?
                       Frequency and clustering say nothing about severity.

basel_traffic_light  : Regulatory output only — not a statistical test.
                       Counts exceptions and returns the Basel (1996) zone
                       and capital multiplier.

backtest_summary()   : Runs all four and returns a single readable result.

Decision framework
------------------
Read the four results in order:

    Binomial rejects?       → recalibrate the model immediately.
    Christoffersen rejects? → volatility model is too slow (EWMA lambda too
                              high, window too long). Model gets the annual
                              count right but clusters all failures in crises.
    Basel Red / Yellow?     → regulatory conversation required regardless of
                              what the statistical tests say.
    CVaR ratio > 1.2?       → flag for senior review. Do not act on one year
                              alone — the test needs ~10 years to have power.
    All pass?               → model is performing as expected. Document and
                              move on.

References
----------
- Christoffersen (1998), International Economic Review
- McNeil & Frey (2000), Journal of Empirical Finance — CVaR exceedance
- Basel Committee (1996) — Traffic Light framework
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats
from numpy.typing import ArrayLike


def _exceptions(returns: np.ndarray, var_forecasts: np.ndarray) -> np.ndarray:
    """True where realised loss exceeds VaR forecast."""
    return returns < -var_forecasts


# ── Result containers ──────────────────────────────────────────────────────

@dataclass
class BinomialResult:
    """
    Result of the exact binomial frequency test.

    H0: true exception rate = 1 - confidence (e.g. 1% for 99% VaR).

    The p-value is P(X >= x | H0 true): the probability of observing this
    many exceptions or more purely by chance if the model is correctly
    calibrated. This is an exact one-tailed test — no asymptotic approximation.

    Decision rule at significance = 0.05
    -------------------------------------
    p_value < 0.05  → REJECT: too many exceptions → model underestimates risk.
    p_value >= 0.05 → ok: exception count is consistent with the model.

    Examples
    --------
    p_value = 0.01 → only 1% chance of seeing this many exceptions if the
                     model is correct → strong evidence against H0.
    p_value = 0.80 → result is normal under H0 → model performs adequately.
    """
    exceptions:    int
    n_obs:         int
    expected:      float   # n_obs * (1 - confidence)
    actual_rate:   float
    expected_rate: float
    p_value:       float
    reject:        bool

    def __repr__(self) -> str:
        verdict = "REJECT" if self.reject else "ok"
        return (
            f"Binomial [{verdict}] "
            f"{self.exceptions}/{self.n_obs} exceptions "
            f"({self.actual_rate:.2%} vs expected {self.expected_rate:.2%}  "
            f"[{self.expected:.1f} expected]  "
            f"p={self.p_value:.4f})"
        )


@dataclass
class ChristoffersenResult:
    """
    Result of Christoffersen (1998) conditional coverage test.

    Tests whether exceptions are independent of each other. A model that
    fails in clusters (crisis periods) passes a pure frequency test but
    fails here. The test decomposes into three LR statistics:

        LR_CC = LR_UC + LR_IND   (joint = frequency + independence)

    Each LR statistic is -2 * log(likelihood ratio), which is asymptotically
    chi-squared under H0. LR_UC and LR_IND each have 1 degree of freedom
    (one restriction each). LR_CC has 2 (both restrictions combined).

    How to read the three p-values
    --------------------------------
    p_value_unconditional (LR_UC):
        Same question as the binomial test — is the total count right?
        Small (< 0.05) → too many or too few exceptions overall.

    p_value_independence (LR_IND):
        Are exceptions clustering in time?
        Small (< 0.05) → breaches arrive in runs, not randomly.
        Common failure mode: model is fine in calm markets but breaches
        pile up during stress periods because volatility is underestimated.

    p_value_joint (LR_CC):
        Both frequency and independence at once.
        Small (< 0.05) → model fails overall.

    Practical examples
    ------------------
    p_unc=0.60, p_ind=0.92, p_joint=0.85  → passes both (good)
    p_unc=0.03, p_ind=0.40, p_joint=0.08  → too many exceptions overall
    p_unc=0.45, p_ind=0.007,p_joint=0.02  → right count but clustering
    p_unc=0.01, p_ind=0.02, p_joint=0.001 → fails badly on both
    """
    lr_unconditional:      float
    lr_independence:       float
    lr_joint:              float
    p_value_unconditional: float
    p_value_independence:  float
    p_value_joint:         float
    reject_independence:   bool
    reject_joint:          bool

    def __repr__(self) -> str:
        ind = "REJECT" if self.reject_independence else "ok"
        cc  = "REJECT" if self.reject_joint        else "ok"
        return (
            f"Christoffersen  "
            f"IND:{ind} (p={self.p_value_independence:.4f})  "
            f"CC:{cc} (p={self.p_value_joint:.4f})"
        )


@dataclass
class CVaRResult:
    """
    Result of the CVaR exceedance test.

    On VaR exception days only, tests H0: mean(actual_loss - CVaR_forecast) = 0
    via a one-sample t-test. This asks: on the days we got hurt, was the CVaR
    forecast large enough to cover the actual loss on average?

    The t-statistic is: mean(excess) / (std(excess) / sqrt(n_exceptions))
    The denominator (the standard error) is what makes this test weak in
    practice: at 99% VaR you only get 2-3 breach days per year, so sqrt(n)
    is tiny and the standard error stays large, swamping any real signal.
    You need roughly 10 years of data before the test can reliably detect
    even a meaningful miscalibration. Treat it as a long-horizon diagnostic,
    not a short-run gate.

    ratio = mean(actual_loss) / mean(CVaR_forecast)
        > 1.0 → model underestimates tail severity (dangerous)
        ≈ 1.0 → well calibrated
        < 1.0 → model is over-conservative

    Examples
    --------
    p_val=0.008, ratio=1.28 → strong evidence CVaR underestimates tail → red flag
    p_val=0.62,  ratio=1.03 → CVaR well calibrated
    p_val=0.03,  ratio=0.85 → CVaR too conservative
    """
    n_exceptions: int
    mean_excess:  float
    t_statistic:  float
    p_value:      float
    reject:       bool
    ratio:        float

    def __repr__(self) -> str:
        if np.isnan(self.t_statistic):
            return (
                f"CVaR exceedance [insufficient data — {self.n_exceptions} exceptions] "
                f"ratio={self.ratio:.3f}"
            )
        verdict = "REJECT" if self.reject else "ok"
        return (
            f"CVaR exceedance [{verdict}] "
            f"n_exc={self.n_exceptions}  "
            f"ratio={self.ratio:.3f} (1.0=perfect)  "
            f"p={self.p_value:.4f}"
        )


@dataclass
class BaselResult:
    """
    Basel (1996) Traffic Light — regulatory output, not a statistical test.

    Zone and capital multiplier are determined by exception count over
    250 trading days. Thresholds are regulatory conventions set in 1996,
    not statistically derived.

    Zones
    -----
    Green  (0-4 exceptions)  → no action, multiplier = 3.00
    Yellow (5-9 exceptions)  → supervisor monitoring, multiplier = 3.40-3.85
    Red    (10+ exceptions)  → presumed model failure, multiplier = 4.00

    Even if the binomial test passes at the 5% significance level, you can
    still land in Yellow — the Basel thresholds are not aligned with any
    particular statistical significance level.
    """
    zone:        str     # Green / Yellow / Red
    exceptions:  int
    multiplier:  float

    def __repr__(self) -> str:
        return (
            f"Basel [{self.zone}] "
            f"{self.exceptions} exceptions over 250 days  "
            f"capital multiplier x{self.multiplier:.2f}  "
            f"[regulatory output — not a statistical test]"
        )


@dataclass
class BacktestSummary:
    binomial:       BinomialResult
    christoffersen: ChristoffersenResult
    cvar_test:      CVaRResult | None
    basel:          BaselResult
    exception_rate: float
    n_obs:          int

    def __repr__(self) -> str:
        lines = [
            "=" * 60,
            "  BACKTEST SUMMARY",
            "=" * 60,
            f"  Observations  : {self.n_obs}",
            f"  Exception rate: {self.exception_rate:.2%}  "
            f"(expected {self.binomial.expected_rate:.2%})",
            f"  {self.binomial}",
            f"  {self.christoffersen}",
        ]
        if self.cvar_test is not None:
            lines.append(f"  {self.cvar_test}")
        lines += [f"  {self.basel}", "=" * 60]
        return "\n".join(lines)


# ── 1. Binomial Test ───────────────────────────────────────────────────────

def binomial_test(
    returns: ArrayLike,
    var_forecasts: ArrayLike,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> BinomialResult:
    """
    Exact binomial test for VaR exception frequency.

    Under H0, exceptions X ~ Binomial(T, p) where p = 1 - confidence.
    The p-value is the exact one-tailed probability P(X >= x | H0),
    computed without any asymptotic approximation.

    This is a one-tailed upper test: we only flag too many exceptions.
    Too few exceptions means a conservative model, which is not a risk
    management concern (though it may indicate an overly cautious model).

    Scipy binomial reference
    ------------------------
    P(X <= k) = stats.binom.cdf(k, T, p)   left tail, cumulative up to k
    P(X =  k) = stats.binom.pmf(k, T, p)   exactly k
    P(X >  k) = stats.binom.sf(k, T, p)    right tail, strictly greater
    P(X >= k) = stats.binom.sf(k-1, T, p)  right tail, at least k  <- used here

    Parameters
    ----------
    returns       : array-like, shape (T,)
    var_forecasts : array-like, shape (T,) — positive VaR estimates
    confidence    : float, default 0.99
    significance  : float, default 0.05

    Returns
    -------
    BinomialResult
    """
    r = np.asarray(returns,       dtype=float)
    v = np.asarray(var_forecasts, dtype=float)

    T = len(r)
    x = int(_exceptions(r, v).sum())   # observed exception count
    p = 1.0 - confidence               # theoretical exception probability under H0

    # P(X >= x) = 1 - P(X <= x-1): probability of seeing this many or more
    # exceptions if the model is correctly calibrated.
    p_val = float(stats.binom.sf(x - 1, T, p))

    return BinomialResult(
        exceptions=x,
        n_obs=T,
        expected=T * p,
        actual_rate=x / T,
        expected_rate=p,
        p_value=p_val,
        reject=p_val < significance,
    )


# ── 2. Christoffersen Test ─────────────────────────────────────────────────

def christoffersen_test(
    returns: ArrayLike,
    var_forecasts: ArrayLike,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> ChristoffersenResult:
    """
    Christoffersen (1998) conditional coverage test.

    The core question: are VaR breaches independent of each other, or do
    they cluster in stress periods? A model can pass a frequency test (right
    annual count) but still be dangerous if all failures arrive in the same
    two-week crisis window.

    The test reduces the return series to a binary sequence of 0s and 1s,
    then counts the four possible day-to-day transitions:

        n00 = calm   -> calm      n01 = calm   -> breach
        n10 = breach -> calm      n11 = breach -> breach  <- clustering signal

    Under independence, knowing yesterday was a breach should not make today
    more likely to breach: pi_01 should equal pi_11. The LR test formalises
    this by comparing a restricted model (one probability, pi_hat) against an
    unrestricted model (two probabilities, pi_01 and pi_11).

    The -2 * log(likelihood ratio) trick
    -------------------------------------
    Each LR statistic is -2 * log(L_restricted / L_unrestricted).
    By Wilks' theorem, this is asymptotically chi-squared under H0, which
    gives us a p-value without needing to derive a new distribution. The -2
    scaling is a mathematical calibration — it has no deeper intuition beyond
    making the result chi-squared distributed.

    Parameters
    ----------
    returns       : array-like, shape (T,)
    var_forecasts : array-like, shape (T,)
    confidence    : float, default 0.99
    significance  : float, default 0.05

    Returns
    -------
    ChristoffersenResult
    """
    r   = np.asarray(returns,       dtype=float)
    v   = np.asarray(var_forecasts, dtype=float)
    exc = _exceptions(r, v).astype(int)

    T   = len(exc)
    p   = 1.0 - confidence
    x   = exc.sum()
    eps = 1e-10   # clipping floor to prevent log(0)

    # ── Transition counts ──────────────────────────────────────────────────
    n00 = int(((exc[:-1] == 0) & (exc[1:] == 0)).sum())  # calm   -> calm
    n01 = int(((exc[:-1] == 0) & (exc[1:] == 1)).sum())  # calm   -> breach
    n10 = int(((exc[:-1] == 1) & (exc[1:] == 0)).sum())  # breach -> calm
    n11 = int(((exc[:-1] == 1) & (exc[1:] == 1)).sum())  # breach -> breach

    n0     = n00 + n01                                    # total days preceded by no exception
    n1     = n10 + n11                                    # total days preceded by an exception
    pi_01  = n01 / n0 if n0 > 0 else 0.0                 # P(breach | no breach yesterday)
    pi_11  = n11 / n1 if n1 > 0 else 0.0                 # P(breach | breach yesterday) — clustering signal if >> pi_01
    pi_hat = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0  # unconditional rate from transition counts

    p_hat = np.clip(x / T,    eps, 1 - eps)  # observed exception rate, clipped for log stability
    ph_c  = np.clip(pi_hat,   eps, 1 - eps)  # clipped unconditional rate — restricted model
    p01_c = np.clip(pi_01,    eps, 1 - eps)  # clipped conditional rate after no-exception — unrestricted model
    p11_c = np.clip(pi_11,    eps, 1 - eps)  # clipped conditional rate after exception    — unrestricted model

    # ── LR_UC: frequency test ──────────────────────────────────────────────
    # Compares observed rate p_hat to theoretical rate p.
    # How much does the log-likelihood improve by using p_hat instead of p?
    lr_uc = -2.0 * (
        x       * np.log(p / p_hat)             # exception days:     expected rate vs observed
        + (T-x) * np.log((1-p) / (1-p_hat))     # non-exception days
    )

    # ── LR_IND: independence test ──────────────────────────────────────────
    # Restricted model (H0): one probability pi_hat governs every day.
    # Unrestricted model:    two probabilities pi_01 and pi_11.
    # Large LR_IND means the two-probability world fits much better -> clustering is real.
    lr_ind = -2.0 * (
        (n00 + n10) * np.log(1 - ph_c)          # restricted: log-prob of no breach, unconditional
        + (n01 + n11) * np.log(ph_c)            # restricted: log-prob of breach,    unconditional
        - n00 * np.log(1 - p01_c)              # unrestricted: no breach after no breach
        - n01 * np.log(p01_c)                  # unrestricted: breach after no breach
        - n10 * np.log(1 - p11_c)              # unrestricted: no breach after breach
        - n11 * np.log(p11_c)                  # unrestricted: breach after breach
    )

    lr_cc = lr_uc + lr_ind   # joint: frequency + independence, asymptotically chi2(2)

    pv_uc  = float(stats.chi2.sf(lr_uc,  df=1))  # p-value: one restriction  (p_hat == p)
    pv_ind = float(stats.chi2.sf(lr_ind, df=1))  # p-value: one restriction  (pi_01 == pi_11)
    pv_cc  = float(stats.chi2.sf(lr_cc,  df=2))  # p-value: two restrictions combined

    return ChristoffersenResult(
        lr_unconditional=float(lr_uc),
        lr_independence=float(lr_ind),
        lr_joint=float(lr_cc),
        p_value_unconditional=pv_uc,
        p_value_independence=pv_ind,
        p_value_joint=pv_cc,
        reject_independence=pv_ind < significance,
        reject_joint=pv_cc        < significance,
    )


# ── 3. CVaR Exceedance Test ────────────────────────────────────────────────

def cvar_exceedance_test(
    returns: ArrayLike,
    var_forecasts: ArrayLike,
    cvar_forecasts: ArrayLike,
    significance: float = 0.05,
) -> CVaRResult:
    """
    Test whether CVaR forecasts correctly size tail losses.

    On VaR exception days only, tests H0: mean(actual_loss - CVaR_forecast) = 0
    via a one-sample t-test. The three input arrays all live on the same time
    axis (shape T,). var_forecasts identifies which days are exception days via
    exc_mask. actual and cvar_e are then the subsets of r and cv on those days
    only — the quiet days are irrelevant to this test.

    t-statistic = mean(excess) / (std(excess) / sqrt(n_exceptions))

    The denominator is the standard error. With 99% VaR over 250 days you
    expect only 2-3 breach days, so sqrt(n) is tiny and the standard error
    stays large, swamping any real signal. This test needs ~10 years of data
    to have meaningful power. Until then, watch the ratio as a qualitative
    signal rather than relying on the p-value as a hard gate.

    Parameters
    ----------
    returns        : array-like, shape (T,) — daily portfolio returns
    var_forecasts  : array-like, shape (T,) — used only to locate exception days
    cvar_forecasts : array-like, shape (T,) — CVaR forecast for every day;
                     only the exception-day subset is tested
    significance   : float, default 0.05

    Returns
    -------
    CVaRResult

    Reference: McNeil & Frey (2000), Journal of Empirical Finance.
    """
    r  = np.asarray(returns,        dtype=float)
    v  = np.asarray(var_forecasts,  dtype=float)
    cv = np.asarray(cvar_forecasts, dtype=float)

    exc_mask     = _exceptions(r, v)             # boolean mask: True on breach days
    n_exceptions = int(exc_mask.sum())

    if n_exceptions == 0:
        return CVaRResult(
            n_exceptions=0, mean_excess=np.nan, t_statistic=np.nan,
            p_value=np.nan, reject=False, ratio=np.nan,
        )

    actual = -r[exc_mask]    # realised loss on breach days (positive number)
    cvar_e =  cv[exc_mask]   # CVaR forecast on those same days
    excess = actual - cvar_e # positive = model undershot, negative = model overshot
    ratio  = float(actual.mean() / cvar_e.mean()) if cvar_e.mean() != 0 else np.nan

    if n_exceptions < 3:     # t-test is meaningless with fewer than 3 observations
        return CVaRResult(
            n_exceptions=n_exceptions, mean_excess=float(excess.mean()),
            t_statistic=np.nan, p_value=np.nan, reject=False, ratio=ratio,
        )

    # One-sample t-test: is the average excess significantly different from zero?
    t_stat, p_val = stats.ttest_1samp(excess, popmean=0.0)

    return CVaRResult(
        n_exceptions=n_exceptions,
        mean_excess=float(excess.mean()),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        reject=float(p_val) < significance,
        ratio=ratio,
    )


# ── 4. Basel Traffic Light ─────────────────────────────────────────────────

_YELLOW_MULTIPLIERS = {5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}

def basel_traffic_light(
    returns: ArrayLike,
    var_forecasts: ArrayLike,
    window: int = 250,
) -> BaselResult:
    """
    Basel (1996) Traffic Light — regulatory output, not a statistical test.

    Counts exceptions over the most recent `window` days and assigns a zone
    (Green / Yellow / Red) with the corresponding capital multiplier.
    Thresholds and multipliers are regulatory conventions, not derived from
    statistical theory.

    Parameters
    ----------
    returns       : array-like, shape (T,)
    var_forecasts : array-like, shape (T,)
    window        : int, default 250 — Basel standard = 250 trading days

    Returns
    -------
    BaselResult
    """
    r = np.asarray(returns,       dtype=float)
    v = np.asarray(var_forecasts, dtype=float)
    x = int(_exceptions(r[-window:], v[-window:]).sum())  # exceptions in the most recent window

    if   x <= 4: zone, mult = "Green",  3.00
    elif x <= 9: zone, mult = "Yellow", _YELLOW_MULTIPLIERS.get(x, 3.40)
    else:        zone, mult = "Red",    4.00

    return BaselResult(zone=zone, exceptions=x, multiplier=mult)


# ── 5. backtest_summary ────────────────────────────────────────────────────

def backtest_summary(
    returns: ArrayLike,
    var_forecasts: ArrayLike,
    cvar_forecasts: ArrayLike | None = None,
    confidence: float = 0.99,
    significance: float = 0.05,
    window: int = 250,
) -> BacktestSummary:
    """
    Run all validation tests and return a single BacktestSummary.

    Parameters
    ----------
    returns        : array-like, shape (T,)
    var_forecasts  : array-like, shape (T,)
    cvar_forecasts : array-like, shape (T,) or None.
                     If supplied, runs the CVaR exceedance test.
    confidence     : float, default 0.99
    significance   : float, default 0.05
    window         : int, default 250 — Basel rolling window

    Returns
    -------
    BacktestSummary
    """
    r = np.asarray(returns,       dtype=float)
    v = np.asarray(var_forecasts, dtype=float)

    binom_res = binomial_test(r, v, confidence, significance)
    christ    = christoffersen_test(r, v, confidence, significance)
    basel     = basel_traffic_light(r, v, window)
    exc_rate  = float(_exceptions(r, v).mean())

    cvar_result = (
        cvar_exceedance_test(
            r, v, np.asarray(cvar_forecasts, dtype=float), significance
        )
        if cvar_forecasts is not None else None
    )

    return BacktestSummary(
        binomial=binom_res,
        christoffersen=christ,
        cvar_test=cvar_result,
        basel=basel,
        exception_rate=exc_rate,
        n_obs=len(r),
    )
