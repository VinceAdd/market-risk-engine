"""
risk_engine
===========
A minimal market risk library.

Modules
-------
volatility : EWMA conditional volatility
risk_models : Historical VaR, FHS-VaR, CVaR, Covariance VaR
rolling_var : Rolling out-of-sample VaR and CVaR forecast series
modelvalidation : Binomial, Christoffersen, CVaR exceedance, Basel Traffic Light
"""

from .volatility import ewma_volatility
from .risk_models import var_historical, var_fhs, cvar, cvar_fhs, var_parametric_cov
from .rolling_var import rolling_var, rolling_cvar
from .model_validation import (
    binomial_test,
    christoffersen_test,
    cvar_exceedance_test,
    basel_traffic_light,
    backtest_summary,
)

__version__ = "1.0.0"

__all__ = [
    "ewma_volatility",
    "var_historical",
    "var_fhs",
    "cvar",
    "cvar_fhs",
    "var_parametric_cov",
    "rolling_var",
    "rolling_cvar",
    "binomial_test",
    "christoffersen_test",
    "cvar_exceedance_test",
    "basel_traffic_light",
    "backtest_summary",
]
