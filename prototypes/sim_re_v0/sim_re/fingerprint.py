"""Fingerprint battery: turn an observable series into a feature vector.

Composes statsmodels + scipy rather than reimplementing tests. Every feature is
chosen because it separates the zoo families:

  level tests (run on the observable as-is)
    adf_p      ADF p-value      low  => level is stationary (mean-reverting)
    kpss_p     KPSS p-value     high => level is stationary
    half_life  OU half-life     small & finite => fast mean reversion

  return tests (run on the first difference of the observable)
    excess_kurt   fat tails (jumps / regimes)
    jarque_bera_p normality of returns (low => non-normal)
    lb_ret_p      Ljung-Box on returns      low => return autocorrelation (trend)
    lb_sq_p       Ljung-Box on squared ret  low => volatility clustering
    arch_p        ARCH-LM                   low => conditional heteroskedasticity
    var_ratio     Lo-MacKinlay VR(q)        >1 trend, <1 revert, ~1 random walk
    hurst         Hurst exponent            >0.5 trend, <0.5 revert, ~0.5 RW
    jump_frac     fraction of |r| > 4 sigma (vs ~6e-5 expected under normal)

The router never inspects raw prices -- only this vector -- so the battery is the
single source of truth for what "an OU looks like" vs "a jump diffusion looks
like". Cointegration is pairwise and lives in `coint_pvalue` below.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# Canonical feature order. The router relies on this ordering.
FEATURES = (
    "adf_p", "kpss_p", "half_life",
    "excess_kurt", "jarque_bera_p",
    "lb_ret_p", "lb_sq_p", "arch_p",
    "var_ratio", "hurst", "jump_frac", "acf1",
)


def _acf1(ret):
    """Lag-1 autocorrelation of returns. ~phi for momentum, ~0 for a random walk."""
    r = np.asarray(ret, float)
    if len(r) < 3 or r.std() == 0:
        return 0.0
    return float(np.corrcoef(r[:-1], r[1:])[0, 1])


def _half_life(level):
    """OU half-life via dx_t = a + b * x_{t-1}. Returns large number if no reversion."""
    x = np.asarray(level, float)
    lag = x[:-1]
    dx = np.diff(x)
    b = np.polyfit(lag, dx, 1)[0]
    if b >= 0:
        return 1e4  # no mean reversion -> effectively infinite
    hl = -np.log(2) / np.log(1 + b)
    return float(np.clip(hl, 0.0, 1e4))


def _variance_ratio(ret, q=8):
    """Lo-MacKinlay variance ratio VR(q)."""
    r = np.asarray(ret, float)
    n = len(r)
    if n < q * 4:
        return 1.0
    var_1 = np.var(r, ddof=1)
    agg = np.cumsum(r)
    rq = agg[q:] - agg[:-q]
    var_q = np.var(rq, ddof=1)
    if var_1 == 0:
        return 1.0
    return float(var_q / (q * var_1))


def _hurst(series, max_lag=40):
    """Hurst via the rescaled-lag / variance-of-differences regression."""
    x = np.asarray(series, float)
    lags = range(2, min(max_lag, len(x) // 2))
    tau = [np.sqrt(np.std(x[lag:] - x[:-lag])) for lag in lags]
    tau = np.asarray(tau)
    good = tau > 0
    if good.sum() < 4:
        return 0.5
    poly = np.polyfit(np.log(np.array(list(lags))[good]), np.log(tau[good]), 1)
    return float(poly[0] * 2.0)


def _jump_frac(ret, k=4.0):
    r = np.asarray(ret, float)
    sd = np.std(r)
    if sd == 0:
        return 0.0
    return float(np.mean(np.abs(r - r.mean()) > k * sd))


def fingerprint(series, vr_q=8, lb_lags=10) -> dict:
    """Compute the full feature dict for one observable series."""
    x = np.asarray(series, float)
    ret = np.diff(x)
    out = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS interpolation / small-sample notes
        out["adf_p"] = float(adfuller(x, autolag="AIC")[1])
        try:
            out["kpss_p"] = float(kpss(x, regression="c", nlags="auto")[1])
        except Exception:
            out["kpss_p"] = 0.5
        out["half_life"] = _half_life(x)

        out["excess_kurt"] = float(stats.kurtosis(ret, fisher=True))
        out["jarque_bera_p"] = float(stats.jarque_bera(ret)[1])

        lb = acorr_ljungbox(ret, lags=[lb_lags], return_df=True)
        out["lb_ret_p"] = float(lb["lb_pvalue"].iloc[0])
        lb_sq = acorr_ljungbox(ret**2, lags=[lb_lags], return_df=True)
        out["lb_sq_p"] = float(lb_sq["lb_pvalue"].iloc[0])
        try:
            out["arch_p"] = float(het_arch(ret, nlags=lb_lags)[1])
        except Exception:
            out["arch_p"] = 0.5

    out["var_ratio"] = _variance_ratio(ret, q=vr_q)
    out["hurst"] = _hurst(x)
    out["jump_frac"] = _jump_frac(ret)
    out["acf1"] = _acf1(ret)
    return out


def to_vector(fp: dict) -> np.ndarray:
    return np.array([fp[f] for f in FEATURES], float)


def coint_pvalue(y1, y2):
    """Engle-Granger cointegration p-value for a candidate pair (low => cointegrated)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(coint(np.asarray(y1, float), np.asarray(y2, float))[1])
