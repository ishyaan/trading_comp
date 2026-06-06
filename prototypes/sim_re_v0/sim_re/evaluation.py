"""Validate-and-kill harness: is the routed edge structural or an overfit?

On a simulator the danger is fitting the sim's own noise. Two guards from the
plan (§1, §6):

  Probabilistic Sharpe Ratio (PSR) -- prob the true Sharpe > a benchmark, given
  the sample length and the return distribution's skew/kurtosis.

  Deflated Sharpe Ratio (DSR) -- PSR with the benchmark raised to account for the
  number of configurations you tried. Trying many strategies inflates the best
  observed Sharpe even with zero real edge; DSR deflates for that selection bias.
  (Bailey & Lopez de Prado, 2014.)

Plus a PurgedKFold splitter so cross-validation folds don't leak via overlapping
labels/embargo -- standard for serial financial data.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def sharpe(returns, periods_per_year=252):
    r = np.asarray(returns, float)
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def probabilistic_sharpe(returns, sr_benchmark=0.0, periods_per_year=252):
    """PSR: P(true annualised Sharpe > sr_benchmark)."""
    r = np.asarray(returns, float)
    n = len(r)
    sr_ann = sharpe(r, periods_per_year)
    sr = sr_ann / np.sqrt(periods_per_year)            # per-period
    sr_b = sr_benchmark / np.sqrt(periods_per_year)
    g = ((r - r.mean()) / r.std(ddof=1))
    skew = np.mean(g**3)
    kurt = np.mean(g**4)                                # non-excess
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    z = (sr - sr_b) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe(returns, n_trials, periods_per_year=252,
                    var_sr_trials=None):
    """DSR: PSR against a benchmark inflated for `n_trials` configurations tried.

    var_sr_trials: variance (in PER-PERIOD Sharpe units) of the Sharpe ratios
    across the trials you ran. If unknown, falls back to the asymptotic sampling
    variance of a single Sharpe estimate, (1 + 0.5*SR^2)/n -- a reasonable proxy
    when all trials run on series of similar length.
    """
    r = np.asarray(returns, float)
    n = len(r)
    sr_pp = sharpe(r, periods_per_year) / np.sqrt(periods_per_year)
    if var_sr_trials is None:
        var_sr_trials = (1.0 + 0.5 * sr_pp**2) / n
    e = np.e
    gamma = 0.5772156649  # Euler-Mascheroni
    maxz = (1 - gamma) * norm.ppf(1 - 1.0 / n_trials) + \
        gamma * norm.ppf(1 - 1.0 / (n_trials * e))
    sr_benchmark = np.sqrt(var_sr_trials) * maxz * np.sqrt(periods_per_year)
    return {
        "psr_vs_deflated_benchmark": probabilistic_sharpe(
            returns, sr_benchmark=sr_benchmark, periods_per_year=periods_per_year),
        "deflated_benchmark_sharpe": float(sr_benchmark),
        "observed_sharpe": sharpe(returns, periods_per_year),
        "n_trials": int(n_trials),
    }


def purged_kfold(n_obs, n_splits=5, embargo=0.01):
    """Yield (train_idx, test_idx) with purging + embargo around each test fold."""
    idx = np.arange(n_obs)
    fold_sizes = np.full(n_splits, n_obs // n_splits)
    fold_sizes[: n_obs % n_splits] += 1
    emb = int(n_obs * embargo)
    start = 0
    for fs in fold_sizes:
        test = idx[start:start + fs]
        lo, hi = test[0], test[-1]
        train = np.concatenate([idx[: max(lo - emb, 0)], idx[hi + 1 + emb:]])
        yield train, test
        start += fs
