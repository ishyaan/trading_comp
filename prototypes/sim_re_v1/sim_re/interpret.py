"""Behavioural hypothesis analyzer: diagnostics -> hypotheses (not verdicts).

This is the honest output layer. It deliberately does **not** say "the DGP is OU".
The simulator is unknown; we never assume it literally follows one textbook
process. Instead we:

  1. read raw statistical diagnostics (from `fingerprint`),
  2. summarise them into a behavioural profile (weak / medium / strong),
  3. form competing *hypotheses* about exploitable structure, each with a
     confidence, supporting and contradicting evidence,
  4. emit warnings (fat tails, short sample, conflicting tests, ambiguity),
  5. recommend a *next test*, never a trade.

The reference DGP zoo (GBM / OU / trending / regime / jump) is used only as a
library of analogies -- "this behaves a bit like our OU reference" -- and the
nearest reference is reported as one more piece of evidence, never as a label.

Design rule: a confident classification must be *earned*. When signals conflict
or the sample is short, confidence collapses and the recommendation degrades to
`collect_more_data` or `sit_out`. There is an explicit "no reliable edge"
hypothesis so that random-walk-like data is reported as such instead of being
force-fit to the nearest family.
"""
from __future__ import annotations

import numpy as np

from .fingerprint import fingerprint


# --------------------------------------------------------------------------- #
# small soft-threshold helpers (keep the logic transparent and defensible)
# --------------------------------------------------------------------------- #
def _ramp(x, lo, hi):
    """Linear 0->1 ramp: <=lo gives 0, >=hi gives 1 (clamped). hi may be < lo."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    t = (x - lo) / (hi - lo)
    return float(min(1.0, max(0.0, t)))


def _bucket(score):
    """Map a 0..1 evidence score to a weak/medium/strong label."""
    if score >= 0.67:
        return "strong"
    if score >= 0.34:
        return "medium"
    return "weak"


def _conf_label(c):
    if c >= 0.67:
        return "high"
    if c >= 0.34:
        return "medium"
    return "low"


# --------------------------------------------------------------------------- #
# extra robustness diagnostics computed directly on the series
# --------------------------------------------------------------------------- #
def _vol_of_vol(ret, n_windows=10):
    """Coefficient of variation of rolling volatility. High => vol regimes."""
    r = np.asarray(ret, float)
    w = max(10, len(r) // n_windows)
    if len(r) < 2 * w:
        return 0.0
    vols = [r[i:i + w].std() for i in range(0, len(r) - w, w)]
    vols = np.asarray([v for v in vols if v > 0])
    if len(vols) < 2 or vols.mean() == 0:
        return 0.0
    return float(vols.std() / vols.mean())


def _std_ratio_halves(ret):
    """|log| ratio of return std between the two halves. High => instability."""
    r = np.asarray(ret, float)
    h = len(r) // 2
    if h < 5:
        return 0.0
    s1, s2 = r[:h].std(), r[h:].std()
    if s1 == 0 or s2 == 0:
        return 0.0
    return float(abs(np.log(s2 / s1)))


def _clustering_signal(ret, acf1):
    """Volatility clustering measured on AR(1)-FILTERED residuals.

    Raw ARCH-LM on returns is fooled by linear momentum: an AR(1) in returns
    induces autocorrelation in *squared* returns even with constant volatility.
    Filtering the lag-1 structure out first means we only flag clustering that is
    genuinely in the volatility, not a side effect of a trend. Returns the lag-1
    autocorrelation of squared residuals in [~0, 1)."""
    r = np.asarray(ret, float)
    r = r - r.mean()
    if len(r) < 4:
        return 0.0
    resid = r[1:] - acf1 * r[:-1]          # remove the linear AR(1) component
    sq = resid ** 2
    if sq.std() == 0:
        return 0.0
    return float(np.corrcoef(sq[:-1], sq[1:])[0, 1])


# --------------------------------------------------------------------------- #
# candidate strategy TESTS (not trade instructions) per behaviour
# --------------------------------------------------------------------------- #
CANDIDATE_STRATEGY_TESTS = {
    "mean_reverting_behavior": [
        "out-of-sample mean-reversion backtest on a held-out window (z-score bands)",
        "estimate OU half-life and check it is stable across sub-samples",
        "purged-CV stat-arb sizing; deflated-Sharpe gate before any deployment",
    ],
    "trending_behavior": [
        "out-of-sample time-series momentum backtest on a held-out window",
        "breakout / moving-average-crossover with purged CV",
        "check the autocorrelation persists out-of-sample, not just in-sample",
    ],
    "volatility_clustering_behavior": [
        "volatility-targeting / regime-scaled exposure backtest",
        "compare a GARCH vol forecast against realised vol out-of-sample",
        "confirm the edge is in *sizing*, not direction, before trading",
    ],
    "jump_fat_tail_behavior": [
        "fat-tail-aware position caps; do NOT rely on a naive Sharpe ratio",
        "test jump-fade vs jump-follow on held-out data with robust sizing",
        "stress-test drawdown under the empirical (not normal) tail",
    ],
    "no_reliable_edge": [
        "run a random-trading control vs the candidate on held-out data",
        "if no edge is confirmed: sit out / pure risk management",
        "collect more data and re-diagnose before committing capital",
    ],
}


# --------------------------------------------------------------------------- #
# the analyzer
# --------------------------------------------------------------------------- #
def analyze_series(series, fp=None, data_floor=250, data_full=1500):
    """Produce a behavioural hypothesis report for one observable series.

    `series`     : 1-D array (a log-price, a level, a spread -- as you observe it).
    `fp`         : optional precomputed fingerprint dict (avoids recompute).
    returns      : the structured report dict (see module docstring / README).
    """
    x = np.asarray(series, float)
    ret = np.diff(x)
    n = len(x)
    if fp is None:
        fp = fingerprint(x)

    # confidence is scaled down on short samples: diagnostics are noisy there.
    data_factor = _ramp(n, data_floor, data_full)

    vol_cv = _vol_of_vol(ret)
    std_ratio = _std_ratio_halves(ret)
    sq_resid_autocorr = _clustering_signal(ret, fp["acf1"])

    # ---- raw diagnostics (named for downstream consumers) ----------------- #
    raw = {
        "adf_pvalue": fp["adf_p"],
        "kpss_pvalue": fp["kpss_p"],
        "hurst": fp["hurst"],
        "variance_ratio": fp["var_ratio"],
        "return_autocorr_lag1": fp["acf1"],
        "excess_kurtosis": fp["excess_kurt"],
        "arch_lm_pvalue": fp["arch_p"],
        # extras, still part of the evidence base
        "ou_half_life": fp["half_life"],
        "jump_fraction": fp["jump_frac"],
        "jarque_bera_pvalue": fp["jarque_bera_p"],
        "ljung_box_returns_pvalue": fp["lb_ret_p"],
        "ljung_box_sq_pvalue": fp["lb_sq_p"],
        "vol_of_vol": vol_cv,
        "std_ratio_halves": std_ratio,
        "sq_resid_autocorr": sq_resid_autocorr,
        "n_observations": n,
    }

    # ---- behavioural votes (each in 0..1), then averaged per dimension ----- #
    # mean reversion: stationary level, anti-persistent, sub-unit variance ratio
    mr_votes = {
        "adf_says_stationary": 1.0 - _ramp(fp["adf_p"], 0.01, 0.10),
        "kpss_says_stationary": _ramp(fp["kpss_p"], 0.05, 0.10),
        "hurst_below_half": 1.0 - _ramp(fp["hurst"], 0.40, 0.50),
        "variance_ratio_below_one": 1.0 - _ramp(fp["var_ratio"], 0.80, 1.00),
        "negative_return_autocorr": 1.0 - _ramp(fp["acf1"], -0.10, 0.00),
        "finite_short_half_life": 1.0 - _ramp(fp["half_life"], 20.0, 400.0),
    }
    # trend / momentum: persistent, super-unit variance ratio, positive autocorr
    tr_votes = {
        "hurst_above_half": _ramp(fp["hurst"], 0.50, 0.60),
        "variance_ratio_above_one": _ramp(fp["var_ratio"], 1.00, 1.25),
        "positive_return_autocorr": _ramp(fp["acf1"], 0.00, 0.10),
        "significant_return_autocorr": (1.0 - _ramp(fp["lb_ret_p"], 0.01, 0.10))
        * (1.0 if fp["acf1"] > 0 else 0.0),
    }
    # stationarity (separate axis): ADF and KPSS agreeing that the level is stable
    st_votes = {
        "adf_says_stationary": 1.0 - _ramp(fp["adf_p"], 0.01, 0.10),
        "kpss_says_stationary": _ramp(fp["kpss_p"], 0.05, 0.10),
    }
    # volatility clustering: measured on AR-FILTERED residuals (so a linear trend
    # is not mistaken for clustering) plus the dispersion of rolling volatility.
    vc_votes = {
        "filtered_squared_resid_autocorr": _ramp(sq_resid_autocorr, 0.03, 0.15),
        "rolling_vol_dispersion": _ramp(vol_cv, 0.25, 0.75),
    }
    # jumps / fat tails: excess kurtosis, tail fraction, non-normal returns
    jp_votes = {
        "excess_kurtosis_high": _ramp(fp["excess_kurt"], 1.0, 6.0),
        "tail_fraction_high": _ramp(fp["jump_frac"], 5e-4, 4e-3),
        "returns_non_normal": 1.0 - _ramp(fp["jarque_bera_p"], 0.01, 0.10),
    }
    # regime instability: vol-of-vol, half-sample vol shift, persistent clustering
    rg_votes = {
        "vol_of_vol_high": _ramp(vol_cv, 0.20, 0.60),
        "half_sample_vol_shift": _ramp(std_ratio, 0.30, 1.00),
        "persistent_vol_clustering": _ramp(sq_resid_autocorr, 0.05, 0.20),
    }

    mr = float(np.mean(list(mr_votes.values())))
    tr = float(np.mean(list(tr_votes.values())))
    st = float(np.mean(list(st_votes.values())))
    vc = float(np.mean(list(vc_votes.values())))
    jp = float(np.mean(list(jp_votes.values())))
    rg = float(np.mean(list(rg_votes.values()) + [vc]))  # clustering reinforces regimes

    behavioral_profile = {
        "mean_reversion_evidence": _bucket(mr),
        "trend_evidence": _bucket(tr),
        "stationarity_evidence": _bucket(st),
        "volatility_clustering": _bucket(vc),
        "jump_or_fat_tail_risk": _bucket(jp),
        "regime_instability": _bucket(rg),
    }

    # ---- assemble hypotheses --------------------------------------------- #
    def _fired(votes, thresh=0.5):
        return [k for k, v in votes.items() if v >= thresh]

    def _not_fired(votes, thresh=0.5):
        return [k for k, v in votes.items() if v < thresh]

    hypotheses = []

    # mean reversion vs trend are mutually contradicting: each penalises the other
    mr_conf = mr * (1.0 - 0.5 * tr) * data_factor
    hypotheses.append({
        "name": "mean_reverting_behavior",
        "confidence": round(mr_conf, 3),
        "confidence_label": _conf_label(mr_conf),
        "supporting_evidence": _fired(mr_votes),
        "contradicting_evidence": _fired(tr_votes) + _not_fired(mr_votes),
        "suggested_strategy_tests": CANDIDATE_STRATEGY_TESTS["mean_reverting_behavior"],
    })

    tr_conf = tr * (1.0 - 0.5 * mr) * data_factor
    hypotheses.append({
        "name": "trending_behavior",
        "confidence": round(tr_conf, 3),
        "confidence_label": _conf_label(tr_conf),
        "supporting_evidence": _fired(tr_votes),
        "contradicting_evidence": _fired(mr_votes) + _not_fired(tr_votes),
        "suggested_strategy_tests": CANDIDATE_STRATEGY_TESTS["trending_behavior"],
    })

    vc_conf = vc * (1.0 - 0.5 * tr) * data_factor
    if vc >= 0.34:
        hypotheses.append({
            "name": "volatility_clustering_behavior",
            "confidence": round(vc_conf, 3),
            "confidence_label": _conf_label(vc_conf),
            "supporting_evidence": _fired(vc_votes) + _fired(rg_votes),
            "contradicting_evidence": _not_fired(vc_votes),
            "suggested_strategy_tests":
                CANDIDATE_STRATEGY_TESTS["volatility_clustering_behavior"],
        })

    jp_conf = jp * data_factor
    if jp >= 0.34:
        hypotheses.append({
            "name": "jump_fat_tail_behavior",
            "confidence": round(jp_conf, 3),
            "confidence_label": _conf_label(jp_conf),
            "supporting_evidence": _fired(jp_votes),
            "contradicting_evidence": _not_fired(jp_votes),
            "suggested_strategy_tests": CANDIDATE_STRATEGY_TESTS["jump_fat_tail_behavior"],
        })

    # the honest baseline: how much does this just look like noise / random walk?
    directional = max(mr, tr, 0.7 * vc, 0.6 * jp)
    edge_conf = (1.0 - directional) * data_factor
    hypotheses.append({
        "name": "no_reliable_edge",
        "confidence": round(edge_conf, 3),
        "confidence_label": _conf_label(edge_conf),
        "supporting_evidence": (
            ["no directional structure stands out"] if directional < 0.34 else []
        ),
        "contradicting_evidence": _fired(mr_votes) + _fired(tr_votes),
        "suggested_strategy_tests": CANDIDATE_STRATEGY_TESTS["no_reliable_edge"],
    })

    hypotheses.sort(key=lambda h: h["confidence"], reverse=True)

    # ---- warnings --------------------------------------------------------- #
    warnings_out = []
    if n < 500:
        warnings_out.append(
            f"Short sample (n={n}): diagnostics are unstable; treat every "
            "hypothesis as tentative.")
    if jp >= 0.34:
        warnings_out.append(
            "Fat tails / jumps present: cap position size, use robust sizing; a "
            "backtest Sharpe may be tail-driven and not repeatable.")
    if rg >= 0.5:
        warnings_out.append(
            "Volatility appears regime-dependent: an edge measured in one regime "
            "may vanish in another; size to the current regime.")
    if (fp["adf_p"] < 0.10) != (fp["kpss_p"] > 0.05):
        warnings_out.append(
            "Stationarity tests disagree (ADF vs KPSS): mean-reversion evidence "
            "is mixed, not conclusive.")
    top = hypotheses[0]
    second = hypotheses[1] if len(hypotheses) > 1 else None
    ambiguous = (
        top["confidence"] < 0.34
        or (second is not None
            and top["confidence"] - second["confidence"] < 0.12
            and top["confidence"] < 0.60)
    )
    if ambiguous:
        warnings_out.append(
            "Evidence is ambiguous: no hypothesis is clearly dominant. Do not "
            "deploy a directional strategy on this alone.")
    # standing caution, always present
    warnings_out.append(
        "Candidate strategies are hypotheses to validate out-of-sample, NOT trade "
        "instructions. Hard risk limits must gate any deployment.")

    # ---- recommended next action ----------------------------------------- #
    if n < data_floor:
        action = "collect_more_data"
    elif ambiguous or top["name"] == "no_reliable_edge":
        action = "sit_out"
    elif top["name"] == "mean_reverting_behavior":
        action = "test_mean_reversion_out_of_sample"
    elif top["name"] == "trending_behavior":
        action = "test_momentum_out_of_sample"
    elif top["name"] == "volatility_clustering_behavior":
        action = "test_volatility_timing_out_of_sample"
    else:  # jump-dominant with no direction: nothing to trade directionally
        action = "sit_out"

    return {
        "raw_diagnostics": raw,
        "behavioral_profile": behavioral_profile,
        "hypotheses": hypotheses,
        "warnings": warnings_out,
        "recommended_next_action": action,
        "data_summary": {"n_observations": n, "data_factor": round(data_factor, 3)},
    }


def nearest_reference(series, router):
    """Optional: closest zoo family as an ANALOGY only (one piece of evidence).

    Returns the router's ranking re-labelled to make clear it is not a verdict.
    """
    res = router.classify(series)
    return {
        "closest_reference_process": res["best"],
        "note": "analogy only -- the sim is not assumed to BE this process",
        "ranked_distances": res["ranked"],
        "distance_drivers": res["drivers"],
    }
