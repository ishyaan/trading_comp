"""The analyzer must form honest behavioural hypotheses, not rigid verdicts.

These tests pin the design contract from the brief:
  - OU-like data    -> mean-reversion is the primary hypothesis
  - trending data   -> trend is the primary hypothesis
  - GBM/random walk -> low edge / no reliable edge, low directional confidence
  - jump-heavy data -> a fat-tail/jump WARNING is raised
  - ambiguous data  -> NOT forced into a confident classification
"""
import numpy as np
import pytest

from sim_re import dgp_zoo, analyze_series


def _primary(series):
    return analyze_series(series)["hypotheses"][0]


def _conf_of(report, name):
    for h in report["hypotheses"]:
        if h["name"] == name:
            return h["confidence"]
    return 0.0


def test_ou_gives_mean_reversion_primary():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(8):
        s = int(rng.integers(0, 2**31 - 1))
        series = dgp_zoo.generate("ou", n=3000, seed=s)
        if _primary(series)["name"] == "mean_reverting_behavior":
            hits += 1
    assert hits >= 6, f"OU recovered mean-reversion only {hits}/8 times"


def test_trending_gives_trend_primary():
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(8):
        s = int(rng.integers(0, 2**31 - 1))
        series = dgp_zoo.generate("trending", n=3000, seed=s)
        if _primary(series)["name"] == "trending_behavior":
            hits += 1
    assert hits >= 6, f"trending recovered trend only {hits}/8 times"


def test_gbm_gives_low_edge_and_low_directional_confidence():
    rng = np.random.default_rng(3)
    edge_primary = 0
    for _ in range(8):
        s = int(rng.integers(0, 2**31 - 1))
        report = analyze_series(dgp_zoo.generate("gbm", n=3000, seed=s))
        if report["hypotheses"][0]["name"] == "no_reliable_edge":
            edge_primary += 1
        # neither directional hypothesis should be confident on a random walk
        assert _conf_of(report, "mean_reverting_behavior") < 0.5
        assert _conf_of(report, "trending_behavior") < 0.5
    assert edge_primary >= 6, f"GBM flagged no-edge only {edge_primary}/8 times"
    assert report["recommended_next_action"] in ("sit_out", "collect_more_data")


def test_jump_data_raises_fat_tail_warning():
    rng = np.random.default_rng(4)
    warned = 0
    for _ in range(6):
        s = int(rng.integers(0, 2**31 - 1))
        report = analyze_series(dgp_zoo.generate("jump", n=3000, seed=s))
        assert report["behavioral_profile"]["jump_or_fat_tail_risk"] in (
            "medium", "strong")
        if any("tail" in w.lower() or "jump" in w.lower()
               for w in report["warnings"]):
            warned += 1
    assert warned >= 5, f"jump data raised a fat-tail warning only {warned}/6 times"


def test_ambiguous_data_is_not_forced_into_confidence():
    # An almost-pure random walk with a whisper of structure: nothing should be
    # confidently asserted.
    rng = np.random.default_rng(7)
    walk = np.cumsum(rng.normal(0, 0.01, size=1500))
    report = analyze_series(walk)
    top = report["hypotheses"][0]
    # either it lands on no-edge, or no directional hypothesis is confident
    if top["name"] != "no_reliable_edge":
        assert top["confidence"] < 0.6
    assert report["recommended_next_action"] in ("sit_out", "collect_more_data")


def test_short_sample_degrades_confidence_and_warns():
    rng = np.random.default_rng(8)
    # a clean OU but very short: confidence must be damped, sample warned
    series = dgp_zoo.generate("ou", n=180, seed=int(rng.integers(0, 2**31 - 1)))
    report = analyze_series(series)
    assert report["data_summary"]["data_factor"] < 1.0
    assert any("short sample" in w.lower() for w in report["warnings"])


def test_output_contract_shape():
    report = analyze_series(dgp_zoo.generate("ou", n=2000, seed=0))
    for key in ("raw_diagnostics", "behavioral_profile", "hypotheses",
                "warnings", "recommended_next_action"):
        assert key in report
    for k in ("adf_pvalue", "kpss_pvalue", "hurst", "variance_ratio",
              "return_autocorr_lag1", "excess_kurtosis", "arch_lm_pvalue"):
        assert k in report["raw_diagnostics"]
    for h in report["hypotheses"]:
        assert set(("name", "confidence", "supporting_evidence",
                    "contradicting_evidence", "suggested_strategy_tests")) <= set(h)
        assert 0.0 <= h["confidence"] <= 1.0
    # candidate strategies must be framed as tests, never as trade instructions
    joined = " ".join(report["hypotheses"][0]["suggested_strategy_tests"]).lower()
    assert "deploy now" not in joined and "trade now" not in joined
    assert any(w in joined for w in ("backtest", "test", "validate", "out-of-sample"))
