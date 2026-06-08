"""sim_re -- diagnose a simulated market's behaviour and form trading hypotheses.

We do NOT assume the simulator literally follows a textbook process. The DGP zoo
(GBM / OU / trending / regime / jump) is a *reference library of analogies*, not a
list of guaranteed truths. The engine reads diagnostics, summarises behaviour, and
emits competing hypotheses with confidence -- decisions, never.

Pipeline:
    raw series -> fingerprint (diagnostics) -> analyze_series (behavioural
    hypotheses + warnings + next test).  The zoo + Router add a "closest
    reference" analogy as one extra piece of evidence.

    from sim_re import analyze_series
    report = analyze_series(unknown_series)
    report["hypotheses"][0]          # primary hypothesis (with confidence)
    report["recommended_next_action"]  # e.g. "test_mean_reversion_out_of_sample"

Every trading decision downstream must pass the frozen hard-risk gate in
`sim_re.risk` BEFORE any strategy logic runs.
"""
from . import dgp_zoo, evaluation, fingerprint, interpret, risk
from .fingerprint import fingerprint as compute_fingerprint
from .fingerprint import coint_pvalue, FEATURES
from .interpret import analyze_series, nearest_reference, CANDIDATE_STRATEGY_TESTS
from .risk import RiskLimits, AccountState, RiskDecision, risk_gate
from .signature import (
    build_library, Router, Library,
    CANDIDATE_STRATEGY_BY_REFERENCE, STRATEGY_FOR,
)

__all__ = [
    "dgp_zoo", "evaluation", "fingerprint", "interpret", "risk",
    "compute_fingerprint", "coint_pvalue", "FEATURES",
    "analyze_series", "nearest_reference", "CANDIDATE_STRATEGY_TESTS",
    "RiskLimits", "AccountState", "RiskDecision", "risk_gate",
    "build_library", "Router", "Library",
    "CANDIDATE_STRATEGY_BY_REFERENCE", "STRATEGY_FOR",
]
