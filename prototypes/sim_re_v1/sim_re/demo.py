"""End-to-end demonstration / self-test.

Run:  python -m sim_re.demo

Shows the full honest pipeline:
  1. behavioural analyzer on each reference family (hypotheses, not verdicts),
  2. the zoo "reference recovery" sanity check (does our analogy point home?),
  3. cointegration detection,
  4. the validate-and-kill deflated-Sharpe gate,
  5. the frozen hard-risk gate that overrides all strategy logic.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from . import dgp_zoo
from .signature import build_library, Router, CANDIDATE_STRATEGY_BY_REFERENCE
from .fingerprint import coint_pvalue
from .interpret import analyze_series
from .evaluation import deflated_sharpe
from .risk import RiskLimits, AccountState, risk_gate


def main():
    print("=" * 70)
    print("ANALYZER: behavioural hypotheses on each reference family")
    print("=" * 70)
    rng = np.random.default_rng(999)
    for ref_name in dgp_zoo.UNIVARIATE:
        series = dgp_zoo.generate(ref_name, n=3500,
                                  seed=int(rng.integers(0, 2**31 - 1)))
        rep = analyze_series(series)
        primary = rep["hypotheses"][0]
        print(f"\n  generated-from={ref_name:9s} -> primary hypothesis: "
              f"{primary['name']} (conf {primary['confidence']:.2f}, "
              f"{primary['confidence_label']})")
        print(f"      next action: {rep['recommended_next_action']}")
        print(f"      profile: {rep['behavioral_profile']}")

    print("\n" + "=" * 70)
    print("REFERENCE RECOVERY: does the closest-analogy point back home?")
    print("(a sanity check on the zoo library -- NOT a claim the sim IS a family)")
    print("=" * 70)
    lib = build_library(n_samples=40, series_len=3500, seed=0)
    router = Router(lib)
    rng = np.random.default_rng(123)
    recovered = Counter()
    total = Counter()
    for ref_name in dgp_zoo.UNIVARIATE:
        for _ in range(20):
            s = int(rng.integers(0, 2**31 - 1))
            series = dgp_zoo.generate(ref_name, n=3500, seed=s)
            closest = router.classify(series)["best"]
            recovered[ref_name] += int(closest == ref_name)
            total[ref_name] += 1
    for ref_name in dgp_zoo.UNIVARIATE:
        print(f"  {ref_name:9s} closest-analogy matches source "
              f"{recovered[ref_name]}/{total[ref_name]}")

    print("\n" + "=" * 70)
    print("PAIRWISE: cointegration detector")
    print("=" * 70)
    y1, y2 = dgp_zoo.cointegrated_pair(n=3500, seed=1)
    g1 = dgp_zoo.gbm(n=3500, seed=2)
    g2 = dgp_zoo.gbm(n=3500, seed=3)
    print(f"  cointegrated pair   -> EG p = {coint_pvalue(y1, y2):.4f}  "
          f"({'candidate pair stat-arb to TEST' if coint_pvalue(y1, y2) < 0.05 else 'no'})")
    print(f"  two unrelated GBMs  -> EG p = {coint_pvalue(g1, g2):.4f}  "
          f"({'cointegrated' if coint_pvalue(g1, g2) < 0.05 else 'no'})")

    print("\n" + "=" * 70)
    print("VALIDATE-AND-KILL: deflated Sharpe separates edge from overfit")
    print("=" * 70)
    edge = rng.normal(0.0016, 0.01, size=750)
    noise = rng.normal(0.0, 0.01, size=750)
    for label, pnl in [("real-edge candidate", edge), ("noise candidate", noise)]:
        d = deflated_sharpe(pnl, n_trials=10)
        verdict = "KEEP" if d["psr_vs_deflated_benchmark"] > 0.95 else "KILL"
        print(f"  {label:22s} obsSharpe={d['observed_sharpe']:5.2f} "
              f"PSR={d['psr_vs_deflated_benchmark']:.2f} -> {verdict}")

    print("\n" + "=" * 70)
    print("HARD-RISK GATE: overrides ALL strategy logic (checked before any order)")
    print("=" * 70)
    limits = RiskLimits()
    scenarios = [
        ("healthy, confident hypothesis",
         AccountState(daily_pnl_fraction=0.005, trades_this_period=3), 0.72),
        ("low hypothesis confidence",
         AccountState(daily_pnl_fraction=0.0, trades_this_period=3), 0.30),
        ("daily loss limit breached",
         AccountState(daily_pnl_fraction=-0.03), 0.80),
        ("behaviour drifted from validated hypothesis",
         AccountState(behavior_still_matches=False), 0.80),
    ]
    for label, state, conf in scenarios:
        d = risk_gate(limits, state, hypothesis_confidence=conf)
        tag = "ALLOW" if d.allow else ("HALT" if d.halt else "SKIP")
        print(f"  {label:42s} -> {tag:5s} ({d.reasons[0]})")


if __name__ == "__main__":
    main()
