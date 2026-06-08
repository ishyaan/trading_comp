"""Week-1 tool: diagnose a REAL sim price series and form trading hypotheses.

    python -m sim_re.analyze path/to/sim_prices.csv --col close --log
    python -m sim_re.analyze a.csv b.csv          # also runs a cointegration test
    python -m sim_re.analyze sim_prices.csv --json # machine-readable report

Reads one column of prices per file and prints a *behavioural hypothesis report*:
raw diagnostics, a weak/medium/strong behavioural profile, competing hypotheses
with confidence and evidence, warnings, and a recommended next *test*. It does
NOT say "the DGP is OU" and it does NOT tell you to trade -- every candidate
strategy is something to validate out-of-sample, gated by the hard risk rules.

The DGP zoo is consulted only for a "closest reference" analogy (one extra piece
of evidence), never as a label. With two files it also reports the Engle-Granger
cointegration p-value.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .signature import build_library, Router, CANDIDATE_STRATEGY_BY_REFERENCE
from .fingerprint import coint_pvalue
from .interpret import analyze_series, nearest_reference


def _load(path, col, use_log):
    df = pd.read_csv(path)
    series = df[col].to_numpy(float) if col else df.iloc[:, -1].to_numpy(float)
    series = series[np.isfinite(series)]
    if use_log:
        if np.any(series <= 0):
            sys.exit(f"--log requested but {path} has non-positive values")
        series = np.log(series)
    return series


def _print_report(path, series, report, ref):
    print("\n" + "=" * 66)
    print(f"FILE: {path}   (n={len(series)})")
    print("=" * 66)

    print("Raw diagnostics:")
    for k, v in report["raw_diagnostics"].items():
        print(f"   {k:26s} {v:.4g}")

    print("\nBehavioural profile (weak / medium / strong):")
    for k, v in report["behavioral_profile"].items():
        print(f"   {k:26s} {v}")

    print("\nHypotheses (ranked by confidence -- NOT verdicts):")
    for h in report["hypotheses"]:
        print(f"   - {h['name']:32s} conf={h['confidence']:.2f} "
              f"({h['confidence_label']})")
        if h["supporting_evidence"]:
            print(f"       supporting:   {', '.join(h['supporting_evidence'])}")
        if h["contradicting_evidence"]:
            print(f"       contradicting:{', '.join(h['contradicting_evidence'])}")

    primary = report["hypotheses"][0]
    print(f"\nPrimary hypothesis: {primary['name']} "
          f"(confidence {primary['confidence']:.2f}, {primary['confidence_label']})")
    print("Candidate strategy TESTS for the primary hypothesis:")
    for t in primary["suggested_strategy_tests"]:
        print(f"   - {t}")

    if ref is not None:
        print(f"\nClosest reference process (analogy only): "
              f"{ref['closest_reference_process']}")
        print(f"   suggests testing: "
              f"{CANDIDATE_STRATEGY_BY_REFERENCE[ref['closest_reference_process']]}")

    print("\nWarnings:")
    for w in report["warnings"]:
        print(f"   ! {w}")

    print(f"\n>> Recommended next action: {report['recommended_next_action']}")
    print("   (a TEST, not a trade. Validate out-of-sample, then obey risk rules.)")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Diagnose a sim price series and form trading hypotheses.")
    p.add_argument("csv", nargs="+", help="CSV file(s); a column of prices.")
    p.add_argument("--col", default=None, help="Column name (default: last column).")
    p.add_argument("--log", action="store_true",
                   help="Take log of the series (use for raw prices).")
    p.add_argument("--json", action="store_true",
                   help="Emit the machine-readable report(s) as JSON.")
    p.add_argument("--no-reference", action="store_true",
                   help="Skip building the zoo reference library (faster).")
    p.add_argument("--samples", type=int, default=40,
                   help="Library Monte-Carlo samples per reference (default 40).")
    args = p.parse_args(argv)

    router = None
    if not args.no_reference:
        print("Building reference library from the DGP zoo (analogy only) ...",
              flush=True)
        lib = build_library(n_samples=args.samples, series_len=3500, seed=0,
                            verbose=False)
        router = Router(lib)

    series_list = []
    json_out = []
    for path in args.csv:
        s = _load(path, args.col, args.log)
        series_list.append(s)
        report = analyze_series(s)
        ref = nearest_reference(s, router) if router is not None else None
        if args.json:
            json_out.append({"file": path, "report": report, "reference": ref})
        else:
            _print_report(path, s, report, ref)

    coint = None
    if len(series_list) == 2:
        n = min(map(len, series_list))
        pv = coint_pvalue(series_list[0][:n], series_list[1][:n])
        coint = {"engle_granger_pvalue": pv,
                 "interpretation": ("cointegrated -- candidate pair stat-arb to TEST"
                                    if pv < 0.05 else "not cointegrated")}
        if not args.json:
            print("\n" + "=" * 66)
            print(f"PAIRWISE cointegration (Engle-Granger): p = {pv:.4f}  "
                  f"-> {coint['interpretation']}")

    if args.json:
        print(json.dumps({"files": json_out, "cointegration": coint},
                         indent=2, default=float))


if __name__ == "__main__":
    main()
