"""Week-1 tool: fingerprint and route a REAL sim price series.

    python -m sim_re.analyze path/to/sim_prices.csv --col close --log
    python -m sim_re.analyze a.csv b.csv          # also runs a cointegration test

Reads one column of prices per file, builds (or reuses) the signature library,
prints the observed fingerprint, the routed DGP family with confidence and the
features that drove the decision, and the pre-validated strategy posture to
deploy. With two files it also reports the Engle-Granger cointegration p-value.

This is the bridge from Phase 0 (zoo, built blind) to Week 1 (the real sim).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .signature import build_library, Router, STRATEGY_FOR
from .fingerprint import coint_pvalue, FEATURES


def _load(path, col, use_log):
    df = pd.read_csv(path)
    series = df[col].to_numpy(float) if col else df.iloc[:, -1].to_numpy(float)
    series = series[np.isfinite(series)]
    if use_log:
        if np.any(series <= 0):
            sys.exit(f"--log requested but {path} has non-positive values")
        series = np.log(series)
    return series


def main(argv=None):
    p = argparse.ArgumentParser(description="Fingerprint & route a sim price series.")
    p.add_argument("csv", nargs="+", help="CSV file(s); a column of prices.")
    p.add_argument("--col", default=None, help="Column name (default: last column).")
    p.add_argument("--log", action="store_true",
                   help="Take log of the series (use for raw prices).")
    p.add_argument("--samples", type=int, default=40,
                   help="Library Monte-Carlo samples per DGP (default 40).")
    args = p.parse_args(argv)

    print("Building signature library from the DGP zoo ...", flush=True)
    lib = build_library(n_samples=args.samples, series_len=3500, seed=0, verbose=False)
    router = Router(lib)

    series_list = []
    for path in args.csv:
        s = _load(path, args.col, args.log)
        series_list.append(s)
        res = router.classify(s)
        print("\n" + "=" * 64)
        print(f"FILE: {path}   (n={len(s)})")
        print("=" * 64)
        print("Observed fingerprint:")
        for f in FEATURES:
            print(f"   {f:14s} {res['fingerprint'][f]:.4g}")
        print(f"\nRouted DGP:  {res['best'].upper()}")
        print("Ranked:")
        for r in res["ranked"]:
            print(f"   {r['dgp']:9s} dist={r['distance']:.2f} conf={r['confidence']:.0%}")
        print(f"Decision drivers: {', '.join(res['drivers'])}")
        print(f"\n>> Deploy: {STRATEGY_FOR[res['best']]}")

    if len(series_list) == 2:
        n = min(map(len, series_list))
        pv = coint_pvalue(series_list[0][:n], series_list[1][:n])
        print("\n" + "=" * 64)
        print(f"PAIRWISE cointegration (Engle-Granger): p = {pv:.4f}  "
              f"-> {'COINTEGRATED, consider pair stat-arb' if pv < 0.05 else 'not cointegrated'}")


if __name__ == "__main__":
    main()
