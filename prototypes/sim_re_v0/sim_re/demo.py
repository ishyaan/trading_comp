"""End-to-end demonstration / self-test.

Run:  python -m sim_re.demo
"""
from __future__ import annotations

import numpy as np

from . import dgp_zoo
from .signature import build_library, Router, STRATEGY_FOR
from .fingerprint import coint_pvalue
from .evaluation import deflated_sharpe, sharpe


def main():
    print("=" * 70)
    print("BUILD: signature library from the DGP zoo")
    print("=" * 70)
    lib = build_library(n_samples=40, series_len=3500, seed=0)

    print("\n" + "=" * 70)
    print("VALIDATE: out-of-sample classification (fresh seeds, not in library)")
    print("=" * 70)
    router = Router(lib)
    rng = np.random.default_rng(999)
    n_per = 30
    correct = 0
    total = 0
    confusion = {a: {b: 0 for b in dgp_zoo.UNIVARIATE} for a in dgp_zoo.UNIVARIATE}
    sample_shown = set()

    for true_name in dgp_zoo.UNIVARIATE:
        for _ in range(n_per):
            s = int(rng.integers(0, 2**31 - 1))
            series = dgp_zoo.generate(true_name, n=3500, seed=s)
            res = router.classify(series)
            pred = res["best"]
            confusion[true_name][pred] += 1
            total += 1
            correct += int(pred == true_name)
            if true_name not in sample_shown:
                sample_shown.add(true_name)
                conf = res["ranked"][0]["confidence"]
                print(f"\n  true={true_name:9s} -> pred={pred:9s} "
                      f"(conf {conf:.0%}, drivers: {', '.join(res['drivers'])})")
                print(f"      strategy: {STRATEGY_FOR[pred]}")

    print("\n" + "-" * 70)
    print(f"  out-of-sample accuracy: {correct}/{total} = {correct/total:.0%}")
    print("\n  confusion matrix (rows=true, cols=predicted):")
    header = "           " + "".join(f"{n[:5]:>7s}" for n in dgp_zoo.UNIVARIATE)
    print(header)
    for a in dgp_zoo.UNIVARIATE:
        row = "".join(f"{confusion[a][b]:>7d}" for b in dgp_zoo.UNIVARIATE)
        print(f"    {a:7s}{row}")

    print("\n" + "=" * 70)
    print("PAIRWISE: cointegration detector")
    print("=" * 70)
    y1, y2 = dgp_zoo.cointegrated_pair(n=3500, seed=1)
    g1 = dgp_zoo.gbm(n=3500, seed=2)
    g2 = dgp_zoo.gbm(n=3500, seed=3)
    print(f"  cointegrated pair   -> EG p-value = {coint_pvalue(y1, y2):.4f}  "
          f"({'COINTEGRATED' if coint_pvalue(y1, y2) < 0.05 else 'no'})")
    print(f"  two unrelated GBMs  -> EG p-value = {coint_pvalue(g1, g2):.4f}  "
          f"({'COINTEGRATED' if coint_pvalue(g1, g2) < 0.05 else 'no'})")

    print("\n" + "=" * 70)
    print("VALIDATE-AND-KILL: deflated Sharpe on a candidate edge")
    print("=" * 70)
    # Simulate a strategy P&L: a genuine edge vs pure noise, after trying configs.
    edge = rng.normal(0.0016, 0.01, size=750)      # real structural drift
    noise = rng.normal(0.0, 0.01, size=750)         # zero edge, overfit candidate
    for label, pnl in [("real-edge candidate", edge), ("noise candidate", noise)]:
        d = deflated_sharpe(pnl, n_trials=10)
        verdict = "KEEP" if d["psr_vs_deflated_benchmark"] > 0.95 else "KILL"
        print(f"  {label:22s} obs Sharpe={d['observed_sharpe']:5.2f} "
              f"deflated-benchmark={d['deflated_benchmark_sharpe']:4.2f} "
              f"PSR={d['psr_vs_deflated_benchmark']:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
