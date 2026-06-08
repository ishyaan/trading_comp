"""CI gate: reference-recovery sanity check on the zoo library.

The router is NOT a verdict layer -- it only supplies a "closest reference
process" analogy to the behavioural analyzer (see test_interpret.py for the real
output contract). This test just confirms the reference library is sane: a series
generated from a known family should land closest to that same family most of the
time. If it didn't, the analogy evidence would be useless.
"""
import numpy as np
import pytest

from sim_re import build_library, Router, dgp_zoo, coint_pvalue


@pytest.fixture(scope="module")
def router():
    lib = build_library(n_samples=40, series_len=3000, seed=0, verbose=False)
    return Router(lib)


def test_out_of_sample_accuracy(router):
    rng = np.random.default_rng(123)
    correct = total = 0
    for name in dgp_zoo.UNIVARIATE:
        for _ in range(12):
            s = int(rng.integers(0, 2**31 - 1))
            series = dgp_zoo.generate(name, n=3000, seed=s)
            correct += int(router.classify(series)["best"] == name)
            total += 1
    assert correct / total >= 0.80, f"accuracy {correct}/{total} below floor"


def test_each_family_mostly_recovered(router):
    rng = np.random.default_rng(7)
    for name in dgp_zoo.UNIVARIATE:
        hits = sum(
            router.classify(dgp_zoo.generate(name, n=3000,
                            seed=int(rng.integers(0, 2**31 - 1))))["best"] == name
            for _ in range(8)
        )
        assert hits >= 5, f"{name}: only {hits}/8 recovered"


def test_cointegration_detection():
    # Cointegrated pair: reliably significant.
    y1, y2 = dgp_zoo.cointegrated_pair(n=3000, seed=1)
    assert coint_pvalue(y1, y2) < 0.05
    # Independent random walks: not significant on average (single draws can be
    # spuriously low -- Granger-Newbold -- so judge the median over seeds).
    ps = [coint_pvalue(dgp_zoo.gbm(n=3000, seed=10 + i),
                       dgp_zoo.gbm(n=3000, seed=200 + i)) for i in range(7)]
    assert np.median(ps) > 0.10
