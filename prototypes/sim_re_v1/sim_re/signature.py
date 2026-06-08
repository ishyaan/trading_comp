"""Signature library + router.

`build_library` runs the fingerprint battery over many randomised realisations of
each zoo family and stores, per family, the centroid (median feature vector) and
the spread. It also fits a global per-feature scaler so that features measured on
different scales (a p-value vs a half-life of 400) contribute comparably to the
distance.

`Router.classify` fingerprints an unknown series, standardises it with the same
scaler, and reports the nearest family by standardised distance, plus a softmax
confidence and the per-feature contributions (so you can defend the call in
London: "it routed to OU because adf_p and half_life dominated the distance").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dgp_zoo
from .fingerprint import FEATURES, fingerprint, to_vector


@dataclass
class Library:
    centroids: dict          # name -> median feature vector (raw)
    spreads: dict            # name -> IQR-ish spread per feature (raw)
    mean: np.ndarray         # global feature mean (for scaling)
    std: np.ndarray          # global feature std  (for scaling)
    names: list = field(default_factory=list)

    def standardise(self, vec):
        return (np.asarray(vec, float) - self.mean) / self.std


def build_library(n_samples=60, series_len=4000, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    raw = {name: [] for name in dgp_zoo.UNIVARIATE}
    for name in dgp_zoo.UNIVARIATE:
        if verbose:
            print(f"  fingerprinting {name:9s} x{n_samples} ...", flush=True)
        for _ in range(n_samples):
            s = int(rng.integers(0, 2**31 - 1))
            series = dgp_zoo.generate(name, n=series_len, seed=s)
            raw[name].append(to_vector(fingerprint(series)))
    all_vecs = np.vstack([np.vstack(v) for v in raw.values()])
    mean = all_vecs.mean(axis=0)
    std = all_vecs.std(axis=0)
    std[std == 0] = 1.0
    centroids = {k: np.median(np.vstack(v), axis=0) for k, v in raw.items()}
    spreads = {k: np.subtract(*np.percentile(np.vstack(v), [75, 25], axis=0))
               for k, v in raw.items()}
    return Library(centroids, spreads, mean, std, list(dgp_zoo.UNIVARIATE))


class Router:
    def __init__(self, library: Library):
        self.lib = library

    def classify(self, series, top_k=3):
        vec = to_vector(fingerprint(series))
        z = self.lib.standardise(vec)
        dists = {}
        contribs = {}
        for name in self.lib.names:
            cz = self.lib.standardise(self.lib.centroids[name])
            diff2 = (z - cz) ** 2
            dists[name] = float(np.sqrt(diff2.sum()))
            contribs[name] = dict(zip(FEATURES, diff2))
        order = sorted(dists, key=dists.get)
        d = np.array([dists[n] for n in order])
        conf = np.exp(-d)
        conf = conf / conf.sum()
        ranked = [
            {"dgp": order[i], "distance": dists[order[i]], "confidence": float(conf[i])}
            for i in range(min(top_k, len(order)))
        ]
        best = order[0]
        top_drivers = sorted(contribs[best].items(), key=lambda kv: -kv[1])[:3]
        return {
            "best": best,
            "ranked": ranked,
            "drivers": [f for f, _ in top_drivers],
            "fingerprint": dict(zip(FEATURES, vec)),
        }


# Map the closest reference family to a CANDIDATE strategy to *test* -- never a
# trade instruction. The router gives an analogy ("behaves like our OU"); these
# are the experiments that analogy suggests, all gated by out-of-sample
# validation and the hard risk rules.
CANDIDATE_STRATEGY_BY_REFERENCE = {
    "gbm": "no exploitable structure expected -> sit out / pure risk management",
    "ou": "candidate: OU/stat-arb mean reversion, sized by half-life (validate OOS)",
    "trending": "candidate: time-series momentum / trend following (validate OOS)",
    "regime": "candidate: regime-scaled exposure to the detected vol state (validate OOS)",
    "jump": "candidate: fat-tail-aware caps; test fade-vs-ignore jumps (validate OOS)",
}

# Backwards-compatible alias (old name implied a final 'deploy this' instruction).
STRATEGY_FOR = CANDIDATE_STRATEGY_BY_REFERENCE
