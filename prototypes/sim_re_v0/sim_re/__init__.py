"""sim_re -- reverse-engineer a simulated market's data-generating process.

Pipeline:  DGP zoo  ->  fingerprint battery  ->  signature library  ->  router

    from sim_re import build_library, Router, dgp_zoo
    lib = build_library()
    router = Router(lib)
    result = router.classify(unknown_series)   # -> {"best": "ou", ...}
"""
from . import dgp_zoo, evaluation, fingerprint
from .fingerprint import fingerprint as compute_fingerprint
from .fingerprint import coint_pvalue, FEATURES
from .signature import build_library, Router, Library, STRATEGY_FOR

__all__ = [
    "dgp_zoo", "evaluation", "fingerprint",
    "compute_fingerprint", "coint_pvalue", "FEATURES",
    "build_library", "Router", "Library", "STRATEGY_FOR",
]
