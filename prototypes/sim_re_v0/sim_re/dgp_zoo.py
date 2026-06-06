"""DGP zoo: synthetic data-generating processes with KNOWN parameters.

Each univariate generator returns a 1-D numpy array representing the *observable*
you would see in the sim (a log-price for the random-walk-like families, a level
for the mean-reverting family). The fingerprint battery treats the observable
as-is, so what you simulate here is what you would fingerprint on the real sim.

The point of the zoo is not to be realistic markets -- it is to be a labelled
training set. You run the fingerprint on each known process to learn "this is
what an OU looks like under our diagnostics", then pattern-match the real sim
against that library in Week 1.
"""
from __future__ import annotations

import numpy as np

# Names are the labels the router predicts. Keep them stable; they key the library.
UNIVARIATE = (
    "gbm",            # efficient random walk -> sit out / pure risk management
    "ou",             # mean reverting       -> stat-arb / mean reversion
    "trending",       # positively autocorrelated returns -> momentum
    "regime",         # vol / drift regime switching -> regime switcher
    "jump",           # jump diffusion (fat tails) -> robust sizing, fade jumps
)


def _rng(seed):
    return np.random.default_rng(seed)


def gbm(n=4000, mu=0.05, sigma=0.20, s0=100.0, dt=1 / 252, seed=None):
    """Geometric Brownian motion. Observable = log price (a random walk)."""
    r = _rng(seed)
    drift = (mu - 0.5 * sigma**2) * dt
    shocks = r.normal(drift, sigma * np.sqrt(dt), size=n)
    log_price = np.log(s0) + np.cumsum(shocks)
    return log_price


def ou(n=4000, theta=5.0, mu=0.0, sigma=0.30, x0=0.0, dt=1 / 252, seed=None):
    """Ornstein-Uhlenbeck. Observable = a stationary level (e.g. a spread).

    Half-life of mean reversion = ln(2) / theta (in the same time units as 1/dt).
    """
    r = _rng(seed)
    x = np.empty(n)
    x[0] = x0
    sd = sigma * np.sqrt(dt)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + r.normal(0.0, sd)
    return x


def trending(n=4000, phi=0.25, sigma=0.012, s0=100.0, seed=None):
    """Positively autocorrelated returns (AR(1) on log-returns). Momentum."""
    r = _rng(seed)
    eps = r.normal(0.0, sigma, size=n)
    ret = np.empty(n)
    ret[0] = eps[0]
    for t in range(1, n):
        ret[t] = phi * ret[t - 1] + eps[t]
    return np.log(s0) + np.cumsum(ret)


def regime(n=4000, dt=1 / 252, seed=None,
           drifts=(0.10, -0.10), vols=(0.12, 0.45), p_stay=0.985, s0=100.0):
    """Two-state Markov regime switch in drift and volatility (vol clustering)."""
    r = _rng(seed)
    state = 0
    shocks = np.empty(n)
    for t in range(n):
        if r.random() > p_stay:
            state = 1 - state
        mu, sig = drifts[state], vols[state]
        shocks[t] = r.normal((mu - 0.5 * sig**2) * dt, sig * np.sqrt(dt))
    return np.log(s0) + np.cumsum(shocks)


def jump(n=4000, mu=0.05, sigma=0.15, s0=100.0, dt=1 / 252, seed=None,
         lam=12.0, jump_mu=-0.002, jump_sigma=0.04):
    """Merton jump diffusion. Diffusive random walk + compound Poisson jumps."""
    r = _rng(seed)
    drift = (mu - 0.5 * sigma**2) * dt
    diff = r.normal(drift, sigma * np.sqrt(dt), size=n)
    n_jumps = r.poisson(lam * dt, size=n)
    jumps = np.array([
        r.normal(jump_mu * k, jump_sigma * np.sqrt(k)) if k > 0 else 0.0
        for k in n_jumps
    ])
    return np.log(s0) + np.cumsum(diff + jumps)


def cointegrated_pair(n=4000, theta=6.0, sigma_spread=0.25, sigma_common=0.02,
                      beta=1.0, dt=1 / 252, seed=None):
    """Two log-price series sharing a common stochastic trend.

    y2 follows a random walk (the common factor); y1 = beta*y2 + spread, where
    the spread is a stationary OU process. The pair is cointegrated even though
    each leg on its own looks like a random walk. Returns (y1, y2).
    """
    r = _rng(seed)
    common = np.cumsum(r.normal(0.0, sigma_common, size=n))
    spread = ou(n=n, theta=theta, mu=0.0, sigma=sigma_spread, dt=dt,
                seed=None if seed is None else seed + 7)
    y2 = np.log(100.0) + common
    y1 = np.log(100.0) + beta * common + spread
    return y1, y2


# --- parameter randomisers: vary params within a plausible band per draw so the
# --- signature library captures within-family spread, not a single point.
def random_params(name, rng):
    if name == "gbm":
        return dict(mu=rng.uniform(-0.1, 0.15), sigma=rng.uniform(0.10, 0.35))
    if name == "ou":
        return dict(theta=rng.uniform(2.0, 12.0), sigma=rng.uniform(0.15, 0.45))
    if name == "trending":
        return dict(phi=rng.uniform(0.18, 0.42), sigma=rng.uniform(0.008, 0.018))
    if name == "regime":
        return dict(p_stay=rng.uniform(0.975, 0.992),
                    vols=(rng.uniform(0.08, 0.16), rng.uniform(0.35, 0.55)))
    if name == "jump":
        return dict(lam=rng.uniform(6.0, 20.0), sigma=rng.uniform(0.10, 0.20),
                    jump_sigma=rng.uniform(0.03, 0.06))
    raise KeyError(name)


_GEN = {"gbm": gbm, "ou": ou, "trending": trending, "regime": regime, "jump": jump}


def generate(name, n=4000, seed=None, **overrides):
    """Generate one realisation of a named univariate DGP."""
    rng = _rng(seed)
    params = random_params(name, rng)
    params.update(overrides)
    # give the generator its own seed so realisations differ
    gen_seed = None if seed is None else int(rng.integers(0, 2**31 - 1))
    return _GEN[name](n=n, seed=gen_seed, **params)
