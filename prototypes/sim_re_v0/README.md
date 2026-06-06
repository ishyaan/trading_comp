# sim_re — reverse-engineer a simulated market

A small, defensible engine for the **"Model to Market"** simulated-market
competition. It does one job, the centrepiece of the plan's Research & DGP
workstream: **figure out what process generated the sim's data, then route to the
strategy pre-validated for that process.**

```
DGP zoo  ──▶  fingerprint battery  ──▶  signature library  ──▶  router  ──▶  strategy posture
(known          (statistical            ("what an OU            (nearest
 processes)      diagnostics)            looks like")            match)
```

The whole point: you build this **blind** in Phase 0 (you can't see the sim yet),
then in Week 1 you run `analyze` on the real sim data and it pattern-matches in
seconds against everything you pre-characterised.

## Install & run

```bash
pip install -e .          # or: pip install numpy pandas scipy scikit-learn statsmodels arch
python -m sim_re.demo     # build library + out-of-sample self-test (currently ~93%)
pytest                    # CI gate: enforces classification accuracy + cointegration
```

Week-1 use on the real sim (one column of prices per CSV):

```bash
python -m sim_re.analyze sim_prices.csv --col close --log      # fingerprint + route
python -m sim_re.analyze legA.csv legB.csv --log               # also tests cointegration
```

## What's in each module

| File | Role | Plan deliverable |
|---|---|---|
| `dgp_zoo.py` | Synthetic generators with known params: GBM, OU, trending(AR), regime-switch, jump-diffusion, cointegrated pair | "DGP zoo" |
| `fingerprint.py` | 12-feature diagnostic battery (ADF/KPSS, variance-ratio, Hurst, ARCH-LM, Ljung-Box, jump fraction, OU half-life, lag-1 ACF, Engle-Granger) | "fingerprint battery" |
| `signature.py` | Monte-Carlo each family → centroid + scaler = the signature library; nearest-centroid `Router` with confidence + decision drivers | "signature library" + "router" |
| `evaluation.py` | Probabilistic & **Deflated** Sharpe (deflates for # configs tried) + PurgedKFold | "validate-and-kill" harness |
| `analyze.py` | CLI: fingerprint + route a real CSV; pairwise cointegration | Week-1 Day-1 tool |
| `demo.py` | End-to-end self-test + confusion matrix | Definition-of-Done #1 |

## How it reads in London (the defensible story)

- **Every classification is explained**: the router reports which features drove
  the decision (e.g. "routed OU because adf_p, half_life, kpss_p dominated"), so
  you can whiteboard *why*, not just *what*.
- **It exploits structure, not bugs** — it trades the data-generating process the
  diagnostics reveal, which is legitimate and patch-proof.
- **It's honest about overfitting**: deflated Sharpe + purged CV separate a real
  structural edge from a fit to the sim's noise.

## Deliberate scope / where to take it next

- The router covers 5 univariate families + pairwise cointegration. Mean-reversion
  is detected at the *level*; add a within-instrument cointegration / basket leg if
  the sim turns out to be multi-asset.
- The regime family is the catch-all volatility bucket; a quiet regime is genuinely
  near-GBM, which is why those two are the only families that occasionally swap.
  Resolve it live by estimating the regime transition rate, not by forcing the call.
- `analyze.py` rebuilds the library each run (a few seconds). For Week 1, pickle the
  `Library` once so diagnosis on the real feed is instant.
- This is the **stable core**. Keep the tuned strategies + venue adapter as the
  separate "disposable edges" layer, per the plan.
