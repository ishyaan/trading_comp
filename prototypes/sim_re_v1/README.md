# sim_re — a behavioural hypothesis engine for a simulated market

A small, **defensible** diagnostic engine for the *"Model to Market"*
simulated-market competition. It does **not** claim to identify the simulator's
true data-generating process. We do not assume the sim *is* a GBM, an OU, a jump
diffusion, or a regime-switcher. Those are **reference analogies**, not truths.

Instead the engine reads a price/return series and answers a more honest
question: *what behaviour does this market show, how strong is the evidence, and
what should I test next?*

```
raw series ─▶ diagnostics ─▶ behavioural profile ─▶ hypotheses (with confidence)
              (ADF, KPSS,     (weak/medium/strong    + warnings
               Hurst, VR,      per behaviour)        + recommended NEXT TEST
               ARCH, ...)                            (never a trade)
                         ▲
            DGP zoo ─────┘  "closest reference" = ONE extra analogy, not a label
```

## Design philosophy (read this first)

- **The DGP zoo is a reference library, not a list of guaranteed truths.** GBM /
  OU / trending / regime / jump are labelled synthetic processes we use to learn
  "what mean reversion *looks like* under our diagnostics". They are analogies.
- **The analyzer forms hypotheses, not classifications.** Output is
  `Primary hypothesis: mean-reverting behaviour, confidence 0.7` — with the
  supporting and contradicting evidence — never `Detected DGP: OU`.
- **There is an explicit "no reliable edge" outcome.** Random-walk-like data is
  reported as having no exploitable structure, not force-fit to a family.
- **Confidence must be earned.** When tests disagree or the sample is short,
  confidence collapses and the recommendation degrades to `collect_more_data`
  or `sit_out`. Ambiguous data is never forced into a confident call.
- **Candidate strategies are *tests*, not instructions.** Every suggestion is an
  out-of-sample experiment to run, gated by validation and the hard risk rules.
- **All trading decisions require validation and risk controls.** The trading bot
  must obey the frozen hard-risk gate (`sim_re.risk`) *before* any strategy logic.

## Install & run

```bash
pip install -e .          # numpy pandas scipy scikit-learn statsmodels arch
python -m sim_re.demo     # analyzer + reference recovery + risk gate, end-to-end
pytest                    # CI gate: hypothesis contract + reference sanity + risk
```

Week-1 use on the real sim (one column of prices per CSV):

```bash
python -m sim_re.analyze sim_prices.csv --col close --log      # hypothesis report
python -m sim_re.analyze sim_prices.csv --log --json           # machine-readable
python -m sim_re.analyze legA.csv legB.csv --log               # + cointegration test
python -m sim_re.analyze sim_prices.csv --log --no-reference   # skip the zoo analogy
```

## The output object

`analyze_series(series)` returns:

```jsonc
{
  "raw_diagnostics": {
    "adf_pvalue": 0.004, "kpss_pvalue": 0.10, "hurst": 0.41,
    "variance_ratio": 0.78, "return_autocorr_lag1": -0.12,
    "excess_kurtosis": 0.3, "arch_lm_pvalue": 0.6, "...": "..."
  },
  "behavioral_profile": {
    "mean_reversion_evidence": "strong",   // weak / medium / strong
    "trend_evidence": "weak",
    "stationarity_evidence": "strong",
    "volatility_clustering": "weak",
    "jump_or_fat_tail_risk": "weak",
    "regime_instability": "weak"
  },
  "hypotheses": [                          // ranked by confidence, NOT verdicts
    {
      "name": "mean_reverting_behavior",
      "confidence": 0.71,                  // 0..1, earned not assumed
      "confidence_label": "high",
      "supporting_evidence":   ["adf_says_stationary", "hurst_below_half", "..."],
      "contradicting_evidence":["..."],
      "suggested_strategy_tests": ["out-of-sample mean-reversion backtest", "..."]
    }
  ],
  "warnings": ["Candidate strategies are hypotheses to validate ... NOT trade instructions.", "..."],
  "recommended_next_action": "test_mean_reversion_out_of_sample"
  //  one of: test_mean_reversion_out_of_sample | test_momentum_out_of_sample |
  //          test_volatility_timing_out_of_sample | sit_out | collect_more_data
}
```

## What's in each module

| File | Role |
|---|---|
| `dgp_zoo.py` | Reference library of synthetic processes (GBM, OU, trending, regime, jump, cointegrated pair) — labelled analogies, **not** assumed truths |
| `fingerprint.py` | Diagnostic battery (ADF/KPSS, variance-ratio, Hurst, ARCH-LM, Ljung-Box, jump fraction, OU half-life, lag-1 ACF, Engle-Granger) |
| **`interpret.py`** | **The honest output layer: diagnostics → behavioural profile → hypotheses + warnings + next test. The centrepiece.** |
| `signature.py` | Monte-Carlo each reference family → nearest-reference `Router`. Used only to supply the "closest analogy" evidence |
| `evaluation.py` | Probabilistic & **Deflated** Sharpe (deflates for # configs tried) + PurgedKFold — the validate-and-kill harness |
| **`risk.py`** | **Frozen hard-risk gate that overrides ALL strategy logic** |
| `analyze.py` | CLI: hypothesis report for a real CSV (+ pairwise cointegration) |
| `demo.py` | End-to-end self-test of the whole pipeline |

## A note on volatility clustering (a defensible detail)

A linear trend (AR(1) in returns) induces autocorrelation in *squared* returns,
so a naive ARCH-LM test will flag "volatility clustering" on a pure momentum
series. We avoid that false positive by testing clustering on **AR-filtered
residuals** and corroborating with the dispersion of rolling volatility — so
momentum is not mistaken for clustering. (See `interpret._clustering_signal`.)

## How the analyzer connects to the trading bot

The analyzer is the **research brain**; it is exploratory and forms hypotheses.
The trading bot is the **frozen hand**; it is rigid and only executes
pre-approved rules. The handoff is one-directional and gated:

1. **Offline:** run `analyze_series` → pick a primary hypothesis → build a
   candidate strategy for it → validate **out-of-sample** with deflated Sharpe +
   purged CV. Only a strategy that survives gets *frozen*.
2. **Freeze:** the surviving rule and its risk limits (`RiskLimits`) become
   constants. The bot may not invent or tune strategies live.
3. **Live (e.g. exam week):** every order passes `risk.risk_gate(...)` **before**
   any strategy logic. Hard rules override everything:
   - max daily loss, max total drawdown  → **HALT** (flatten, stop the session)
   - too many consecutive API/order errors → **HALT**
   - live behaviour no longer matches the validated hypothesis → **HALT**
   - hypothesis confidence below floor, trade throttle hit, position/leverage
     over limit → **SKIP** this order
4. **Drift check:** periodically re-run the analyzer on live data; if the
   behavioural profile diverges from the validated hypothesis, set
   `behavior_still_matches = False` and the gate halts.

The analyzer never trades. The bot never explores. Risk always wins ties.
