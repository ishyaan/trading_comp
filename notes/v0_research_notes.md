# v0 Research Notes

## What this prototype is

This is my first simulator reverse-engineering analyzer. It tries to classify price movements by comparing observed data to synthetic data-generating processes.

## What it currently contains

- Synthetic DGP examples
- Price-movement fingerprinting
- A simple signature library
- A router/classifier
- Basic tests

## What I understand so far

- ADF/KPSS are used to check stationarity
- OU-like processes should look more mean-reverting
- GBM-like processes should look more like random walks
- Jump processes should show fatter tails
- Trending processes should show autocorrelation

## Current limitations

- This is not yet a trading system
- The classifier may fail if the real simulator does not match the DGP zoo
- Some diagnostics may be unreliable on short samples
- I still need to verify which diagnostics are actually useful

## Next steps

- Run tests locally
- Add example outputs to `reports/`
- Add plots for fingerprints
- Decide which parts should move into the main `src/` folder