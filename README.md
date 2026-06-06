# Quant Competition

This repository contains my work for the quantitative trading competition.

## Goal

Reverse-engineer the simulated market, choose a suitable strategy, validate it, and trade with disciplined risk management.

## Main workflow

1. Run diagnostics on market data
2. Identify the likely data-generating process
3. Select a strategy
4. Backtest and validate
5. Apply risk controls
6. Run live execution
7. Log every decision

## Key files

- `decision_log.md`: important hypotheses, tests, results, and decisions
- `runbook.md`: live trading rules and kill-switches
- `questions_for_organizers.md`: questions to clarify with the competition organizers
- `reports/dgp_report.md`: simulator diagnostics and findings
- `configs/`: strategy and risk settings
- `src/quant_comp/`: main code
- `scripts/`: runnable scripts