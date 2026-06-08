"""Frozen hard-risk gate -- the layer that overrides ALL strategy logic.

Philosophy (explicitly the opposite of the analyzer's job): the analyzer is
exploratory and forms hypotheses; this module is rigid and says NO. During the
exam-week autonomous run, the trading bot must not invent strategies -- it runs
only frozen, pre-approved rules, and *every* order first passes this gate.

The gate is checked BEFORE any strategy logic. If any hard rule is breached it
blocks (and may demand a full halt). These limits are constants you freeze after
validation; they are not tuned live.

This module has no opinion about *what* to trade -- only about whether trading is
permitted at all right now. Keep it dumb, total, and easy to audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskLimits:
    """Frozen hard limits. Set once, after validation; never edited live."""
    max_daily_loss: float = 0.02          # fraction of capital lost today
    max_total_drawdown: float = 0.10      # fraction from peak equity
    max_position_size: float = 0.20       # fraction of capital in one position
    max_leverage: float = 1.0             # gross exposure / capital
    max_trades_per_period: int = 50       # throttle per period (e.g. per day)
    min_confidence_to_trade: float = 0.50 # analyzer hypothesis confidence floor
    max_consecutive_errors: int = 3       # API/order failures before halt


@dataclass
class AccountState:
    """Live, mutable state the gate reads. Populated by the execution bot."""
    daily_pnl_fraction: float = 0.0       # signed; negative = loss today
    drawdown_fraction: float = 0.0        # >=0; distance below peak equity
    open_position_size: float = 0.0       # fraction of capital, this instrument
    gross_leverage: float = 0.0
    trades_this_period: int = 0
    consecutive_errors: int = 0
    behavior_still_matches: bool = True   # set False if live data drifts from
    #                                       the validated hypothesis


@dataclass
class RiskDecision:
    allow: bool
    halt: bool                 # True => stop trading for the session, not just skip
    reasons: list = field(default_factory=list)


def risk_gate(limits: RiskLimits, state: AccountState,
              hypothesis_confidence: float,
              intended_position_size: float = 0.0) -> RiskDecision:
    """Return whether an order may proceed. Hard rules first; strategy logic never
    runs unless this returns allow=True.

    `halt=True` means a structural breach (loss / drawdown / error / behaviour
    drift): the bot should flatten and stop for the session, not merely skip one
    order.
    """
    reasons = []
    halt = False

    # --- structural breaches: stop the whole session ----------------------- #
    if state.daily_pnl_fraction <= -abs(limits.max_daily_loss):
        reasons.append("max daily loss hit -> HALT")
        halt = True
    if state.drawdown_fraction >= limits.max_total_drawdown:
        reasons.append("max total drawdown hit -> HALT")
        halt = True
    if state.consecutive_errors >= limits.max_consecutive_errors:
        reasons.append("too many consecutive API/order errors -> HALT")
        halt = True
    if not state.behavior_still_matches:
        reasons.append("live behaviour no longer matches validated hypothesis -> HALT")
        halt = True

    # --- per-order vetoes: skip this order, session may continue ------------ #
    if hypothesis_confidence < limits.min_confidence_to_trade:
        reasons.append("hypothesis confidence below floor -> skip")
    if state.trades_this_period >= limits.max_trades_per_period:
        reasons.append("max trades per period reached -> skip")
    if intended_position_size > limits.max_position_size:
        reasons.append("intended position exceeds max size -> skip")
    if state.gross_leverage > limits.max_leverage:
        reasons.append("gross leverage over limit -> skip")

    allow = (not halt) and not reasons
    if allow:
        reasons.append("all hard risk checks passed")
    return RiskDecision(allow=allow, halt=halt, reasons=reasons)
