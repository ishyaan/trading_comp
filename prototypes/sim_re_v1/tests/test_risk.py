"""The frozen hard-risk gate must override strategy logic and fail safe."""
from sim_re import RiskLimits, AccountState, risk_gate


def test_healthy_confident_state_is_allowed():
    d = risk_gate(RiskLimits(), AccountState(daily_pnl_fraction=0.005,
                  trades_this_period=1), hypothesis_confidence=0.8)
    assert d.allow and not d.halt


def test_low_confidence_blocks_but_does_not_halt():
    d = risk_gate(RiskLimits(), AccountState(), hypothesis_confidence=0.2)
    assert not d.allow and not d.halt


def test_daily_loss_halts():
    d = risk_gate(RiskLimits(max_daily_loss=0.02),
                  AccountState(daily_pnl_fraction=-0.05),
                  hypothesis_confidence=0.9)
    assert not d.allow and d.halt


def test_drawdown_halts():
    d = risk_gate(RiskLimits(max_total_drawdown=0.10),
                  AccountState(drawdown_fraction=0.15),
                  hypothesis_confidence=0.9)
    assert not d.allow and d.halt


def test_consecutive_errors_halt():
    d = risk_gate(RiskLimits(max_consecutive_errors=3),
                  AccountState(consecutive_errors=3),
                  hypothesis_confidence=0.9)
    assert not d.allow and d.halt


def test_behaviour_drift_halts():
    d = risk_gate(RiskLimits(), AccountState(behavior_still_matches=False),
                  hypothesis_confidence=0.9)
    assert not d.allow and d.halt


def test_position_size_and_leverage_skip():
    d = risk_gate(RiskLimits(max_position_size=0.20),
                  AccountState(), hypothesis_confidence=0.9,
                  intended_position_size=0.5)
    assert not d.allow and not d.halt


def test_trade_throttle_skips():
    d = risk_gate(RiskLimits(max_trades_per_period=10),
                  AccountState(trades_this_period=10),
                  hypothesis_confidence=0.9)
    assert not d.allow and not d.halt
