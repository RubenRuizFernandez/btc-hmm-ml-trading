"""Tests for MT5 execution planning."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest.functional import FunctionalTradeConfig
from src.mt5.executor import (
    ControllerState,
    LiveSignal,
    MT5RuntimeConfig,
    ManagedPosition,
    _ensure_margin_within_cap,
    _margin_cap_usd,
    _tick_age_seconds,
    _latest_managed_stop_loss_deal,
    apply_controller_overrides,
    build_execution_plan,
    build_strategy_comment,
    compute_target_volume_lots,
    parse_strategy_comment,
)


def make_signal(units: int) -> LiveSignal:
    return LiveSignal(
        bar_time=pd.Timestamp("2026-04-08 10:00:00", tz="UTC"),
        close_price=70000.0,
        trend_score=4.0,
        raw_regime_state=0,
        regime_state=0 if units > 0 else 4 if units < 0 else 3,
        regime_label="Super Bull" if units > 0 else "Bear" if units < 0 else "Sideways",
        desired_units=units,
        selected_min_bars=1,
        target_notional=abs(units) * FunctionalTradeConfig().unit_notional,
    )


def make_symbol_info():
    return SimpleNamespace(
        trade_contract_size=1.0,
        volume_step=0.01,
        volume_min=0.01,
        volume_max=100.0,
        digits=2,
        point=0.01,
        filling_mode=1,
        name="BTCUSD",
    )


def make_tick():
    return SimpleNamespace(bid=69990.0, ask=70010.0)


def test_comment_round_trip_preserves_signed_units():
    signal = make_signal(-2)
    comment = build_strategy_comment("btc-rg", signal)
    assert parse_strategy_comment(comment, "btc-rg") == -2


def test_compute_target_volume_uses_notional_and_contract_size():
    cfg = FunctionalTradeConfig()
    signal = make_signal(3)
    volume = compute_target_volume_lots(signal, 70000.0, make_symbol_info(), cfg)
    assert volume == pytest.approx(1.0, abs=1e-6)


def test_plan_opens_when_signal_is_non_flat_and_no_positions():
    plan = build_execution_plan(
        signal=make_signal(2),
        positions=[],
        symbol_info=make_symbol_info(),
        tick=make_tick(),
        strategy_config=FunctionalTradeConfig(),
        runtime_config=MT5RuntimeConfig(),
    )

    assert len(plan.actions) == 1
    assert plan.actions[0].kind == "OPEN"
    assert plan.actions[0].side == "BUY"
    assert plan.target_volume > 0
    assert plan.actual_notional <= plan.max_notional_cap + 1e-6
    assert plan.stop_loss is not None


def test_plan_closes_and_reopens_when_signal_reverses():
    runtime = MT5RuntimeConfig()
    current = ManagedPosition(
        ticket=1,
        side="BUY",
        volume=0.67,
        price_open=68000.0,
        sl=65000.0,
        comment="btc-rg|u=2|r=1|mb=1",
        symbol=runtime.symbol,
        magic=runtime.magic_number,
        signed_units=2,
    )

    plan = build_execution_plan(
        signal=make_signal(-1),
        positions=[current],
        symbol_info=make_symbol_info(),
        tick=make_tick(),
        strategy_config=FunctionalTradeConfig(),
        runtime_config=runtime,
    )

    assert [action.kind for action in plan.actions] == ["CLOSE", "OPEN"]
    assert plan.actual_notional <= plan.max_notional_cap + 1e-6
    assert plan.actions[0].side == "SELL"
    assert plan.actions[1].side == "SELL"


def test_plan_updates_stop_when_position_matches_signal():
    runtime = MT5RuntimeConfig()
    signal = make_signal(1)
    plan_seed = build_execution_plan(
        signal=signal,
        positions=[],
        symbol_info=make_symbol_info(),
        tick=make_tick(),
        strategy_config=FunctionalTradeConfig(),
        runtime_config=runtime,
    )
    current = ManagedPosition(
        ticket=7,
        side="BUY",
        volume=plan_seed.target_volume,
        price_open=69950.0,
        sl=plan_seed.stop_loss - 100.0,
        comment=build_strategy_comment(runtime.comment_prefix, signal),
        symbol=runtime.symbol,
        magic=runtime.magic_number,
        signed_units=1,
    )

    plan = build_execution_plan(
        signal=signal,
        positions=[current],
        symbol_info=make_symbol_info(),
        tick=make_tick(),
        strategy_config=FunctionalTradeConfig(),
        runtime_config=runtime,
    )

    assert len(plan.actions) == 1
    assert plan.actions[0].kind == "UPDATE_SL"
    assert plan.actions[0].ticket == 7


def test_apply_controller_overrides_blocks_entries_during_cooldown():
    signal = make_signal(2)
    state = ControllerState(
        cooldown_until=pd.Timestamp("2026-04-09 12:00:00", tz="UTC"),
        last_stop_loss_time=pd.Timestamp("2026-04-08 12:00:00", tz="UTC"),
        last_stop_loss_ticket=99,
    )

    effective_signal, blocked_reason = apply_controller_overrides(
        signal,
        state,
        now=pd.Timestamp("2026-04-08 18:00:00", tz="UTC"),
    )

    assert effective_signal.desired_units == 0
    assert effective_signal.target_notional == 0.0
    assert blocked_reason is not None
    assert "Cooldown active" in blocked_reason


def test_latest_managed_stop_loss_deal_filters_magic_reason_and_symbol():
    runtime = MT5RuntimeConfig()
    fake_mt5 = SimpleNamespace(DEAL_REASON_SL=4, DEAL_ENTRY_OUT=1, DEAL_ENTRY_INOUT=2)
    deals = [
        SimpleNamespace(
            ticket=1,
            magic=runtime.magic_number,
            symbol=runtime.symbol,
            reason=3,
            entry=1,
            time=1_700_000_000,
            time_msc=1_700_000_000_000,
        ),
        SimpleNamespace(
            ticket=2,
            magic=runtime.magic_number,
            symbol=runtime.symbol,
            reason=4,
            entry=1,
            time=1_700_000_100,
            time_msc=1_700_000_100_000,
        ),
        SimpleNamespace(
            ticket=3,
            magic=runtime.magic_number,
            symbol=runtime.symbol,
            reason=4,
            entry=1,
            time=1_700_000_200,
            time_msc=1_700_000_200_000,
        ),
        SimpleNamespace(
            ticket=4,
            magic=999,
            symbol=runtime.symbol,
            reason=4,
            entry=1,
            time=1_700_000_300,
            time_msc=1_700_000_300_000,
        ),
    ]

    latest = _latest_managed_stop_loss_deal(deals, runtime, fake_mt5)

    assert latest is not None
    assert latest.ticket == 3


def test_tick_age_seconds_is_clamped_at_zero_for_future_broker_time():
    now = pd.Timestamp("2026-04-08 10:00:00", tz="UTC")
    tick = SimpleNamespace(time_msc=int(pd.Timestamp("2026-04-08 10:05:00", tz="UTC").timestamp() * 1000))

    age = _tick_age_seconds(tick, now)

    assert age == 0.0


def test_margin_cap_usd_uses_equity_percentage():
    cap = _margin_cap_usd(200000.0, 0.175)

    assert cap == pytest.approx(35000.0)


def test_ensure_margin_within_cap_raises_when_exceeded():
    with pytest.raises(RuntimeError, match="Estimated margin exceeds configured cap"):
        _ensure_margin_within_cap(36000.0, 35000.0, 0.175)
