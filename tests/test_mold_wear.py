"""合模力退化跟踪：基线、防误报、迟滞换模窗口、设定档与换模重置。"""

from __future__ import annotations

import pytest

from glassmelt.errors import InterlockError
from glassmelt.mold.wear import DEFAULT_STATION, ClampWearMonitor


@pytest.fixture()
def wear(platform) -> ClampWearMonitor:
    return platform.wear


def _shots(platform, count: int, force_kn: float) -> None:
    """按给定峰值力走 count 模合模→脱模。"""

    for _ in range(count):
        platform.closer.close(force_kn)
        platform.closer.open_mold("脱模")


def _settle(platform, force_kn: float) -> None:
    """跑过跑合期并锁好基线。"""

    limits = platform.config.mold_wear
    platform.closer.adjust_setpoint(force_kn)
    _shots(platform, limits.warmup_shots, force_kn)
    _shots(platform, limits.baseline_shots, force_kn)


def test_fresh_mold_starts_in_warmup_and_collects_no_degradation(platform, wear):
    _shots(platform, 3, 47.0)
    state = wear.last_state()
    assert state.phase == "warming"
    assert state.alarm is False
    assert state.baseline_kn == pytest.approx(0.0)
    assert state.shots_since_change == 3


def test_baseline_requires_enough_shots(platform, wear):
    limits = platform.config.mold_wear
    platform.closer.adjust_setpoint(50.0)
    _shots(platform, limits.warmup_shots, 50.0)
    _shots(platform, limits.baseline_shots - 1, 50.0)
    assert wear.last_state().phase == "learning"
    assert wear.last_state().baseline_shots == limits.baseline_shots - 1
    _shots(platform, 1, 50.0)
    state = wear.last_state()
    assert state.phase == "healthy"
    assert state.baseline_kn == pytest.approx(50.0)


def test_low_force_during_warmup_or_learning_never_alarms(platform, wear):
    limits = platform.config.mold_wear
    # 刚换模即便力很低也只在跑合/学习，绝不误报换模窗口。
    _shots(platform, limits.warmup_shots + limits.baseline_shots - 1, 38.0)
    assert wear.last_state().phase in ("warming", "learning")
    assert wear.last_state().alarm is False


def test_wear_percent_uses_baseline_to_critical_span(platform, wear):
    limits = platform.config.mold_wear
    base = 50.0
    _settle(platform, base)
    force = base - (base - limits.critical_force_kn) * 0.6
    _shots(platform, limits.confirm_shots, force)
    state = wear.last_state()
    assert state.phase == "watch"
    assert state.wear_pct == pytest.approx(60.0, abs=0.5)
    assert state.remaining_pct == pytest.approx(40.0, abs=0.5)


def test_single_force_drop_does_not_open_window_hysteresis(platform, wear):
    limits = platform.config.mold_wear
    _settle(platform, 50.0)
    crash = 50.0 - (50.0 - limits.critical_force_kn) * 0.9
    _shots(platform, 1, crash)
    state = wear.last_state()
    assert state.phase == "healthy"
    assert state.confirming == 1
    assert state.alarm is False


def test_window_opens_after_consecutive_confirmations(platform, wear):
    limits = platform.config.mold_wear
    _settle(platform, 50.0)
    crash = 50.0 - (50.0 - limits.critical_force_kn) * 0.9
    _shots(platform, limits.confirm_shots, crash)
    state = wear.last_state()
    assert state.phase == "window"
    assert state.alarm is True
    assert state.wear_pct >= limits.window_wear_pct


def test_critical_phase_is_reached_below_seal_force(platform, wear):
    limits = platform.config.mold_wear
    _settle(platform, 50.0)
    _shots(platform, limits.confirm_shots, limits.critical_force_kn - 1.0)
    state = wear.last_state()
    assert state.phase == "critical"
    assert state.wear_pct >= 100.0
    assert state.alarm is True


def test_recovery_steps_down_immediately(platform, wear):
    limits = platform.config.mold_wear
    _settle(platform, 50.0)
    crash = 50.0 - (50.0 - limits.critical_force_kn) * 0.9
    _shots(platform, limits.confirm_shots, crash)
    assert wear.last_state().phase == "window"
    _shots(platform, 1, 50.0)
    assert wear.last_state().phase == "healthy"


def test_setpoint_change_rebaselines_without_false_wear(platform, wear):
    limits = platform.config.mold_wear
    _settle(platform, 50.0)
    platform.closer.adjust_setpoint(60.0)
    _shots(platform, limits.baseline_shots, 60.0)
    state = wear.last_state()
    assert state.phase == "healthy"
    assert state.baseline_kn == pytest.approx(60.0)
    assert state.wear_pct == pytest.approx(0.0)
    assert state.alarm is False


def test_small_force_slip_with_same_setpoint_is_still_wear(platform, wear):
    # 设定档没变、力往下掉就是磨损，不能被调机逻辑掩盖。
    limits = platform.config.mold_wear
    _settle(platform, 50.0)
    slip = 50.0 - (50.0 - limits.critical_force_kn) * 0.9
    _shots(platform, limits.confirm_shots, slip)
    assert wear.last_state().phase == "window"


def test_setpoint_below_close_minimum_is_rejected(platform):
    with pytest.raises(InterlockError):
        platform.closer.adjust_setpoint(platform.config.mold.close_min_force_kn - 1.0)


def test_change_mold_resets_baseline_and_epoch(platform, wear):
    _settle(platform, 50.0)
    assert wear.last_state().change_epoch == 1
    report = platform.change_mold("计划换模")
    assert report["station"] == DEFAULT_STATION
    state = wear.last_state()
    assert state.phase == "warming"
    assert state.change_epoch == 2
    assert state.shots_since_change == 0
    assert state.baseline_kn == pytest.approx(0.0)


def test_stations_are_tracked_independently(wear):
    wear.observe(60.0, station="section-2", setpoint_kn=60.0)
    a = wear.last_state(DEFAULT_STATION)
    b = wear.last_state("section-2")
    assert a is None
    assert b.station == "section-2"
    assert b.phase == "warming"
    wear.reset(DEFAULT_STATION, reason="换模")
    assert wear.last_state("section-2").change_epoch == 1
    assert wear.last_state(DEFAULT_STATION).change_epoch == 1


def test_wear_audit_emits_window_warning(platform):
    limits = platform.config.mold_wear
    _settle(platform, 50.0)
    crash = 50.0 - (50.0 - limits.critical_force_kn) * 0.9
    _shots(platform, limits.confirm_shots, crash)
    records = [
        item
        for item in platform.audit.entries(500)
        if item["subject"] == "clamp_wear" and item["action"] == "clamp_wear_updated"
    ]
    assert any(item["severity"] == "warning" for item in records)


def test_wear_state_is_exposed_in_status_and_readiness(running):
    payload = running.status()["mold"]["wear"]
    assert payload["critical_force_kn"] == running.config.mold_wear.critical_force_kn
    assert DEFAULT_STATION in payload["stations"]
    assert "mold_wear" in running.readiness_report()
