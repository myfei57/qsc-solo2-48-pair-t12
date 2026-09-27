"""模具磨损跟踪：认基线、认线期抑制、退化外推、换模窗口与换模复位。"""

from __future__ import annotations

import copy

import pytest

from glassmelt.config import PlatformConfig, config_to_mapping, default_config
from glassmelt.errors import ConfigError
from glassmelt.mold.wear import (
    CRITICAL,
    HEALTHY,
    LEARNING,
    REPLACE_WINDOW,
    SETTLING,
    WATCH,
    MoldWearMonitor,
)

FRESH_KN = 47.04
MIN_KN = 42.0
BUDGET_KN = FRESH_KN - MIN_KN


def _monitor(platform, machine="tank1"):
    return MoldWearMonitor(
        platform.clock,
        platform.bus,
        platform.config.mold,
        platform.config.wear,
        machine=machine,
    )


def _run(monitor, forces, closed=True):
    return [monitor.observe(force, closed=closed) for force in forces]


def test_learning_until_baseline_then_settling_until_min_samples(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    assessments = _run(monitor, [FRESH_KN] * (wear.min_samples - 1))
    assert all(item.suppressed for item in assessments)
    assert assessments[0].state == LEARNING
    assert assessments[0].suppress_reason == "insufficient_samples"
    assert assessments[wear.baseline_samples - 1].state == SETTLING
    assert assessments[wear.baseline_samples - 1].suppress_reason == "fresh_mold_settling"
    assert monitor.snapshot()["window_open"] is False


def test_stable_force_stays_healthy_with_no_remaining_cycle_forecast(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    _run(monitor, [FRESH_KN] * (wear.min_samples + 10))
    assessment = monitor.last_assessment()
    assert assessment.state == HEALTHY
    assert assessment.suppressed is False
    assert assessment.wear_fraction == pytest.approx(0.0, abs=1e-6)
    assert assessment.remaining_cycles is None
    assert assessment.slope_kn_per_cycle == pytest.approx(0.0)
    assert assessment.last_force_kn == pytest.approx(FRESH_KN)


def test_baseline_is_learned_per_machine_from_its_own_history(platform):
    wear = platform.config.wear
    monitor_a = _monitor(platform, machine="tank1")
    _run(monitor_a, [46.0] * (wear.min_samples + 5))
    assert monitor_a.last_assessment().fresh_force_kn == pytest.approx(46.0)

    monitor_b = _monitor(platform, machine="tank2")
    _run(monitor_b, [50.0] * (wear.min_samples + 5))
    assert monitor_b.last_assessment().fresh_force_kn == pytest.approx(50.0)
    assert monitor_b.machine == "tank2"


def test_slow_decline_forecasts_remaining_and_opens_replace_window(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    # 头 min_samples 模认线，之后每模固定跌 0.01 kN：跌幅慢到中位数还
    # 远没到窗口线，但外推剩余模数已进入提前量——这才是"提前给窗口"。
    forces = [FRESH_KN] * wear.min_samples + [
        FRESH_KN - 0.01 * (index + 1)
        for index in range(45)
    ]
    assessments = _run(monitor, forces)
    window_events = [
        event for event in platform.bus.history("mold") if event.action == "mold_wear_window"
    ]
    opened = next(item for item in assessments if item.window_open)
    assert opened.state == REPLACE_WINDOW
    assert opened.remaining_cycles is not None
    assert opened.remaining_cycles <= wear.lead_margin_cycles
    assert opened.wear_fraction < wear.window_fraction
    assert window_events and window_events[0].payload["severity"] == "warning"


def test_replace_window_latches_through_single_shots_of_noise(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    forces = [FRESH_KN] * wear.min_samples + [
        FRESH_KN - 0.01 * (index + 1) for index in range(45)
    ]
    _run(monitor, forces)
    assert monitor.last_assessment().window_open is True
    # 哪怕这几模力回到新鲜值，中位数窗口里仍是退化数据，窗口必须闩住。
    for _ in range(5):
        assert monitor.observe(FRESH_KN, closed=True).window_open is True
    assert monitor.last_assessment().state == REPLACE_WINDOW


def test_window_clears_only_after_sustained_recovery(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    forces = [FRESH_KN] * wear.min_samples + [
        FRESH_KN - 0.01 * (index + 1) for index in range(43)
    ]
    _run(monitor, forces)
    assert monitor.last_assessment().window_open is True
    # 中位数窗口先被新鲜样本洗出退化段，之后还要连续 clear_samples 模健康
    # 才解除闩锁；中途任何一拍重新变差，计数作废。
    shots = 0
    while monitor.last_assessment().window_open:
        shots += 1
        monitor.observe(FRESH_KN, closed=True)
        assert shots < 60
    assert monitor.last_assessment().state == HEALTHY


def test_single_shot_force_collapse_alarms_critical_immediately(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    _run(monitor, [FRESH_KN] * wear.min_samples)
    critical_level = MIN_KN + BUDGET_KN * wear.critical_fraction
    assessment = monitor.observe(critical_level - 0.1, closed=True)
    assert assessment.state == CRITICAL
    assert assessment.window_open is True
    actions = [event.action for event in platform.bus.history("mold")]
    assert "mold_wear_critical" in actions


def test_persistent_low_force_alarms_critical_on_median(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    low = MIN_KN + BUDGET_KN * wear.critical_fraction - 0.2
    forces = [FRESH_KN] * wear.min_samples + [low] * wear.trend_window
    _run(monitor, forces)
    assert monitor.last_assessment().state == CRITICAL


def test_noisy_non_trending_force_does_not_forecast_or_warn(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    pattern = [FRESH_KN + 0.3, FRESH_KN - 0.3]
    forces = [pattern[index % 2] for index in range(wear.min_samples + 20)]
    _run(monitor, forces)
    assessment = monitor.last_assessment()
    assert assessment.remaining_cycles is None
    assert assessment.state in (HEALTHY, WATCH)
    assert monitor.snapshot()["window_open"] is False


def test_replace_mold_resets_life_and_baseline_without_false_alarm(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    forces = [FRESH_KN] * wear.min_samples + [
        FRESH_KN - 0.01 * (index + 1) for index in range(45)
    ]
    _run(monitor, forces)
    assert monitor.last_assessment().window_open is True

    reset = monitor.replace_mold("分模缝跑料换模")
    assert reset.mold_life == 2
    assert reset.state == LEARNING
    assert reset.suppressed is True
    assert reset.window_open is False
    assert reset.samples == 0

    fresh = _run(monitor, [48.5] * (wear.min_samples + 5))
    assert fresh[-1].state == HEALTHY
    assert fresh[-1].mold_life == 2
    assert fresh[-1].fresh_force_kn == pytest.approx(48.5)

    replaced = [event for event in platform.bus.history("mold") if event.action == "mold_replaced"]
    assert replaced and replaced[-1].payload["mold_life"] == 2


def test_short_close_is_counted_but_not_used_as_a_force_sample(platform):
    monitor = _monitor(platform)
    wear = platform.config.wear
    _run(monitor, [FRESH_KN] * 8, closed=True)
    short = monitor.observe(MIN_KN - 1.0, closed=False)
    assert short.samples == 8
    assert short.short_closes == 1
    _run(monitor, [FRESH_KN] * (wear.min_samples - 8), closed=True)
    assessment = monitor.last_assessment()
    assert assessment.samples == wear.min_samples
    assert assessment.short_closes == 1


def test_platform_collects_force_samples_from_close_events(running):
    wear = running.config.wear
    # 启动顺序本身走过一遍成型循环，已经留下 1 个样本。
    for _ in range(wear.min_samples + 4):
        running.blow_mold(path="cycle")
    assessment = running.wear.last_assessment()
    assert assessment is not None
    assert assessment.samples == wear.min_samples + 5
    assert assessment.machine == running.namespaces.unit
    assert assessment.state == HEALTHY
    assert running.status()["mold"]["wear"]["assessment"]["state"] == HEALTHY


def test_platform_replace_mold_action_resets_tracking(running):
    payload = running.replace_mold("测试换模")
    assert payload["wear"]["mold_life"] == 2
    assert payload["assessment"]["state"] == LEARNING
    assert running.readiness_report()["mold_wear"]["mold_life"] == 2


def test_console_endpoint_replaces_mold(console):
    from tests.test_console import _get, _post

    _server, base = console
    status, payload = _post(f"{base}/api/mold/replace", {"reason": "界面换模"})
    assert status == 200
    assert payload["ok"] is True
    assert payload["result"]["wear"]["mold_life"] == 2
    state_status, state = _get(f"{base}/api/state")
    assert state_status == 200
    assert "/api/mold/replace" in state["endpoints"]["api"]


def test_wear_config_rejects_inverted_fractions_and_sample_order():
    mapping = config_to_mapping(default_config())
    broken = copy.deepcopy(mapping)
    broken["limits"]["mold_wear"]["window_fraction"] = 0.9
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(broken)
    broken_order = copy.deepcopy(mapping)
    broken_order["limits"]["mold_wear"]["min_samples"] = 4
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(broken_order)
