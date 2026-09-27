"""模具磨损跟踪与换模窗口。

合模只有"到位／没到位"一个判据，分模缝跑料是终局，不是预警。本模块按每
台机自己的合模力历史给模具做寿命管理，不跨模具、不跨机台拼数据：

1. **认基线**：换模（或首用）后的头 ``baseline_samples`` 模只取样、不判
   退化，取这段合模力的中位数当作这套模具的"新鲜合模力"；
2. **认线期抑制**：样本不足 ``min_samples`` 之前一律只报 learning／
   settling。刚换上的模具合模力本来就会飘，这个阶段报警等于误报；
3. **退化度量**：基线认稳后，最近一个趋势窗口的中位合模力相对基线的跌
   幅占"基线 → 合模力下限"预算的比例，就是磨损程度；
4. **窗口外推**：窗口内用 Theil–Sen 中位斜率估每模跌多少。负斜率点对不
   够多（分不清趋势和噪声）时不预报剩余模数；趋势明确时外推跌到换模窗
   口力还剩多少模；
5. **换模窗口**：磨损比例连续 ``confirm_samples`` 模达到窗口线、或外推
   剩余模数小于提前量，窗口打开并闩住，直到现场换模、或连续
   ``clear_samples`` 模明显恢复才解除。中位数跌破临界线直接报 critical。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from statistics import median

from ..config import MoldLimits, WearLimits
from ..eventbus import Event, EventBus

#: 抑制/退化状态。
LEARNING = "learning"
SETTLING = "settling"
HEALTHY = "healthy"
WATCH = "watch"
REPLACE_WINDOW = "replace_window"
CRITICAL = "critical"

#: Theil–Sen 点对中负斜率占比至少多少，才承认这是一条下行趋势。
NEGATIVE_SLOPE_CONFIDENCE = 0.6


@dataclass(frozen=True)
class WearAssessment:
    """一次模具磨损评估（每模合模后产出一拍）。"""

    machine: str
    mold_life: int
    state: str
    suppressed: bool
    suppress_reason: str | None
    samples: int
    short_closes: int
    fresh_force_kn: float | None
    recent_force_kn: float | None
    last_force_kn: float | None
    force_drop_kn: float | None
    wear_fraction: float | None
    slope_kn_per_cycle: float | None
    remaining_cycles: int | None
    window_open: bool
    at: float

    def as_dict(self) -> dict:
        return {
            "machine": self.machine,
            "mold_life": self.mold_life,
            "state": self.state,
            "suppressed": self.suppressed,
            "suppress_reason": self.suppress_reason,
            "samples": self.samples,
            "short_closes": self.short_closes,
            "fresh_force_kn": _round(self.fresh_force_kn),
            "recent_force_kn": _round(self.recent_force_kn),
            "last_force_kn": _round(self.last_force_kn),
            "force_drop_kn": _round(self.force_drop_kn),
            "wear_fraction": _round(self.wear_fraction),
            "slope_kn_per_cycle": _round(self.slope_kn_per_cycle),
            "remaining_cycles": self.remaining_cycles,
            "window_open": self.window_open,
            "at": round(self.at, 3),
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


class MoldWearMonitor:
    """按一台机的合模力序列跟踪当前模具的磨损。"""

    def __init__(
        self,
        clock,
        bus: EventBus,
        limits: MoldLimits,
        wear: WearLimits,
        machine: str,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._limits = limits
        self._wear = wear
        self._machine = machine
        self._forces: list[float] = []
        self._mold_life = 1
        self._short_closes = 0
        self._fresh_force_kn: float | None = None
        self._toward_window = 0
        self._healthy_streak = 0
        self._window_open = False
        self._last: WearAssessment | None = None
        self._lock = threading.RLock()
        self._unsubscribe = bus.subscribe("mold", self._on_event)

    @property
    def machine(self) -> str:
        return self._machine

    def _on_event(self, event: Event) -> None:
        """订阅合模事件：每模合模力都进同一套模具的寿命样本。"""

        if event.component != "mold" or event.payload.get("subject") != "mold_closer":
            return
        if event.action not in ("mold_closed", "mold_close_short"):
            return
        force = event.payload.get("force_kn")
        if force is None:
            return
        self.observe(float(force), closed=event.action == "mold_closed")

    def observe(self, force_kn: float, closed: bool) -> WearAssessment:
        """落一样本并现场评估磨损状态。"""

        moment = self._clock.now()
        with self._lock:
            if not closed:
                self._short_closes += 1
            else:
                self._forces.append(float(force_kn))
                if (
                    self._fresh_force_kn is None
                    and len(self._forces) >= self._wear.baseline_samples
                ):
                    self._fresh_force_kn = float(
                        median(self._forces[: self._wear.baseline_samples])
                    )
            assessment = self._assess(moment)
            self._last = assessment
        self._publish(assessment)
        return assessment

    def replace_mold(self, reason: str) -> WearAssessment:
        """现场换模：作废基线与全部样本，新模具从认线期重新起步。"""

        moment = self._clock.now()
        with self._lock:
            self._forces = []
            self._fresh_force_kn = None
            self._short_closes = 0
            self._toward_window = 0
            self._healthy_streak = 0
            self._window_open = False
            self._mold_life += 1
            assessment = self._assess(moment)
            self._last = assessment
        self._bus.publish(
            Event.of(
                "mold",
                "mold",
                "mold_replaced",
                subject="mold_wear",
                severity="notice",
                machine=self._machine,
                mold_life=assessment.mold_life,
                reason=reason,
                at=round(moment, 3),
            )
        )
        self._publish(assessment)
        return assessment

    def last_assessment(self) -> WearAssessment | None:
        """返回最近一次磨损评估（仅供展示与审计）。"""

        with self._lock:
            return self._last

    def snapshot(self) -> dict:
        """磨损跟踪快照：最新评估、样本量与阈值。"""

        with self._lock:
            return {
                "machine": self._machine,
                "mold_life": self._mold_life,
                "samples": len(self._forces),
                "short_closes": self._short_closes,
                "window_open": self._window_open,
                "min_force_kn": self._limits.close_min_force_kn,
                "window_fraction": self._wear.window_fraction,
                "critical_fraction": self._wear.critical_fraction,
                "lead_margin_cycles": self._wear.lead_margin_cycles,
                "assessment": self._last.as_dict() if self._last else None,
            }

    def _assess(self, moment: float) -> WearAssessment:
        samples = len(self._forces)
        if samples < self._wear.baseline_samples:
            return self._assessment(
                LEARNING,
                moment,
                suppressed=True,
                reason="insufficient_samples",
            )
        if samples < self._wear.min_samples:
            return self._assessment(
                SETTLING,
                moment,
                suppressed=True,
                reason="fresh_mold_settling",
            )

        fresh = float(self._fresh_force_kn)
        last = self._forces[-1]
        recent = float(median(self._forces[-self._wear.trend_window :]))
        drop = fresh - recent
        budget = fresh - self._limits.close_min_force_kn
        fraction = max(0.0, drop / budget) if budget > 0 else 0.0
        slope, confident = self._trend_slope()
        window_level = self._limits.close_min_force_kn + budget * self._wear.window_fraction
        critical_level = self._limits.close_min_force_kn + budget * self._wear.critical_fraction
        remaining: int | None = None
        if confident and slope < 0.0:
            # 剩余模数外推到换模窗口力，提前量才有意义；临界由中位数直判。
            headroom = recent - window_level
            if headroom <= 0.0:
                remaining = 0
            else:
                remaining = max(0, int(round(headroom / (-slope))))

        # 中位数看持续磨损；当拍力一旦已贴近合模力下限，说明模具这一炮就
        # 快压不住了，不能等中位数慢慢爬下来再报 critical。
        if fraction >= self._wear.critical_fraction or (
            recent <= critical_level or last <= critical_level
        ):
            raw = CRITICAL
        elif fraction >= self._wear.window_fraction or (
            remaining is not None and remaining <= self._wear.lead_margin_cycles
        ):
            raw = REPLACE_WINDOW
        elif fraction >= self._wear.watch_fraction:
            raw = WATCH
        else:
            raw = HEALTHY

        was_open = self._window_open
        self._update_latch(raw)
        if not was_open and self._window_open:
            self._bus.publish(
                Event.of(
                    "mold",
                    "mold",
                    "mold_wear_window",
                    subject="mold_wear",
                    severity="warning",
                    machine=self._machine,
                    mold_life=self._mold_life,
                    wear_fraction=round(fraction, 4),
                    remaining_cycles=remaining,
                    samples=samples,
                )
            )

        if raw == CRITICAL:
            state = CRITICAL
        elif self._window_open:
            state = REPLACE_WINDOW
        else:
            state = raw
        return self._assessment(
            state,
            moment,
            fresh=fresh,
            recent=recent,
            last=last,
            drop=drop,
            fraction=fraction,
            slope=slope,
            remaining=remaining,
        )

    def _update_latch(self, raw: str) -> None:
        """换模窗口闩锁：临界立即开窗，窗口线需连续确认，连续恢复才关。"""

        if raw == CRITICAL:
            # 临界意味着这一炮就可能压不住，不能等连续确认，立即开窗。
            self._toward_window += 1
            self._healthy_streak = 0
            self._window_open = True
        elif raw == REPLACE_WINDOW:
            self._toward_window += 1
            self._healthy_streak = 0
            if self._toward_window >= self._wear.confirm_samples:
                self._window_open = True
        else:
            self._toward_window = 0
            self._healthy_streak += 1
            if self._window_open and self._healthy_streak >= self._wear.clear_samples:
                self._window_open = False

    def _trend_slope(self) -> tuple[float, bool]:
        """窗口内 Theil–Sen 中位斜率（kN/模）与下行可信度。"""

        window = self._forces[-self._wear.trend_window :]
        if len(window) < 2:
            return 0.0, False
        slopes: list[float] = []
        for i in range(len(window) - 1):
            for j in range(i + 1, len(window)):
                slopes.append((window[j] - window[i]) / (j - i))
        value = float(median(slopes))
        negative = sum(1 for item in slopes if item < 0.0) / len(slopes)
        return value, negative >= NEGATIVE_SLOPE_CONFIDENCE

    def _assessment(
        self,
        state: str,
        moment: float,
        *,
        suppressed: bool = False,
        reason: str | None = None,
        fresh: float | None = None,
        recent: float | None = None,
        last: float | None = None,
        drop: float | None = None,
        fraction: float | None = None,
        slope: float | None = None,
        remaining: int | None = None,
    ) -> WearAssessment:
        return WearAssessment(
            machine=self._machine,
            mold_life=self._mold_life,
            state=state,
            suppressed=suppressed,
            suppress_reason=reason,
            samples=len(self._forces),
            short_closes=self._short_closes,
            fresh_force_kn=fresh,
            recent_force_kn=recent,
            last_force_kn=last,
            force_drop_kn=drop,
            wear_fraction=fraction,
            slope_kn_per_cycle=slope,
            remaining_cycles=remaining,
            window_open=self._window_open,
            at=moment,
        )

    def _publish(self, assessment: WearAssessment) -> None:
        severity = {
            CRITICAL: "alarm",
            REPLACE_WINDOW: "warning",
            WATCH: "notice",
        }.get(assessment.state, "info")
        event = Event.of(
            "mold",
            "mold",
            "mold_wear_evaluated",
            subject="mold_wear",
            severity=severity,
            **{
                key: value
                for key, value in assessment.as_dict().items()
                if key not in ("machine", "at")
            },
        )
        self._bus.publish(event)
        if assessment.state == CRITICAL:
            self._bus.publish(
                Event.of(
                    "mold",
                    "mold",
                    "mold_wear_critical",
                    subject="mold_wear",
                    severity="alarm",
                    machine=self._machine,
                    mold_life=assessment.mold_life,
                    wear_fraction=_round(assessment.wear_fraction),
                    recent_force_kn=_round(assessment.recent_force_kn),
                )
            )
