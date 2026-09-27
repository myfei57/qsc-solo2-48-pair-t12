"""合模力退化跟踪与换模窗口。

合模到位只回答"这一炮能不能吹"，回答不了"模具还能用多久"。分模面磨损、
铰链间隙变大都会让同一档合模设定下的峰值合模力逐模下滑，等力跌破密封力、
分模缝开始跑料才停机，属于已经打废了才知道。退化监视器每模记一个峰值合
模力，按"换模后基线力 → 临界密封力"折算退化百分比，在真正跑料之前给出
换模窗口。

误报必须压住，否则窗口喊得比换模还勤就没人信了，因此有四道闸：

1. **跑合期**：刚换模的头几模贴合面还在磨合，力自己会变，这一段一律
   ``warming``，不下退化结论也不攒基线；
2. **基线样本**：跑合过后要攒够一窗同设定档样本才锁基线，样本不够只是
   ``learning``，不拿三五模的均值当基准；
3. **确认模次**：单次力掉不立刻升级，连续若干模落在更差档才确认，吸收
   料温、料重造成的逐模抖动（恶化带迟滞，恢复即时降档）；
4. **设定档偏移**：合模设定档被人调过，峰值力整档平移是工况变化不是磨损。
   判定只认合模机构随事件带出的**设定档**：设定相对基线设定偏移超阈，本窗
   冻结退化结论、按新档重新攒基线。绝不能拿"力突然掉了一档"去猜调机——
   真实的快速磨损也长那样，猜了就会在低力上重新锁基线，把老化掩盖成正常。

每台机（工位）各算各的基线与退化，一台机换模不能冲掉别的机台的模型。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque

from ..config import MoldWearLimits
from ..eventbus import Event, EventBus

#: 默认工位：平台当前只驱动一个合模机构，多工位时由事件里的 station 区分。
DEFAULT_STATION = "section-1"

#: 退化状态由轻到重的次序，用于迟滞比较与快照排序。
PHASES = ("warming", "learning", "healthy", "watch", "window", "critical")

#: 每一次合模都贡献峰值力样本：临界退化段的力会低到当次判不到位，那正是
#: 要提前预警的区段，不能只收"到位"的一炮而把临界带漏掉。
_CLOSE_ACTIONS = ("mold_closed", "mold_close_short")

_RATED_PHASES = ("healthy", "watch", "window", "critical")


@dataclass(frozen=True)
class WearState:
    """一次合模力退化评估。"""

    station: str
    phase: str
    wear_pct: float
    baseline_kn: float
    baseline_ready: bool
    force_kn: float
    setpoint_kn: float
    critical_kn: float
    remaining_pct: float
    shots_since_change: int
    baseline_shots: int
    confirming: int
    frozen: bool
    change_epoch: int
    at: float

    @property
    def alarm(self) -> bool:
        """是否已进入需要安排换模的区段（窗口或临界）。"""

        return self.phase in ("window", "critical")

    def as_dict(self) -> dict:
        return {
            "station": self.station,
            "phase": self.phase,
            "wear_pct": round(self.wear_pct, 2),
            "baseline_kn": round(self.baseline_kn, 3),
            "baseline_ready": self.baseline_ready,
            "force_kn": round(self.force_kn, 3),
            "setpoint_kn": round(self.setpoint_kn, 3),
            "critical_kn": round(self.critical_kn, 3),
            "remaining_pct": round(self.remaining_pct, 2),
            "shots_since_change": self.shots_since_change,
            "baseline_shots": self.baseline_shots,
            "confirming": self.confirming,
            "frozen": self.frozen,
            "change_epoch": self.change_epoch,
            "alarm": self.alarm,
            "at": round(self.at, 3),
        }


class _StationModel:
    """单个工位的基线窗、设定档、确认计数与迟滞状态。"""

    def __init__(self, limits: MoldWearLimits, nominal_kn: float, change_epoch: int) -> None:
        self._limits = limits
        self._nominal_kn = nominal_kn
        self.change_epoch = change_epoch
        self.shots = 0
        self.samples: Deque[float] = deque()
        self.window_setpoint_kn: float | None = None
        self.baseline_kn: float | None = None
        self.baseline_setpoint_kn: float | None = None
        self.force_kn: float | None = None
        self.setpoint_kn = nominal_kn
        self.phase = "warming"
        self.candidate_phase: str | None = None
        self.confirming = 0

    def setpoint_shifted(self) -> bool:
        """当前设定档相对已锁基线设定是否偏移超阈（没锁基线不算偏移）。"""

        if self.baseline_setpoint_kn is None:
            return False
        return (
            abs(self.setpoint_kn - self.baseline_setpoint_kn)
            / self.baseline_setpoint_kn
            >= self._limits.setpoint_shift_pct / 100.0
        )


class ClampWearMonitor:
    """按工位跟踪峰值合模力退化并给出换模窗口。"""

    def __init__(
        self,
        clock,
        bus: EventBus,
        limits: MoldWearLimits,
        nominal_close_kn: float,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._limits = limits
        self._nominal_close_kn = float(nominal_close_kn)
        self._stations: dict[str, _StationModel] = {}
        self._changes = 0
        self._last: dict[str, WearState] = {}
        self._lock = threading.RLock()
        bus.subscribe("mold", self._on_event)

    # ---- 事件入口 -------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        if event.action in _CLOSE_ACTIONS:
            force = event.payload.get("force_kn")
            if isinstance(force, (int, float)) and not isinstance(force, bool):
                setpoint = event.payload.get("setpoint_kn", self._nominal_close_kn)
                if not isinstance(setpoint, (int, float)) or isinstance(setpoint, bool):
                    setpoint = self._nominal_close_kn
                self.observe(
                    float(force),
                    station=str(event.payload.get("station", DEFAULT_STATION)),
                    setpoint_kn=float(setpoint),
                )
        elif event.action == "mold_changed":
            if event.payload.get("origin") == "clamp_wear":
                return
            self.reset(
                str(event.payload.get("station", DEFAULT_STATION)),
                reason=str(event.payload.get("reason", "总线换模事件")),
            )

    # ---- 核心评估 -------------------------------------------------------

    def observe(
        self,
        force_kn: float,
        station: str = DEFAULT_STATION,
        setpoint_kn: float | None = None,
    ) -> WearState:
        """记录一个工位本模的峰值合模力与当前设定档，现场重新评估退化。"""

        moment = self._clock.now()
        force = float(force_kn)
        with self._lock:
            model = self._model(station)
            model.shots += 1
            model.force_kn = force
            model.setpoint_kn = (
                self._nominal_close_kn if setpoint_kn is None else float(setpoint_kn)
            )

            if self._in_warmup(model):
                phase = "warming"
            elif self._follow_setpoint_shift(model):
                phase = "learning"
            elif model.baseline_kn is None:
                phase = self._collect_baseline(model, force)
            else:
                phase = self._evaluate_wear(model, force)

            frozen = model.setpoint_shifted()
            baseline_ready = model.baseline_kn is not None and not frozen
            baseline = model.baseline_kn if baseline_ready else 0.0
            wear = 0.0 if not baseline_ready else self._wear_pct(model.baseline_kn, force)
            state = WearState(
                station=station,
                phase=phase,
                wear_pct=max(0.0, wear),
                baseline_kn=baseline,
                baseline_ready=baseline_ready,
                force_kn=force,
                setpoint_kn=model.setpoint_kn,
                critical_kn=self._limits.critical_force_kn,
                remaining_pct=max(0.0, 100.0 - max(0.0, wear)),
                shots_since_change=model.shots,
                baseline_shots=len(model.samples),
                confirming=model.confirming,
                frozen=frozen,
                change_epoch=model.change_epoch,
                at=moment,
            )
            self._last[station] = state
        self._publish(state)
        return state

    def _in_warmup(self, model: _StationModel) -> bool:
        if model.shots <= self._limits.warmup_shots:
            # 跑合期内连样本都不攒，避免把打磨段的力带进基线。
            model.samples.clear()
            model.window_setpoint_kn = None
            model.baseline_kn = None
            model.baseline_setpoint_kn = None
            model.candidate_phase = None
            model.confirming = 0
            return True
        return False

    def _follow_setpoint_shift(self, model: _StationModel) -> bool:
        """设定档偏移超阈时冻结退化结论，按新档重新攒一窗基线。"""

        if not model.setpoint_shifted():
            return False
        if not model.samples or model.window_setpoint_kn != model.setpoint_kn:
            model.samples.clear()
            model.window_setpoint_kn = model.setpoint_kn
            model.baseline_kn = None
            model.baseline_setpoint_kn = None
        model.candidate_phase = None
        model.confirming = 0
        # 在新档攒够样本后立即重锁基线，随后恢复退化评估。
        self._collect_baseline(model, model.force_kn if model.force_kn is not None else 0.0)
        return True

    def _collect_baseline(self, model: _StationModel, force: float) -> str:
        """同设定档下攒基线；窗口中途换档则整窗作废重来。"""

        if model.window_setpoint_kn is None:
            model.window_setpoint_kn = model.setpoint_kn
        elif model.window_setpoint_kn != model.setpoint_kn:
            model.samples.clear()
            model.window_setpoint_kn = model.setpoint_kn
        model.samples.append(force)
        if len(model.samples) >= self._limits.baseline_shots:
            model.baseline_kn = sum(model.samples) / len(model.samples)
            model.baseline_setpoint_kn = model.window_setpoint_kn
            model.phase = "healthy"
            return "healthy"
        return "learning"

    def _evaluate_wear(self, model: _StationModel, force: float) -> str:
        """按退化百分比选档，并用连续确认模次做迟滞。"""

        target = self._phase_for(self._wear_pct(model.baseline_kn, force))
        current = model.phase if model.phase in _RATED_PHASES else "healthy"
        rank = PHASES.index(target)

        if target == current:
            model.candidate_phase = None
            model.confirming = 0
        elif rank > PHASES.index(current):
            # 恶化方向：要连续确认，挡住单模抖动。
            if model.candidate_phase == target:
                model.confirming += 1
            else:
                model.candidate_phase = target
                model.confirming = 1
            if model.confirming >= self._limits.confirm_shots:
                current = target
                model.candidate_phase = None
                model.confirming = 0
        else:
            # 恢复方向：力回来了直接降档。
            current = target
            model.candidate_phase = None
            model.confirming = 0

        model.phase = current
        return current

    def _phase_for(self, wear_pct: float) -> str:
        if wear_pct >= 100.0:
            return "critical"
        if wear_pct >= self._limits.window_wear_pct:
            return "window"
        if wear_pct >= self._limits.watch_wear_pct:
            return "watch"
        return "healthy"

    def _wear_pct(self, baseline_kn: float, force_kn: float) -> float:
        span = baseline_kn - self._limits.critical_force_kn
        if span <= 0.0:
            # 基线本身已贴着临界力，按全退化处理，逼出换模而不是默默放行。
            return 100.0 if force_kn <= self._limits.critical_force_kn else 0.0
        return (baseline_kn - force_kn) / span * 100.0

    # ---- 换模重置 -------------------------------------------------------

    def reset(self, station: str = DEFAULT_STATION, reason: str = "换模") -> WearState:
        """换模后作废该工位的基线与退化结论，从跑合期重新开始。"""

        moment = self._clock.now()
        with self._lock:
            prior = self._stations.get(station)
            epoch = (prior.change_epoch + 1) if prior else 1
            self._stations[station] = _StationModel(
                self._limits, self._nominal_close_kn, epoch
            )
            self._changes += 1
            state = WearState(
                station=station,
                phase="warming",
                wear_pct=0.0,
                baseline_kn=0.0,
                baseline_ready=False,
                force_kn=0.0,
                setpoint_kn=self._nominal_close_kn,
                critical_kn=self._limits.critical_force_kn,
                remaining_pct=100.0,
                shots_since_change=0,
                baseline_shots=0,
                confirming=0,
                frozen=False,
                change_epoch=epoch,
                at=moment,
            )
            self._last[station] = state
        self._bus.publish(
            Event.of(
                "mold",
                "mold",
                "mold_changed",
                subject="clamp_wear",
                severity="notice",
                station=station,
                reason=reason,
                change_epoch=epoch,
                origin="clamp_wear",
            )
        )
        return state

    # ---- 读取 -----------------------------------------------------------

    def last_state(self, station: str = DEFAULT_STATION) -> WearState | None:
        """返回该工位最近一次评估（仅供展示与审计）。"""

        with self._lock:
            return self._last.get(station)

    def stations(self) -> list[str]:
        with self._lock:
            return sorted(self._last)

    def snapshot(self) -> dict:
        with self._lock:
            stations = {
                station: self._last[station].as_dict()
                for station in sorted(self._last)
            }
            changes = self._changes
        return {
            "nominal_close_kn": round(self._nominal_close_kn, 3),
            "critical_force_kn": self._limits.critical_force_kn,
            "warmup_shots": self._limits.warmup_shots,
            "baseline_shots": self._limits.baseline_shots,
            "confirm_shots": self._limits.confirm_shots,
            "setpoint_shift_pct": self._limits.setpoint_shift_pct,
            "window_wear_pct": self._limits.window_wear_pct,
            "changes": changes,
            "stations": stations,
        }

    # ---- 内部 -----------------------------------------------------------

    def _model(self, station: str) -> _StationModel:
        model = self._stations.get(station)
        if model is None:
            # 没有显式换模事件就来了合模样本：按第 1 代新模从跑合期起步，
            # 绝不拿历史缺失当零退化。
            model = _StationModel(self._limits, self._nominal_close_kn, change_epoch=1)
            self._stations[station] = model
        return model

    def _publish(self, state: WearState) -> None:
        severity = {
            "warming": "info",
            "learning": "info",
            "healthy": "notice",
            "watch": "notice",
            "window": "warning",
            "critical": "alarm",
        }[state.phase]
        payload = state.as_dict()
        payload.pop("at", None)
        self._bus.publish(
            Event.of(
                "mold",
                "mold",
                "clamp_wear_updated",
                subject="clamp_wear",
                severity=severity,
                at=state.at,
                **payload,
            )
        )
