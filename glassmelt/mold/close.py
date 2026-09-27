"""合模。

合模到位是吹气的硬前置条件。模具重开之后，上一拍的"合模到位"结论必须
作废：合模力、代次都要按当下重新判。吹气侧拿到的是哪一拍的合模结论，
要跟着代次一起落到审计里，事后才能对上"这一炮吹气时模具到底是合着的
还是开着的"。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import MoldLimits
from ..errors import InterlockError
from ..eventbus import Event, EventBus


@dataclass(frozen=True)
class Closure:
    """一次合模状态判定。"""

    closed: bool
    force_kn: float
    epoch: int
    at: float

    def as_dict(self) -> dict:
        return {
            "closed": self.closed,
            "force_kn": round(self.force_kn, 3),
            "epoch": self.epoch,
            "at": round(self.at, 3),
        }


class MoldCloser:
    """合模机构。"""

    def __init__(self, clock, bus: EventBus, limits: MoldLimits, station: str = "section-1") -> None:
        self._clock = clock
        self._bus = bus
        self._limits = limits
        self._station = station
        self._closed = False
        self._force_kn = 0.0
        self._setpoint_kn = limits.close_min_force_kn * 1.12
        self._epoch = 0
        self._last_closure: Closure | None = None
        self._closes = 0
        self._opens = 0
        self._setpoint_changes = 0
        self._lock = threading.RLock()

    @property
    def setpoint_kn(self) -> float:
        """当前合模设定档（指令峰值力）；退化跟踪靠它区分调机与磨损。"""

        with self._lock:
            return self._setpoint_kn

    def adjust_setpoint(self, setpoint_kn: float, reason: str = "调整合模设定") -> float:
        """调整合模设定档。设定一档变，峰值力整档平移是工况变化不是磨损。"""

        setpoint = float(setpoint_kn)
        if setpoint < self._limits.close_min_force_kn:
            raise InterlockError(
                f"合模设定 {setpoint:.3f} kN 低于到位下限 "
                f"{self._limits.close_min_force_kn:.3f} kN，按此设定无法稳定到位",
                component="mold",
            )
        with self._lock:
            changed = abs(setpoint - self._setpoint_kn) > 1e-9
            self._setpoint_kn = setpoint
            if changed:
                self._setpoint_changes += 1
            current = self._setpoint_kn
        if changed:
            self._bus.publish(
                Event.of(
                    "mold",
                    "mold",
                    "mold_setpoint_adjusted",
                    subject="mold_closer",
                    severity="notice",
                    station=self._station,
                    setpoint_kn=round(current, 3),
                    reason=reason,
                )
            )
        return current

    def close(self, force_kn: float | None = None) -> Closure:
        """合模，返回现场判定的合模状态。

        不传实测力时按当前设定档取峰值力（仿真／手动触发用）；实测算例直接
        给本模实测峰值。设定档随事件一起带出，退化监视器据此区分调机与磨损。
        """

        with self._lock:
            setpoint = self._setpoint_kn
        force = setpoint if force_kn is None else float(force_kn)
        with self._lock:
            self._force_kn = force
            self._closed = self._limits.close_ok(force)
            self._epoch += 1
            self._closes += 1
            closure = self.closure()
        self._bus.publish(
            Event.of(
                "mold",
                "mold",
                "mold_closed" if closure.closed else "mold_close_short",
                subject="mold_closer",
                severity="notice" if closure.closed else "warning",
                station=self._station,
                force_kn=round(force, 3),
                setpoint_kn=round(setpoint, 3),
                epoch=closure.epoch,
            )
        )
        return closure

    def open_mold(self, reason: str) -> Closure:
        """脱模或重试时重开模具，返回重开后的合模状态。"""

        with self._lock:
            self._closed = False
            self._force_kn = 0.0
            self._epoch += 1
            self._opens += 1
            closure = self.closure()
        self._bus.publish(
            Event.of(
                "mold",
                "mold",
                "mold_opened",
                subject="mold_closer",
                severity="notice",
                reason=reason,
                epoch=closure.epoch,
            )
        )
        return closure

    def closure(self) -> Closure:
        """现场评估合模状态：始终按当前合模力与代次重新判定。"""

        with self._lock:
            closure = Closure(
                closed=bool(self._closed and self._limits.close_ok(self._force_kn)),
                force_kn=self._force_kn,
                epoch=self._epoch,
                at=self._clock.now(),
            )
            self._last_closure = closure
        return closure

    def last_closure(self) -> Closure | None:
        """返回最近一次合模评估（仅供展示与审计）。"""

        with self._lock:
            return self._last_closure

    def is_closed(self) -> bool:
        """现场判定模具当前是否已合拢到位。"""

        return self.closure().closed

    def counts(self) -> dict:
        with self._lock:
            return {
                "closes": self._closes,
                "opens": self._opens,
                "epoch": self._epoch,
                "setpoint_changes": self._setpoint_changes,
            }

    @property
    def station(self) -> str:
        """该合模机构所属工位名，退化监视器按它分台建模。"""

        return self._station

    def snapshot(self) -> dict:
        closure = self.last_closure()
        counts = self.counts()
        with self._lock:
            return {
                "closed": bool(self._closed),
                "force_kn": round(self._force_kn, 3),
                "setpoint_kn": round(self._setpoint_kn, 3),
                "min_force_kn": self._limits.close_min_force_kn,
                "epoch": counts["epoch"],
                "closes": counts["closes"],
                "opens": counts["opens"],
                "setpoint_changes": counts["setpoint_changes"],
                "last_closure": closure.as_dict() if closure else None,
            }
