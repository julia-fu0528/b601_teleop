"""Seeed reBot Arm 102 leader arm (7 Fashion Star UART bus servos) and leader stand-ins."""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from .config import LeaderCfg


def unwrap(value: float, lo: float, hi: float) -> float:
    """Bring a multi-turn servo angle into the +/-180 deg window centred on the joint range (Seeed's method)."""
    center = 0.5 * (lo + hi)
    low, high = center - 180.0, center + 180.0
    for k in range(4096):
        for cand in (value + k * 360.0, value - k * 360.0):
            if low <= cand <= high:
                return cand
    return value - round((value - center) / 360.0) * 360.0


def map_to_follower(leader_deg: np.ndarray, cfg: LeaderCfg) -> np.ndarray:
    """Leader servo angles (deg, 7) -> follower joint targets (deg, 7): direction * (leader - offset)."""
    return np.asarray(cfg.directions, float) * (np.asarray(leader_deg, float) - np.asarray(cfg.offsets, float))


class RebotLeader:
    """Direct serial access. read_deg() takes ~5 ms for all 7 servos (one sync-monitor command)."""

    def __init__(self, cfg: LeaderCfg) -> None:
        from motorbridge_smart_servo import FashionStarServo
        self.cfg = cfg
        self.bus = FashionStarServo(cfg.port, baudrate=int(cfg.baudrate))
        missing = [i for i in cfg.ids if not self.bus.ping(i)]
        if missing:
            self.bus.close()
            raise RuntimeError(f"leader servos not responding: ids {missing} on {cfg.port}")

    def read_deg(self) -> np.ndarray:
        res = self.bus.sync_monitor(list(self.cfg.ids))
        out = np.zeros(len(self.cfg.ids))
        for k, sid in enumerate(self.cfg.ids):
            m = res.get(sid)
            if m is None:
                raise RuntimeError(f"leader servo id {sid} did not answer")
            lo, hi = self.cfg.ranges[k]
            out[k] = unwrap(float(m.angle_deg), lo, hi)
        return out

    def close(self) -> None:
        try:
            self.bus.close()
        except Exception:
            pass


class LeaderReader:
    """Background thread: keeps the freshest leader sample so the CAN loop never waits on the serial port."""

    def __init__(self, leader: RebotLeader) -> None:
        self.leader = leader
        self._lock = threading.Lock()
        self._t: float | None = None
        self._deg: np.ndarray | None = None
        self.errors = 0
        self.reads = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="leader-reader", daemon=True)

    def start(self) -> "LeaderReader":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                deg = self.leader.read_deg()
                with self._lock:
                    self._t = time.perf_counter()
                    self._deg = deg
                    self.reads += 1
            except Exception:
                self.errors += 1
                time.sleep(0.005)

    def sample(self, t_now: float | None = None) -> tuple[float, np.ndarray] | None:
        with self._lock:
            if self._deg is None:
                return None
            return self._t, self._deg.copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.leader.close()


class ScriptedLeader:
    """Leader stand-in for simulation/tests: deg7 = fn(t)."""

    def __init__(self, fn: Callable[[float], np.ndarray]) -> None:
        self.fn = fn

    def sample(self, t_now: float) -> tuple[float, np.ndarray] | None:
        return t_now, np.asarray(self.fn(t_now), float)

    def close(self) -> None:
        pass
