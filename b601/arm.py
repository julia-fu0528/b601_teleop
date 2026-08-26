"""motorbridge wrapper for the B601-RS RobStride motors on CAN.

Positions are read through the mechPos parameter (0x7019): on RS firmware the cached
MotorState from the feedback stream is not reliable (Seeed calibration notes, 2026-07-17).
"""
from __future__ import annotations

import time

import numpy as np
from motorbridge import CallError, Controller, Mode

from .config import Config

MECH_POS = 0x7019
KP_MAX = 500.0   # RobStride MIT frame encoding range
KD_MAX = 5.0


class VelocityEstimator:
    """Finite-difference velocity with a first-order low-pass (mechVel 0x701A is not rad/s on RS firmware)."""

    def __init__(self, n: int, tau_s: float = 0.03) -> None:
        self._prev: np.ndarray | None = None
        self.v = np.zeros(n)
        self.tau = float(tau_s)

    def reset(self) -> None:
        self._prev = None
        self.v[:] = 0.0

    def update(self, q: np.ndarray, dt: float) -> np.ndarray:
        if self._prev is None:
            self._prev = q.copy()
            return self.v
        raw = (q - self._prev) / max(dt, 1e-4)
        self._prev = q.copy()
        a = dt / (self.tau + dt)
        self.v += a * (raw - self.v)
        return self.v


class PositionSource:
    """Live joint positions at loop rate.

    Enabled motors answer every MIT command with a feedback frame (pos/vel/torque/temps) that
    motorbridge caches -> free. Disabled motors only expose mechPos through a parameter read
    (~10 ms each), so those are refreshed one joint per cycle, round-robin (they do not move).
    Every `verify_every` cycles one feedback-fed joint is cross-checked against mechPos; after
    `max_mismatch` disagreements > `tol` that joint falls back to parameter reads.
    """

    def __init__(self, arm, active: np.ndarray, *, verify_every: int = 10, tol: float = 0.03, max_mismatch: int = 3) -> None:
        self.arm = arm
        self.n = len(active)
        self.active = np.asarray(active, dtype=bool)
        self.verify_every = int(verify_every)
        self.tol = float(tol)
        self.max_mismatch = int(max_mismatch)
        self.q = np.asarray(arm.read_q(), dtype=float).copy()
        self.vel_fb = np.full(self.n, np.nan)
        self.torq = np.full(self.n, np.nan)
        self.temp = np.full(self.n, np.nan)
        self.src = ["param"] * self.n
        self.fallback = np.zeros(self.n, dtype=bool)
        self.mismatch = np.zeros(self.n, dtype=int)
        self.warnings: list[str] = []
        self._cycle = 0
        self._rr_need = 0
        self._rr_verify = 0

    def update(self) -> np.ndarray:
        states = self.arm.poll_states()
        need = []
        for i in range(self.n):
            st = states[i]
            if self.active[i] and st is not None and not self.fallback[i]:
                self.q[i] = float(st.pos)
                self.vel_fb[i] = float(st.vel)
                self.torq[i] = float(st.torq)
                self.temp[i] = max(float(st.t_rotor), float(st.t_mos))
                self.src[i] = "fb"
            else:
                self.src[i] = "param"
                need.append(i)
        fb = [i for i in range(self.n) if self.src[i] == "fb"]
        if fb and self._cycle % self.verify_every == 0:
            j = fb[self._rr_verify % len(fb)]
            self._rr_verify += 1
            qp = float(self.arm.read_q_param([j])[0])
            if abs(qp - self.q[j]) > self.tol:
                self.mismatch[j] += 1
                msg = f"feedback pos {self.q[j]:+.4f} vs mechPos {qp:+.4f} on joint index {j} (mismatch {self.mismatch[j]}/{self.max_mismatch})"
                self.warnings.append(msg)
                if self.mismatch[j] >= self.max_mismatch:
                    self.fallback[j] = True
                    self.warnings.append(f"joint index {j}: feedback position unreliable -> falling back to mechPos reads")
                self.q[j] = qp
            else:
                self.mismatch[j] = 0
        elif need:
            j = need[self._rr_need % len(need)]
            self._rr_need += 1
            self.q[j] = float(self.arm.read_q_param([j])[0])
        self._cycle += 1
        return self.q.copy()

    def summary(self) -> str:
        nfb = sum(1 for s in self.src if s == "fb")
        return f"fb:{nfb} param:{self.n - nfb}" + (" FALLBACK" if self.fallback.any() else "")


class RobstrideArm:
    """Arm + optional gripper. Nothing is energized until enable() is called."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.n = len(cfg.joints)
        self.ctrl = Controller(cfg.channel)
        self.motors = [self.ctrl.add_robstride_motor(j.id, cfg.host_id, j.model) for j in cfg.joints]
        self.gripper = None
        if cfg.gripper is not None:
            self.gripper = self.ctrl.add_robstride_motor(cfg.gripper.id, cfg.host_id, cfg.gripper.model)
        self._timeout = int(cfg.loop.read_timeout_ms)
        self.enabled = np.zeros(self.n, dtype=bool)
        self.gripper_enabled = False

    # ---- reading -------------------------------------------------------------------------
    def read_q(self) -> np.ndarray:
        """Joint positions (rad) via mechPos. Raises CallError on a bus timeout."""
        return np.array([m.robstride_get_param_f32(MECH_POS, self._timeout) for m in self.motors], dtype=float)

    def read_q_param(self, idx) -> np.ndarray:
        """mechPos for a subset of joints (each read costs ~10 ms on the motorbridge ABI)."""
        return np.array([self.motors[i].robstride_get_param_f32(MECH_POS, self._timeout) for i in idx], dtype=float)

    def read_gripper_q(self) -> float:
        if self.gripper is None:
            return 0.0
        return float(self.gripper.robstride_get_param_f32(MECH_POS, self._timeout))

    def gripper_state(self) -> tuple[float, float] | None:
        """(pos, vel) of the gripper from its MIT feedback; None if no feedback yet."""
        if self.gripper is None:
            return None
        st = self.gripper.get_state()
        return None if st is None else (float(st.pos), float(st.vel))

    def poll_states(self) -> list:
        """Cached MotorState per joint (may be None) - used for torque/temperature telemetry only."""
        try:
            self.ctrl.poll_feedback_once()
        except CallError:
            pass
        return [m.get_state() for m in self.motors]

    # ---- mode / power --------------------------------------------------------------------
    def prepare_mit(self, active: np.ndarray, gripper: bool) -> None:
        for i, m in enumerate(self.motors):
            if active[i]:
                m.ensure_mode(Mode.MIT, 1000)
                time.sleep(0.05)
        if gripper and self.gripper is not None:
            self.gripper.ensure_mode(Mode.MIT, 1000)
            time.sleep(0.05)
        time.sleep(0.2)

    def enable(self, active: np.ndarray, gripper: bool) -> None:
        for i, m in enumerate(self.motors):
            if active[i]:
                m.enable()
                self.enabled[i] = True
                time.sleep(0.02)
        if gripper and self.gripper is not None:
            self.gripper.enable()
            self.gripper_enabled = True
        time.sleep(0.1)

    def disable_all(self) -> None:
        for m in self.motors + ([self.gripper] if self.gripper is not None else []):
            try:
                m.disable()
            except CallError:
                pass
        self.enabled[:] = False
        self.gripper_enabled = False

    # ---- commands ------------------------------------------------------------------------
    def send_mit(self, pos, vel, kp, kd, tau, active: np.ndarray) -> None:
        for i, m in enumerate(self.motors):
            if not active[i]:
                continue
            m.send_mit(
                float(pos[i]),
                float(vel[i]),
                float(np.clip(kp[i], 0.0, KP_MAX)),
                float(np.clip(kd[i], 0.0, KD_MAX)),
                float(tau[i]),
            )

    def send_gripper_mit(self, pos: float, kp: float, kd: float, tau: float) -> None:
        if self.gripper is None or not self.gripper_enabled:
            return
        self.gripper.send_mit(float(pos), 0.0, float(np.clip(kp, 0.0, KP_MAX)), float(np.clip(kd, 0.0, KD_MAX)), float(tau))

    def close(self) -> None:
        try:
            self.disable_all()
        finally:
            try:
                self.ctrl.shutdown()
            except CallError:
                pass
            self.ctrl.close()
