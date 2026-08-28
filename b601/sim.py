"""Hardware-free stand-in for RobstrideArm: rigid-body joint simulation with friction.

Used to exercise the controller's phase machine and safety logic without a robot.
Equation of motion per joint: I * qdd = tau_cmd - g_true(q) + tau_ext - friction(v).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dynamics import ArmDynamics


@dataclass(frozen=True)
class SimState:
    pos: float
    vel: float
    torq: float
    t_mos: float = 35.0
    t_rotor: float = 35.0


class SimArm:
    def __init__(
        self,
        dyn: ArmDynamics,
        q0: np.ndarray,
        dt: float,
        *,
        inertia: np.ndarray | None = None,
        coulomb: float | np.ndarray = 0.3,
        static: float | np.ndarray | None = None,
        viscous: float = 0.05,
        truth_scale: float = 1.0,
        external: Callable[[float, np.ndarray], np.ndarray] | None = None,
        substeps: int = 10,
    ) -> None:
        self.dyn = dyn
        self.n = dyn.nq
        self.q = np.asarray(q0, dtype=float).copy()
        self.v = np.zeros(self.n)
        self.dt = float(dt)
        self.I = np.array([0.05, 0.25, 0.12, 0.02, 0.01, 0.005]) if inertia is None else np.asarray(inertia, float)
        self.coulomb = np.asarray(coulomb, dtype=float)   # kinetic (sliding), scalar or per-joint
        self.static = self.coulomb if static is None else np.asarray(static, dtype=float)  # breakaway level
        self.viscous = float(viscous)
        self.truth_scale = float(truth_scale)
        self.external = external
        self.substeps = int(substeps)
        self.t = 0.0
        self.enabled = np.zeros(self.n, dtype=bool)
        self.gripper = None
        self.gripper_enabled = False
        self._cmd = None
        self._tau_applied = np.zeros(self.n)
        self.max_abs_v = 0.0

    # ---- same interface as RobstrideArm --------------------------------------------------
    # poll_states() is called once per control cycle and advances the physics by dt;
    # read_q()/read_q_param() just report the current state.
    def read_q(self) -> np.ndarray:
        return self.q.copy()

    def read_q_param(self, idx) -> np.ndarray:
        return np.array([self.q[i] for i in idx], dtype=float)

    def read_gripper_q(self) -> float:
        return 0.0

    def poll_states(self) -> list:
        self._step()
        return [SimState(float(self.q[i]), float(self.v[i]), float(self._tau_applied[i])) if self.enabled[i] else None
                for i in range(self.n)]

    def prepare_mit(self, active, gripper) -> None:
        pass

    def enable(self, active, gripper) -> None:
        self.enabled = np.asarray(active, dtype=bool).copy()

    def disable_all(self) -> None:
        self.enabled[:] = False
        self._cmd = None

    def send_mit(self, pos, vel, kp, kd, tau, active) -> None:
        self._cmd = (np.array(pos, float), np.array(vel, float), np.array(kp, float),
                     np.array(kd, float), np.array(tau, float), np.asarray(active, bool))

    def send_gripper_mit(self, pos, kp, kd, tau) -> None:
        pass

    def close(self) -> None:
        self.disable_all()

    # ---- physics -------------------------------------------------------------------------
    def _step(self) -> None:
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            tau_cmd = np.zeros(self.n)
            if self._cmd is not None:
                pos, vel, kp, kd, tau, active = self._cmd
                on = active & self.enabled
                tau_cmd[on] = (kp * (pos - self.q) + kd * (vel - self.v) + tau)[on]
            self._tau_applied = tau_cmd
            g_true = self.dyn.gravity_raw(self.q) * self.truth_scale
            ext = self.external(self.t, self.q) if self.external is not None else 0.0
            net = tau_cmd - g_true + ext
            # Coulomb + viscous friction with simple stiction
            fric = self.coulomb * np.tanh(self.v / 0.02) + self.viscous * self.v
            stuck = (np.abs(self.v) < 1e-3) & (np.abs(net) < self.static)
            acc = np.where(stuck, 0.0, (net - fric) / self.I)
            self.v = np.where(stuck, 0.0, self.v + acc * h)
            self.q = self.q + self.v * h
            # hard stops at the URDF limits
            lo_hit = self.q < self.dyn.lower
            hi_hit = self.q > self.dyn.upper
            self.q = np.clip(self.q, self.dyn.lower, self.dyn.upper)
            self.v = np.where(lo_hit | hi_hit, 0.0, self.v)
            self.t += h
        self.max_abs_v = max(self.max_abs_v, float(np.abs(self.v).max()))
