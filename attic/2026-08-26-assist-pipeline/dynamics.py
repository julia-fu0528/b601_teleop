"""Pinocchio gravity model for the arm (gripper finger joints locked)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin


class ArmDynamics:
    def __init__(
        self,
        urdf: str | Path,
        joint_names: list[str],
        lock_joints: tuple[str, ...] = (),
        g_scale: np.ndarray | list[float] | None = None,
        g_bias: np.ndarray | list[float] | None = None,
    ) -> None:
        full = pin.buildModelFromUrdf(str(urdf))
        lock_ids = [full.getJointId(n) for n in lock_joints]
        for n, jid in zip(lock_joints, lock_ids):
            if jid == 0:
                raise ValueError(f"lock joint {n!r} not in URDF")
        self.model = pin.buildReducedModel(full, lock_ids, pin.neutral(full)) if lock_ids else full
        self.data = self.model.createData()

        names = [self.model.names[i] for i in range(1, self.model.njoints)]
        if list(joint_names) != names:
            raise ValueError(f"config joints {list(joint_names)} != URDF joints {names}")
        self.nq = int(self.model.nq)
        if self.nq != len(names):
            raise ValueError("expected 1-DoF revolute joints only")

        self.g_scale = np.ones(self.nq) if g_scale is None else np.asarray(g_scale, dtype=float)
        if self.g_scale.shape != (self.nq,):
            raise ValueError("g_scale length mismatch")
        self.g_bias = np.zeros(self.nq) if g_bias is None else np.asarray(g_bias, dtype=float)
        if self.g_bias.shape != (self.nq,):
            raise ValueError("g_bias length mismatch")
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=float)
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=float)
        self.effort = np.asarray(self.model.effortLimit, dtype=float)

        ee = "gripper_end" if self.model.existFrame("gripper_end") else "link6"
        self._ee_fid = self.model.getFrameId(ee)

    def gravity_raw(self, q: np.ndarray) -> np.ndarray:
        """Generalized gravity g(q) of the URDF (N.m); tau = g(q) holds the arm still."""
        return np.asarray(pin.computeGeneralizedGravity(self.model, self.data, np.asarray(q, dtype=float)))

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Feed-forward: g(q) * g_scale + g_bias (per-joint calibration)."""
        return self.gravity_raw(q) * self.g_scale + self.g_bias

    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """M(q)a + C(q,v)v + g_calibrated(q) via RNEA (N.m): the torque the rigid-body
        model needs for this motion, with the fitted per-joint gravity calibration."""
        tau = np.asarray(pin.rnea(self.model, self.data, np.asarray(q, dtype=float),
                                  np.asarray(v, dtype=float), np.asarray(a, dtype=float)))
        return tau - self.gravity_raw(q) + self.gravity(q)

    def ee_jacobian(self, q: np.ndarray) -> np.ndarray:
        """6 x nq Jacobian of the gripper_end frame, world-aligned axes (rows: linear xyz, angular xyz)."""
        return np.array(pin.computeFrameJacobian(
            self.model, self.data, np.asarray(q, dtype=float), self._ee_fid, pin.LOCAL_WORLD_ALIGNED))

    def limit_violation(self, q: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Per-joint distance outside [lower+margin, upper-margin] (0 when inside)."""
        q = np.asarray(q, dtype=float)
        below = np.maximum(self.lower + margin - q, 0.0)
        above = np.maximum(q - (self.upper - margin), 0.0)
        return below + above
