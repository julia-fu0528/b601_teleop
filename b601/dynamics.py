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
        lock_q: np.ndarray | None = None,
    ) -> None:
        full = pin.buildModelFromUrdf(str(urdf))
        # getJointId returns njoints (not 0) for unknown names, and buildReducedModel then
        # silently locks nothing - check existence explicitly
        for n in lock_joints:
            if not full.existJointName(n):
                raise ValueError(f"lock joint {n!r} not in URDF")
        lock_ids = [full.getJointId(n) for n in lock_joints]
        q_lock = pin.neutral(full) if lock_q is None else np.asarray(lock_q, dtype=float)
        self.model = pin.buildReducedModel(full, lock_ids, q_lock) if lock_ids else full
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
        if not self.model.existFrame(ee):
            raise ValueError("URDF has neither 'gripper_end' nor 'link6' frame for the EE "
                             "(getFrameId would silently return an invalid id)")
        self._ee_fid = self.model.getFrameId(ee)

    def gravity_raw(self, q: np.ndarray) -> np.ndarray:
        """Generalized gravity g(q) of the URDF (N.m); tau = g(q) holds the arm still."""
        return np.asarray(pin.computeGeneralizedGravity(self.model, self.data, np.asarray(q, dtype=float)))

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Feed-forward: g(q) * g_scale + g_bias (per-joint calibration)."""
        return self.gravity_raw(q) * self.g_scale + self.g_bias

    def ee_jacobian(self, q: np.ndarray, frame: str = "world") -> np.ndarray:
        """6 x nq Jacobian of the gripper_end frame (rows: linear xyz, angular xyz).
        frame="world": world-aligned axes; frame="local": tool axes (x = roll/j6 axis)."""
        ref = pin.LOCAL_WORLD_ALIGNED if frame == "world" else pin.LOCAL
        return np.array(pin.computeFrameJacobian(
            self.model, self.data, np.asarray(q, dtype=float), self._ee_fid, ref))

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Joint-space inertia M(q) of the URDF links (no reflected rotor inertia), symmetrized."""
        M = pin.crba(self.model, self.data, np.asarray(q, dtype=float))
        return np.triu(M) + np.triu(M, 1).T

    def coriolis(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Coriolis matrix C(q, qd) with the Christoffel property Mdot = C + C^T."""
        return np.array(pin.computeCoriolisMatrix(
            self.model, self.data, np.asarray(q, dtype=float), np.asarray(v, dtype=float)))

    def limit_violation(self, q: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Per-joint distance outside [lower+margin, upper-margin] (0 when inside)."""
        q = np.asarray(q, dtype=float)
        below = np.maximum(self.lower + margin - q, 0.0)
        above = np.maximum(q - (self.upper - margin), 0.0)
        return below + above
