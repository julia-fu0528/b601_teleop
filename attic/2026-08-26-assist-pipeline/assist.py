"""Drag-teach torque rebalancing (operator's pipeline, 2026-08-26), built step by step.

The hand force F loads every joint through its Jacobian column: tau_i = J_i . F.
Whichever joints' torque crosses their OWN breakaway first yield and absorb the motion
as EE rotation - often the wrist: for in/out pushes j4's lever is ~2x j2/j3's, and for
lateral pushes j5's threshold is crossed at ~half of joint1's push force.

Pipeline (complete):
  1. measure the external torque on the downstream joints (4-6)
  2. infer the external torque on the upstream joints (1-3) via the Jacobian
  3. compare the two halves: downstream torque-loading vs upstream motion-carriage
  4. use the Jacobian to move torque from the yielding half to the stuck half

Step 1 - measurement. External torque is observable only through BACKDRIVE: a stuck
joint hides it inside its static friction band, and in free-float drag (kp = 0) the
motor does not resist a push, so torq_fb cannot see it either. While a joint moves,
the hand torque on it is exactly what sustains the motion:

    tau_down_j = fric_j * tanh(qd_j / v0) + kd_j * qd_j          (zero at rest)

Step 2 - inference. The upstream joints are the stuck ones, so their external torque
cannot be measured the same way; it is reconstructed through the Jacobian. The hand
force that explains the downstream torques is the damped least-squares solution of
A F = tau_down with A = J_down^T (rows = downstream joints), and the torque the
upstream joints are receiving is its transpose map:

    F      = (A^T A + lambda^2 I)^-1 A^T tau_down
    tau_up = J_up^T F

Observability caveat: only the force component in the span of the BACKDRIVING joints'
lever rows is recoverable (min-norm elsewhere). With one wrist joint yielding the
direction is 1-D, so tau_up is exact along that axis and blind across it; more wrist
joints yielding = better conditioning.

Step 3 - comparison. Raw torque against raw torque is the wrong comparison (a joint
yields when its torque crosses its OWN breakaway; for lateral pushes joint1 receives
MORE raw torque than joint5 and still loses). Comparing threshold-normalized loadings
on both sides also fails, for a subtler reason: the step-2 inference inflates the
upstream loading when the force direction is ambiguous, and a stuck-but-loaded joint
is not helping anyway - loading is not carriage. So the halves are compared on what
each is actually DOING: is the downstream half torque-loaded, and is the upstream
half producing any of the EE motion yet?

    r_down   = max |tau_down| / brk_down          ~1 while some wrist joint backdrives
    carry_up = |J_up qd_up| / (|J_up qd_up| + |J_down qd_down|)   upstream share of EE speed
    weight   = clip(r_down, 0, 1) * (1 - carry_up)

weight is the transfer gate for step 4: ~1 when the wrist carries the push alone,
fading toward 0 as the upstream half takes over the motion (no double-driving a half
that is already carrying), and 0 with no hand at all.

Step 4 - transfer (operator's choice: J^T for SIZE, J+ for DIRECTION). The two
Jacobian maps disagree in skewed geometry: the torque a force EXERTS on a joint
(J^T F) can point against the rotation the joint needs to CARRY the motion (J+ u) -
at the folded-wrist pose a radial push loads joint2 negative while joint2 must turn
positive to help. So the transferred torque takes its per-joint pattern from the
motion-carry map and its magnitude from the sensed force:

    u_m  = J_dn qd_dn / |J_dn qd_dn|            direction the wrist is carrying the EE
    p    = J_up^T (J_up J_up^T + lambda^2)^-1 u_m     joint pattern that carries u_m
    s    = gain * |F| / |F_equiv(p)|            so the pattern's EE force = gain x sensed
    tau  = clip(lowpass(weight * s * p), +-tau_max)   per upstream joint

Zero with no hand (gate shut, u_m undefined), bounded by the sensed hand force x gain
and by tau_max, and self-fading as the upstream half takes the motion over (gate) or
the wrist stops slipping (u_m source). No rate targets, no integrator, no damping.
vel_abort and the controller clamp remain as backstops.
"""
from __future__ import annotations

import numpy as np

# downstream Coulomb friction used to infer the hand torque from backdrive velocity:
# Seeed sweep 2026-07-17 (j4, j5, j6); re-measure with the 'f' key if it reads off
FRIC_DOWN = np.array([0.30, 0.21, 0.21])
# upstream breakaway used by the step-3 comparison (j1 assumed like its neighbours)
BRK_UP = np.array([0.53, 0.53, 0.49])
# full-arm Coulomb friction for the DynamicAssist observer (Seeed sweep 2026-07-17)
FRIC_ALL = np.array([0.53, 0.53, 0.49, 0.30, 0.21, 0.21])


class TorqueRebalance:
    def __init__(self, dyn, *, tau_max: float = 1.2, gain: float = 2.0,
                 fric_down: np.ndarray | None = None, kd_down: np.ndarray | None = None,
                 brk_up: np.ndarray | None = None,
                 v0: float = 0.08, tau_f: float = 0.1, dls_lambda: float = 0.05,
                 nup: int = 3) -> None:
        self.dyn = dyn
        self.nup = int(nup)             # joints 1..nup are "upstream"; the rest "downstream"
        self.tau_max = float(tau_max)   # N.m cap per upstream joint (used from step 4 on)
        self.gain = float(gain)         # transfer gain (used from step 4 on)
        self.fric_down = FRIC_DOWN.copy() if fric_down is None else np.asarray(fric_down, float)
        self.kd_down = np.full(3, 0.05) if kd_down is None else np.asarray(kd_down, float)
        self.v0 = float(v0)             # rad/s tanh knee of the friction term (above FD velocity noise)
        self.tau_f = float(tau_f)       # s low-pass on the measurement
        self.lam2 = float(dls_lambda) ** 2  # m^2 damping of the force fit (finite at singularities)
        self.brk_up = BRK_UP.copy() if brk_up is None else np.asarray(brk_up, float)
        self.tau_down = np.zeros(3)     # step 1: filtered external torque on joints 4-6 (N.m)
        self.F = np.zeros(3)            # step 2: inferred hand force (world xyz, N)
        self.tau_up = np.zeros(self.nup)  # step 2: inferred external torque on joints 1-3 (N.m)
        self.r_down = 0.0               # step 3: downstream loading relative to its breakaway
        self.carry_up = 0.0             # step 3: upstream share of the EE speed, in [0, 1]
        self.weight = 0.0               # step 3: transfer gate for step 4, in [0, 1]
        self.tau_out = np.zeros(self.nup)  # step 4: filtered transferred torque (N.m)

    def measure_downstream(self, v: np.ndarray) -> np.ndarray:
        """Step 1: external torque on the downstream joints, from their backdrive."""
        qd = v[self.nup:]
        return self.fric_down * np.tanh(qd / self.v0) + self.kd_down * qd

    def infer_upstream(self, Jv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Step 2: hand force explaining tau_down (DLS), mapped to the upstream joints."""
        A = Jv[:, self.nup:].T                         # rows: downstream joints, cols: xyz force
        F = np.linalg.solve(A.T @ A + self.lam2 * np.eye(3), A.T @ self.tau_down)
        return F, Jv[:, :self.nup].T @ F

    def compare(self, Jv: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
        """Step 3: downstream torque-loading vs upstream motion-carriage, and the gate."""
        r_down = float((np.abs(self.tau_down) / self.fric_down).max())
        v_up = float(np.linalg.norm(Jv[:, :self.nup] @ v[:self.nup]))
        v_down = float(np.linalg.norm(Jv[:, self.nup:] @ v[self.nup:]))
        carry_up = v_up / (v_up + v_down + 1e-6)
        weight = float(np.clip(r_down, 0.0, 1.0) * (1.0 - carry_up))
        return r_down, carry_up, weight

    def transfer(self, Jv: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Step 4: torque for the upstream joints - J^T size, J+ direction (unfiltered)."""
        B = Jv[:, :self.nup]
        v_dn = Jv[:, self.nup:] @ v[self.nup:]
        speed = float(np.linalg.norm(v_dn))
        if speed < 1e-4:
            return np.zeros(self.nup)
        u_m = v_dn / speed
        p = B.T @ np.linalg.solve(B @ B.T + self.lam2 * np.eye(3), u_m)
        if float(np.linalg.norm(p)) < 1e-6:            # motion the upstream half cannot make
            return np.zeros(self.nup)
        A = B.T                                        # statics dual: A F_equiv = p
        F_equiv = np.linalg.solve(A.T @ A + self.lam2 * np.eye(3), A.T @ p)
        fmag = float(np.linalg.norm(F_equiv))
        if fmag < 1e-6:
            return np.zeros(self.nup)
        s = self.gain * float(np.linalg.norm(self.F)) / fmag
        return s * p

    def update(self, q: np.ndarray, v: np.ndarray, dt: float = 0.01, **_) -> np.ndarray:
        Jv = self.dyn.ee_jacobian(q)[:3]
        raw = self.measure_downstream(v)
        self.tau_down += (dt / (self.tau_f + dt)) * (raw - self.tau_down)
        self.F, self.tau_up = self.infer_upstream(Jv)
        self.r_down, self.carry_up, self.weight = self.compare(Jv, v)
        raw_t = self.weight * self.transfer(Jv, v)
        self.tau_out += (dt / (self.tau_f + dt)) * (raw_t - self.tau_out)
        self.tau_out = np.clip(self.tau_out, -self.tau_max, self.tau_max)
        tau = np.zeros(len(v))
        tau[:self.nup] = self.tau_out
        return tau

    def summary(self) -> str:
        """Live view for the controller status line."""
        return ("tau_dn " + " ".join(f"{x:+.2f}" for x in self.tau_down)
                + " -> tau_up " + " ".join(f"{x:+.2f}" for x in self.tau_up)
                + f" w {self.weight:.2f} tx " + " ".join(f"{x:+.2f}" for x in self.tau_out))


class DynamicAssist:
    """Operator's inverse-dynamics formulation (flag --dyn-assist), separate from
    TorqueRebalance. Joint-space rigid-body equation:

        M(q) qdd + C(q, qd) qd + g(q) + fric(qd) = tau_motor + J^T F_ext

    Everything on the left is modeled or measured (qdd by differentiating the measured
    velocities, low-passed), and tau_motor is what was ACTUALLY sent last cycle
    (feed-forward - kd*qd, INCLUDING our own previous assist - without that term the
    assist would attribute its own effect to the hand and feed itself). The residual is
    the external torque; the hand force is its damped least-squares fit with rows
    weighted by |qd| (a stuck joint hides force inside its stiction band - it must not
    vote "zero force"). The motors then carry gain x the felt force on ALL joints:

        tau = clip(gain * drive * J^T F_ext, +-tau_max)

    dividing the driven joints' apparent friction by ~(1 + gain). Two structural
    safeguards, both learned from sim runaways: (a) SENSE/DRIVE SPLIT - a joint that
    is driven must not also sense (weight (1 - drive)); attributing a driven joint's
    own sustained motion to the hand is positive feedback through the accel-filter
    lag, which ran joint4 into vel_abort and joint3 to +1.9 rad in sim. Default
    drive = [1,1,1, wrist_gain..] with wrist_gain 0: the wrist senses, the base is
    driven. (b) DO-NO-HARM GATE - with one yielding joint the force direction is
    ambiguous (min-norm), and J^T of a skewed estimate can press a joint against the
    motion it should carry; any torque component whose sign opposes the J+ carry map
    of the observed EE motion is zeroed. Zero at rest (all sensor weights vanish),
    bounded by the sensed force x gain, tau_max, and the controller's clamps;
    vel_abort remains the backstop.
    """

    def __init__(self, dyn, *, tau_max: float = 1.2, gain: float = 2.0, wrist_gain: float = 0.0,
                 fric: np.ndarray | None = None, v0: float = 0.08,
                 acc_f: float = 0.08, tau_f: float = 0.1, dls_lambda: float = 0.05) -> None:
        self.dyn = dyn
        self.tau_max = float(tau_max)   # N.m cap per joint
        self.gain = float(gain)         # force multiplication at the EE
        self.drive = np.array([1.0, 1.0, 1.0, wrist_gain, wrist_gain, wrist_gain])
        self.fric = FRIC_ALL.copy() if fric is None else np.asarray(fric, float)
        self.v0 = float(v0)             # rad/s tanh knee: friction model + sensor weighting
        self.acc_f = float(acc_f)       # s low-pass on the acceleration estimate
        self.tau_f = float(tau_f)       # s low-pass on the force estimate
        self.lam2 = float(dls_lambda) ** 2
        self.v_prev: np.ndarray | None = None
        self.qdd = None                 # filtered joint accelerations
        self.F = np.zeros(3)            # filtered external force estimate (world xyz, N)
        self.tau_out = None             # last output, for the status line

    def update(self, q: np.ndarray, v: np.ndarray, dt: float = 0.01,
               tau_sent: np.ndarray | None = None, kd_sent: np.ndarray | None = None) -> np.ndarray:
        n = len(v)
        if self.v_prev is None:
            self.v_prev = v.copy()
            self.qdd = np.zeros(n)
            self.tau_out = np.zeros(n)
        qdd_raw = (v - self.v_prev) / max(dt, 1e-4)
        self.v_prev = v.copy()
        self.qdd += (dt / (self.acc_f + dt)) * (qdd_raw - self.qdd)

        tau_motor = (np.zeros(n) if tau_sent is None else np.asarray(tau_sent, float)) \
            - (np.zeros(n) if kd_sent is None else np.asarray(kd_sent, float)) * v
        lhs = self.dyn.inverse_dynamics(q, v, self.qdd) + self.fric * np.tanh(v / self.v0)
        tau_ext = lhs - tau_motor

        Jv = self.dyn.ee_jacobian(q)[:3]
        # moving joints are the credible sensors - but a joint we DRIVE must not also
        # sense: its own sustained motion would be attributed to the hand and feed
        # itself through the accel-filter lag (j3 ran away exactly this way in sim)
        w = np.tanh(np.abs(v) / self.v0) * np.clip(1.0 - self.drive, 0.0, 1.0)
        A = Jv.T * w[:, None]
        F = np.linalg.solve(A.T @ A + self.lam2 * np.eye(3), A.T @ (tau_ext * w))
        self.F += (dt / (self.tau_f + dt)) * (F - self.F)
        tau_raw = self.gain * self.drive * (Jv.T @ self.F)
        # do-no-harm gate: with one yielding joint the force direction is ambiguous
        # (min-norm pick), and J^T of a skewed estimate can press a joint AGAINST the
        # motion it should carry - zero any component that opposes the J+ carry map
        v_ee = Jv @ v
        speed = float(np.linalg.norm(v_ee))
        if speed > 1e-4:
            p = Jv.T @ np.linalg.solve(Jv @ Jv.T + self.lam2 * np.eye(3), v_ee / speed)
            tau_raw = np.where(tau_raw * p < 0.0, 0.0, tau_raw)
        self.tau_out = np.clip(tau_raw, -self.tau_max, self.tau_max)
        return self.tau_out.copy()

    def summary(self) -> str:
        """Live view for the controller status line."""
        F = self.F
        tx = self.tau_out if self.tau_out is not None else np.zeros(6)
        return (f"F [{F[0]:+.1f} {F[1]:+.1f} {F[2]:+.1f}]N tx "
                + " ".join(f"{x:+.2f}" for x in tx))
