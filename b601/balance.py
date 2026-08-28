"""Balanced drag: momentum-observer external-torque estimate + Cartesian inertia shaping.

Derivation and numbers: the "B601 Balanced Drag" artifact (2026-08-27). Summary:

  plant     M qdd + C qd + g + tau_fric = tau_motor + tau_hand
  observer  p = M qd;  p_hat integrates (tau_motor - beta + r),  beta = g_cal - C^T qd
            r = Ko (p - p_hat)  ->  low-passed (tau_hand - tau_fric): the net drive
            beyond friction. Needs no acceleration and no torque sensor (the RobStride
            torque echo is the command, verified on the drag logs).
  target    an isotropic virtual mass at the hand: Lam_d = diag(m_d I3, i_roll, i_rot, i_rot)
            in the tool frame, so a straight push translates instead of folding the wrist.
  law       tau += K(q) r_net,   K = M Md^-1 - I,  Md = J^T Lam_d J.
            Rows of K r with the sign of the hand's torque on that joint ASSIST (arm
            heavier than target there); opposite-sign rows RESIST (the runaway-wrist
            directions). eig(M Md^-1) = eig(Lam Lam_d^-1): the gains are "how much
            heavier than the target, direction by direction".

Safety structure (each is a hard result of the derivation, not tuning):
  * with an exact model the scheme is feed-forward of the hand's own force - every
    instability is proportional to inertia-model error and unmodelled delay;
  * at the ~90 Hz loop with 2 unmodelled cycles + 20 % inertia error the stable limit
    is kappa ~ 2.4 at a 3 Hz observer, hence KAPPA_HARD_MAX = 2 (arm at most 3x lighter)
    and eigen-clipping of K to [-resist, kappa];
  * the URDF M has no rotor inertia = an UNDER-estimate, which is the safe error
    direction (over-estimating is what feeds acceleration back positively);
  * dead-band + rest-time bias learning keep the gravity-model residual from becoming
    a phantom hand; the friction feed-forward is gated on sign(r * qd) so a joint that
    drives itself (r goes negative against its own motion) shuts its own gate - the
    creep mode of plain fric_comp cannot happen;
  * output ramps in over ramp_s, fades to zero near singularities (cond J 25 -> 40),
    is clamped per joint, and a runaway detector (kinetic energy rising while the
    estimated hand power is <= 0) halves the gain each trip. vel_abort/HOLD remain.

kappa = 0 is observe-only: output exactly zero, r logged - the first rollout step.
"""
from __future__ import annotations

import numpy as np

# assumed Coulomb friction for the gated feed-forward (N.m): RS-06 from the capture
# residuals, wrist from Seeed's sweep. Re-measure with the 'f' key; keep <= the real value.
FRIC = np.array([0.50, 0.50, 0.50, 0.30, 0.21, 0.21])
# dead-band on r: ~2x the gravity-fit residual rms per joint (see calib.csv fits)
R0 = np.array([0.10, 0.10, 0.10, 0.05, 0.05, 0.05])
KAPPA_HARD_MAX = 2.0     # stability ceiling at ~90 Hz (see module docstring)


class BalancedDrag:
    def __init__(self, dyn, *, kappa: float = 2.0, m_d: float = 1.8, i_rot: float = 0.06,
                 i_roll: float = 0.0005, f_o: float = 3.0, resist: float | None = None,
                 fric_scale: float = 0.0, tau_cap: np.ndarray | None = None,
                 r0: np.ndarray | None = None, fric: np.ndarray | None = None,
                 f_static: np.ndarray | None = None, fric_mu: np.ndarray | None = None,
                 f_static_mu: np.ndarray | None = None, v_stribeck: float = 0.15,
                 sustain_joints: np.ndarray | None = None, sustain_margin: float = 0.20,
                 v_hold: float = 0.10,
                 v0: float = 0.08, ramp_s: float = 2.0, dls_lambda: float = 0.05,
                 bias_tau: float = 5.0, r_net_max: float = 3.0) -> None:
        if not 0.0 <= kappa <= KAPPA_HARD_MAX:
            raise ValueError(f"kappa in [0, {KAPPA_HARD_MAX}] (stability limit of the ~90 Hz loop)")
        self.dyn = dyn
        self.n = int(dyn.nq)
        self.kappa = float(kappa)
        # default resist floor mirrors the assist ratio: directions may get (1+kappa)x
        # lighter and at most (1+kappa)x heavier
        self.resist = float(kappa / (1.0 + kappa)) if resist is None else float(resist)
        if not 0.0 <= self.resist <= 0.7:
            raise ValueError("resist in [0, 0.7] (-1 would be a lock)")
        # Lam_d in the TOOL frame: local x is the roll/j6 axis. i_roll defaults to ~the
        # URDF's own roll inertia (leave roll natural) until the rotor inertia is identified.
        self.lam_d = np.array([m_d, m_d, m_d, i_roll, i_rot, i_rot], float)
        self._lam_goal = self.lam_d.copy()   # set_target slews lam_d here (tau ~0.5 s)
        self.Ko = 2.0 * np.pi * float(f_o)
        # two-level (Stribeck) friction feed-forward: fric = kinetic level, f_static = breakaway
        # level (defaults to kinetic). Both scaled by fric_scale (0.85 = compensate 85 %).
        raw_kin = (FRIC[: self.n] if fric is None else np.asarray(fric, float)).copy()
        self.fric = raw_kin * float(fric_scale)
        self.f_static = (self.fric.copy() if f_static is None
                         else np.asarray(f_static, float) * float(fric_scale))
        self._raw_kin = raw_kin
        # load-dependent part (fit_friction.py): level_j(q) = base_j + mu_j * |g_cal_j(q)|,
        # tracking the gear-mesh loss that made the per-pose sweeps spread; mu = 0 -> constant
        raw_mu = (np.zeros(self.n) if fric_mu is None
                  else np.clip(np.asarray(fric_mu, float), 0.0, 0.2))
        self.fric_mu = raw_mu * float(fric_scale)
        self._raw_mu = raw_mu
        self.f_static_mu = (np.zeros(self.n) if f_static_mu is None
                            else np.clip(np.asarray(f_static_mu, float), 0.0, 0.2)) * float(fric_scale)
        self.level_max = 1.5    # N.m cap on any friction level (bad fit must not become a big ff)
        self._g_abs = np.zeros(self.n)
        self.v_stribeck = float(v_stribeck)
        # sustained relief, per joint (default: none). Safe ONLY for joints where a released,
        # coasting joint must merely decelerate - the relief is capped at (raw kinetic level
        # at the pose) - sustain_margin, so any residual/bias below the margin cannot self-drive
        # it. Enabled for j1 (vertical axis: no gravity, constant friction, bias ~0.15 < 0.20):
        # its lever-arm makes close-in lateral drags pay full friction under a drive-only gate.
        self.sustain = (np.zeros(self.n, bool) if sustain_joints is None
                        else np.asarray(sustain_joints, bool))
        self.sustain_margin = float(sustain_margin)
        self.v_hold = float(v_hold)   # rad/s: motion clearly established (self-creep cannot reach it)

        self.r0 = R0[: self.n].copy() if r0 is None else np.asarray(r0, float)
        self.tau_cap = np.full(self.n, 1.0) if tau_cap is None else np.asarray(tau_cap, float)
        self.v0 = float(v0)
        self.ramp_s = float(ramp_s)
        self.lam2 = float(dls_lambda) ** 2
        self.bias_tau = float(bias_tau)
        self.r_net_max = float(r_net_max)

        # observer / estimator state
        self._p_hat: np.ndarray | None = None
        self._r_raw = np.zeros(self.n)
        self.r = np.zeros(self.n)          # bias-corrected estimate (logged as r1..r6)
        self.bias = np.zeros(self.n)
        self._still_t = 0.0
        self._sent = None                  # (tau_ff, kp, kd, pos, velcmd) actually commanded
        self._v_prev: np.ndarray | None = None
        # output state
        self.tau_out = np.zeros(self.n)
        self.alpha = 0.0
        self.trips = 0
        self._trip_scale = 1.0
        self._ramp_t = 0.0
        self._idle = 2                     # cycles since last shaped output (2 => re-ramp)
        self._ke_prev: float | None = None
        self._run_t = 0.0
        self._M = np.eye(self.n)

    # ---- hooks called by GravityDragController -------------------------------------------
    def note_sent(self, tau_ff, kp, kd, pos, velcmd) -> None:
        """Record what was actually commanded this cycle (post-clip); the observer uses it
        next cycle as the motor torque over the interval."""
        self._sent = (np.array(tau_ff, float), np.array(kp, float), np.array(kd, float),
                      np.array(pos, float), np.array(velcmd, float))

    def observe(self, q, v, dt) -> None:
        """Run the estimator without producing torque (any phase but DRAG)."""
        self._observe(q, v, dt)
        self._idle = min(self._idle + 1, 2)
        self.tau_out[:] = 0.0
        self.alpha = 0.0

    def set_target(self, m_d: float | None = None, i_rot: float | None = None,
                   scale: float | None = None) -> tuple[float, float]:
        """Retarget the virtual inertia while running (live keys / caller). The change is
        slewed into lam_d over ~0.5 s in update() so K never steps. Returns (m_d, i_rot)."""
        if scale is not None:
            self._lam_goal = self._lam_goal * float(scale)
        if m_d is not None:
            self._lam_goal[:3] = float(m_d)
        if i_rot is not None:
            self._lam_goal[4:] = float(i_rot)
        # sane bounds: 0.2-10 kg, 0.005-0.5 kg.m^2 (roll entry keeps its ratio via scale only)
        self._lam_goal[:3] = np.clip(self._lam_goal[:3], 0.2, 10.0)
        self._lam_goal[3:] = np.clip(self._lam_goal[3:], 0.0002, 0.5)
        return float(self._lam_goal[0]), float(self._lam_goal[4])

    def update(self, q, v, dt=0.01) -> np.ndarray:
        self.lam_d += (self._lam_goal - self.lam_d) * (dt / (0.5 + dt))
        self._observe(q, v, dt)
        if self._idle >= 2:                # fresh DRAG stretch -> ramp the output back in
            self._ramp_t = 0.0
        self._idle = 0
        self._ramp_t += dt

        # friction feed-forward, gated on the estimated drive agreeing with the motion;
        # sustain-enabled joints (j1) additionally keep margin-capped relief while clearly
        # moving, so steady sliding is relieved there without the hand over-pushing
        gate = np.clip(self.r * np.sign(v) / self.r0, 0.0, 1.0)
        lvl = self.fric_curve(v)
        t = np.tanh(v / self.v0)
        f_ff = t * (lvl * gate)
        if self.sustain.any():
            sus = np.clip((np.abs(v) - self.v_hold) / self.v_hold, 0.0, 1.0) * self.sustain
            raw_lvl = np.minimum(self._raw_kin + self._raw_mu * self._g_abs, self.level_max)
            lvl_sus = np.clip(np.minimum(lvl, raw_lvl - self.sustain_margin), 0.0, None)
            f_ff = t * np.maximum(lvl * gate, lvl_sus * sus)
        # soft dead-band, then the drive the hand would have without the modelled friction
        r_db = np.sign(self.r) * np.maximum(np.abs(self.r) - self.r0, 0.0)
        r_net = np.clip(r_db + f_ff, -self.r_net_max, self.r_net_max)

        K, cond = self._shaping_matrix(q)
        sing = np.clip((40.0 - cond) / 15.0, 0.0, 1.0)      # fade out between cond(J) 25 and 40
        self._runaway_check(v, r_net, dt)
        self.alpha = min(1.0, self._ramp_t / self.ramp_s) * sing * self._trip_scale

        self.tau_out = np.clip(self.alpha * (K @ r_net) + f_ff, -self.tau_cap, self.tau_cap)
        return self.tau_out

    def fric_curve(self, v) -> np.ndarray:
        """Stribeck magnitude per joint at the CURRENT pose: near-breakaway level at the
        onset of motion, decaying to the kinetic level as speed builds. Each level is
        base + mu * |g_cal(q)| (all fric_scale-scaled), so the compensation follows the
        gear load instead of a single all-pose median."""
        f_k = np.minimum(self.fric + self.fric_mu * self._g_abs, self.level_max)
        f_s = np.minimum(self.f_static + self.f_static_mu * self._g_abs, self.level_max)
        w = np.exp(-np.square(np.asarray(v, float) / self.v_stribeck))
        return f_k + (f_s - f_k) * w

    def summary(self) -> str:
        s = "bal r=" + " ".join(f"{x:+.2f}" for x in self.r) + f" a={self.alpha:.2f}"
        if self.trips:
            s += f" trips={self.trips}"
        return s

    # ---- internals -----------------------------------------------------------------------
    def _observe(self, q, v, dt) -> None:
        q = np.asarray(q, float)
        v = np.asarray(v, float)
        M = self.dyn.mass_matrix(q)
        C = self.dyn.coriolis(q, v)
        g_cal = self.dyn.gravity(q)
        self._g_abs = np.abs(g_cal)          # load input for the friction level (fric_curve)
        self._M = M
        p = M @ v
        if self._p_hat is None or self._sent is None:
            self._p_hat = p                                  # anchor: r starts at 0
        else:
            tau_ff, kp, kd, pos, velcmd = self._sent
            v_mid = v if self._v_prev is None else 0.5 * (v + self._v_prev)
            tau_m = tau_ff + kp * (pos - q) + kd * (velcmd - v_mid)
            beta = g_cal - C.T @ v                           # calibrated gravity; friction excluded
            self._p_hat = self._p_hat + dt * (tau_m - beta + self._r_raw)
        self._r_raw = self.Ko * (p - self._p_hat)
        self._v_prev = v.copy()

        # at rest r is pure model bias: learn it slowly, use the corrected value
        if np.all(np.abs(v) < 0.02):
            self._still_t += dt
            if self._still_t > 1.0:
                self.bias += (self._r_raw - self.bias) * dt / self.bias_tau
                self.bias = np.clip(self.bias, -0.5, 0.5)
        else:
            self._still_t = 0.0
        self.r = self._r_raw - self.bias

    def _shaping_matrix(self, q) -> tuple[np.ndarray, float]:
        """K = M Md^-1 - I with eigenvalues clipped to [-resist, kappa], and cond(J)."""
        J = self.dyn.ee_jacobian(q, frame="local")
        JJt = J @ J.T
        ev_j = np.linalg.eigvalsh(JJt)
        cond = float(np.sqrt(max(ev_j.max(), 1e-12) / max(ev_j.min(), 1e-12)))
        Jinv = J.T @ np.linalg.solve(JJt + self.lam2 * np.eye(6), np.eye(6))   # damped inverse
        Md_inv = Jinv @ np.diag(1.0 / self.lam_d) @ Jinv.T
        w, V = np.linalg.eigh(self._M)
        w = np.maximum(w, 1e-8)
        Mh = (V * np.sqrt(w)) @ V.T
        Mhi = (V / np.sqrt(w)) @ V.T
        S = Mh @ Md_inv @ Mh - np.eye(self.n)                # symmetric, similar to M Md^-1 - I
        ws, Vs = np.linalg.eigh(S)
        ws = np.clip(ws, -self.resist, self.kappa)
        return Mh @ ((Vs * ws) @ Vs.T) @ Mhi, cond

    def _runaway_check(self, v, r_net, dt) -> None:
        """Kinetic energy rising while the estimated hand power is <= 0 means the shaping
        drives the arm by itself: halve the gain and re-ramp."""
        ke = 0.5 * float(v @ (self._M @ v))
        rising = self._ke_prev is not None and ke > self._ke_prev + 1e-5
        self._ke_prev = ke
        # the KE floor keeps the observer's ~50 ms lag at push onset from reading as a runaway
        if rising and ke > 0.02 and float(v @ r_net) <= 0.02:
            self._run_t += dt
        else:
            self._run_t = 0.0
        if self._run_t > 0.3:
            self.trips += 1
            self._trip_scale *= 0.5
            self._ramp_t = 0.0
            self._run_t = 0.0
