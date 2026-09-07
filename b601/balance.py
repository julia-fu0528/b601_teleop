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
    a phantom hand; the friction feed-forward follows the paper (tanh(v/v0), no drive
    gate): passivity against calibration error comes from the Cartesian damping D_v
    (eq 37) + kd_drag + the fric_scale headroom (see the note above update());
  * output ramps in over ramp_s, fades to zero near singularities (cond J 25 -> 40),
    is clamped per joint, and a runaway detector (kinetic energy rising while the
    estimated hand power is <= 0) halves the gain each trip. vel_abort/HOLD remain.

kappa = 0 is observe-only: output exactly zero, r logged - the first rollout step.
"""
from __future__ import annotations

import numpy as np

# assumed Coulomb friction for the feed-forward (N.m): RS-06 from the capture
# residuals, wrist from Seeed's sweep. Re-measure with the 'f' key; keep <= the real value.
FRIC = np.array([0.50, 0.50, 0.50, 0.30, 0.21, 0.21])
# dead-band on r: ~2x the gravity-fit residual rms per joint (see calib.csv fits)
R0 = np.array([0.10, 0.10, 0.10, 0.05, 0.05, 0.05])
# 2026-09-07 paper alignment: the drive gate and the per-joint sustain margins were REMOVED.
# Friction relief now applies whenever a joint moves (tanh(v/v0) only), like the paper's tau_f_hat.
# Passivity against calibration error is the job of the Cartesian damping D_v (paper eq 37) plus
# the fric_scale headroom: with the tanh knee v0, a worst-case over-relief Delta injects at most
# Delta*v^2/v0 of power, so joint damping >= Delta/v0 (J^T D_v J + kd_drag) strictly dissipates it.
KAPPA_HARD_MAX = 2.0     # stability ceiling at ~90 Hz (see module docstring)


class BalancedDrag:
    def __init__(self, dyn, *, kappa: float = 2.0, m_d: float = 1.8, i_rot: float = 0.06,
                 i_roll: float = 0.0005, f_o: float = 3.0, resist: float | None = None,
                 fric_scale: float = 0.0, tau_cap: np.ndarray | None = None,
                 r0: np.ndarray | None = None, fric: np.ndarray | None = None,
                 f_static: np.ndarray | None = None, fric_mu: np.ndarray | None = None,
                 f_static_mu: np.ndarray | None = None,
                 f_static_pos: np.ndarray | None = None,   # direction-dependent breakaway (paper eq 28;
                 f_static_neg: np.ndarray | None = None,   # None = symmetric f_static both ways)
                 v_stribeck: float = 0.15,
                 v0: float = 0.08, ramp_s: float = 2.0, dls_lambda: float = 0.05,
                 bias_tau: float = 5.0, r_net_max: float = 3.0,
                 fric_viscous: np.ndarray | None = None,       # (1) viscous B per joint (N.m.s/rad)
                 damp_t: float = 0.0, damp_r: float = 0.0,     # (2) Cartesian D_v (N.s/m, N.m.s/rad), tool frame
                 damp_vsat: float = 0.15,                      # (2) damping saturation knee (rad/s): full
                                                               #     strength below, torque capped above
                 break_beta: float = 0.0, break_vs: float = 0.10,   # (3) breakaway assist strength + decay speed
                 alpha_sigma_v: float = 0.0, alpha_kappa0: float = 0.0,  # (4) alpha scheduling (0 = off)
                 detent_kp: float = 0.0, detent_vlatch: float = 0.08,  # (5) latched low-speed hold
                 gate_floor: float = 1.0) -> None:  # (6) soft intent gate: 1 = pure paper (ungated),
                                                    #     0 = hard drive gate; CLI default 0.5 (hybrid)
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
        # RAW calibrated levels (unscaled); fric_scale is applied live in fric_curve so it
        # can be changed at runtime (set_fric_scale / web slider). load-dependent part
        # (fit_friction.py): level_j(q) = base_j + mu_j * |g_cal_j(q)|.
        self._raw_kin = (FRIC[: self.n] if fric is None else np.asarray(fric, float)).copy()
        self._raw_static = (self._raw_kin.copy() if f_static is None
                            else np.asarray(f_static, float).copy())
        # direction-dependent breakaway (fall back to the symmetric level both ways)
        self._raw_static_pos = (self._raw_static.copy() if f_static_pos is None
                                else np.asarray(f_static_pos, float).copy())
        self._raw_static_neg = (self._raw_static.copy() if f_static_neg is None
                                else np.asarray(f_static_neg, float).copy())
        self._raw_mu = (np.zeros(self.n) if fric_mu is None
                        else np.clip(np.asarray(fric_mu, float), 0.0, 0.2))
        self._raw_static_mu = (np.zeros(self.n) if f_static_mu is None
                               else np.clip(np.asarray(f_static_mu, float), 0.0, 0.2))
        self.fric_scale = float(fric_scale)
        self.level_max = 1.5    # N.m cap on any friction level (bad fit must not become a big ff)
        self._g_abs = np.zeros(self.n)
        self.v_stribeck = float(v_stribeck)
        # ---- paper-aligned additions (all default-off so existing behaviour is unchanged) ----
        self.fric_viscous = (np.zeros(self.n) if fric_viscous is None
                             else np.asarray(fric_viscous, float))       # (1) B in tau_f = f*tanh + B*qd
        self.visc_vsat = 0.3   # rad/s: saturate the B*qd compensation at ~the fv sweep's speed range
                               # (it is anti-damping - never extrapolate it beyond calibration)
        # named alternative KINETIC models for live A/B (statics never swap). Each entry:
        # {"kin": [n], "mu": [n], "viscous": [n]}. Populated by the CLI; empty = no toggle.
        self.fric_models: dict[str, dict] = {}
        self.fric_model = ""
        # live per-term ablation of the kinetic model: load slope mu*|g(q)| and viscous B*qd.
        # True = calibrated values, False = that term contributes 0. Statics unaffected.
        self.mu_on = True
        self.visc_on = True
        self.damp_t = float(damp_t); self.damp_r = float(damp_r)         # (2) Cartesian damping D_v (tool frame)
        self.damp_vsat = float(np.clip(damp_vsat, v0, 0.5))   # >= v0 keeps the passivity proof; <= 0.5
        # so the saturation ceiling d_eff*vsat stays below wrist limit-cycle visibility
        self.break_beta = float(break_beta); self.break_vs = float(break_vs)  # (3) breakaway assist
        self.alpha_sigma_v = float(alpha_sigma_v)                        # (4) velocity schedule (0 = off)
        self.alpha_kappa0 = float(alpha_kappa0)                          # (4) singularity schedule (0 = off)
        # (5) latched low-speed detent: a soft restoring spring toward where the joint went quiet,
        # engaged only near rest (faded out while you guide). Unlike damping this HOLDS against a
        # static bias (e.g. the near-vertical j2 lean), so an un-driven joint settles instead of
        # drifting. This is the paper's K_v(x-x0) term (eq 33), which they zero for teleop, but
        # gated on low speed + latched so it never fights active guiding. 0 = off.
        self.detent_kp = float(detent_kp)
        self.detent_vlatch = float(detent_vlatch)
        self._q_latch: np.ndarray | None = None
        # (6) soft intent gate for the relief: s_int = floor + (1-floor)*clip(r*sign(v)/r0).
        # The observer SEES a real hand push (external torque -> r) but model-error creep sits
        # below the dead-band, so the gate discriminates intent from creep. floor keeps relief
        # continuous (no die-off when the push relaxes, unlike the old hard gate) while un-driven
        # motion meets (1 - fric_scale*floor) of the real friction as a structural brake.
        self.gate_floor = float(np.clip(gate_floor, 0.0, 1.0))
        # live mode toggles (keys b / bf while running)
        self.shaping_on = True
        self.fric_on = True
        self._fs = float(fric_scale)

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

    def set_lam(self, i: int, v: float) -> float:
        """Live-set one diagonal entry of the desired Cartesian inertia Lambda_d (tool frame):
        i=0..2 translation (kg), i=3..5 rotation about x(roll)/y/z (kg.m^2). Slewed into lam_d
        over ~0.5 s in update() so K never steps."""
        i = int(i); v = float(v)
        if 0 <= i < 3:
            v = float(np.clip(v, 0.2, 10.0))
        elif 3 <= i < 6:
            v = float(np.clip(v, 0.0002, 0.5))
        else:
            return 0.0
        self._lam_goal[i] = v
        return v

    def set_kappa(self, k: float) -> float:
        """Live inertia-shaping strength (0..KAPPA_HARD_MAX). k=0 turns shaping off; turning
        it back on restarts the 2 s ramp so K never steps. Mirrors --balance."""
        k = float(np.clip(k, 0.0, KAPPA_HARD_MAX))
        was = self.shaping_on
        self.kappa = k
        self.shaping_on = k > 0.0
        if self.shaping_on and not was:
            self._ramp_t = 0.0
        return k

    def set_fric_scale(self, f: float) -> float:
        """Live friction-compensation fraction (0..1). Mirrors --balance-fric."""
        f = float(np.clip(f, 0.0, 1.0))
        self.fric_scale = f
        self.fric_on = f > 0.0
        if f > 0.0:
            self._fs = f      # keep the 'bf' key's launch-reference in sync
        return f

    def set_damp(self, d_t: float | None = None, d_r: float | None = None,
                 vsat: float | None = None) -> tuple[float, float, float]:
        """Live Cartesian damping D_v (tool frame): translational N.s/m, rotational N.m.s/rad,
        and the saturation knee vsat (rad/s - full damping below, torque capped above)."""
        if d_t is not None:
            self.damp_t = float(max(d_t, 0.0))
        if d_r is not None:
            self.damp_r = float(max(d_r, 0.0))
        if vsat is not None:
            self.damp_vsat = float(np.clip(vsat, self.v0, 0.5))   # [v0, 0.5]: guard proof / limit-cycle lid
        return self.damp_t, self.damp_r, self.damp_vsat

    def set_break(self, beta: float) -> float:
        """Live breakaway-assist fraction [0..1]."""
        self.break_beta = float(np.clip(beta, 0.0, 1.0))
        return self.break_beta

    def set_alpha(self, sigma_v: float | None = None, kappa0: float | None = None) -> tuple[float, float]:
        """Live alpha scheduling: velocity knee sigma_v (rad/s) and singularity K0 (0 = off)."""
        if sigma_v is not None:
            self.alpha_sigma_v = float(max(sigma_v, 0.0))
        if kappa0 is not None:
            self.alpha_kappa0 = float(max(kappa0, 0.0))
        return self.alpha_sigma_v, self.alpha_kappa0

    def set_fric_terms(self, mu: bool | None = None, viscous: bool | None = None) -> tuple[bool, bool]:
        """Live per-term ablation of the kinetic friction model: the load slope mu*|g(q)| and the
        viscous B*qd. True = calibrated, False = term zeroed. Composes with set_fric_model (the
        toggles gate whichever model is active). Statics are unaffected."""
        if mu is not None:
            self.mu_on = bool(mu)
        if viscous is not None:
            self.visc_on = bool(viscous)
        return self.mu_on, self.visc_on

    def set_fric_model(self, name: str) -> str:
        """Live-swap the KINETIC friction model (Coulomb levels + load mu + viscous B) to a named
        entry in self.fric_models. Statics (breakaway) are shared and never swapped. Returns the
        active model name (unchanged if `name` is unknown)."""
        m = self.fric_models.get(name)
        if m is None:
            return self.fric_model
        self._raw_kin = np.asarray(m["kin"], float).copy()
        self._raw_mu = np.asarray(m["mu"], float).copy()
        self.fric_viscous = np.asarray(m["viscous"], float).copy()
        self.fric_model = name
        return self.fric_model

    def set_gate_floor(self, f: float) -> float:
        """Live soft-intent-gate floor [0..1]: 1 = pure paper (ungated relief), 0 = hard drive
        gate, between = hybrid (un-driven motion gets floor*relief)."""
        self.gate_floor = float(np.clip(f, 0.0, 1.0))
        return self.gate_floor

    def set_detent(self, kp: float) -> float:
        """Live-set the latched low-speed detent stiffness (N.m/rad); 0 = off, re-latches on next quiet."""
        self.detent_kp = float(max(kp, 0.0))
        self._q_latch = None
        return self.detent_kp

    def update(self, q, v, dt=0.01) -> np.ndarray:
        self.lam_d += (self._lam_goal - self.lam_d) * (dt / (0.5 + dt))
        self._observe(q, v, dt)
        if self._idle >= 2:                # fresh DRAG stretch -> ramp the output back in
            self._ramp_t = 0.0
            self._q_latch = None           # re-latch the detent at the new starting pose
        self._idle = 0
        self._ramp_t += dt

        # Jacobian + cond(J) once (shared by shaping, Cartesian damping, and alpha-kappa schedule)
        J = self.dyn.ee_jacobian(q, frame="local")
        ev_j = np.linalg.eigvalsh(J @ J.T)
        cond = float(np.sqrt(max(ev_j.max(), 1e-12) / max(ev_j.min(), 1e-12)))
        r_db = np.sign(self.r) * np.maximum(np.abs(self.r) - self.r0, 0.0)   # dead-banded hand estimate

        # (4) alpha scheduling: taper friction comp near zero velocity (avoid stiction chatter) and
        #     near singularities (avoid amplifying uncertain friction). 0 -> disabled (=1).
        s_v = 1.0 if self.alpha_sigma_v <= 0 else (1.0 - np.exp(-(v / self.alpha_sigma_v) ** 2))
        s_k = 1.0 if self.alpha_kappa0 <= 0 else float(np.clip(self.alpha_kappa0 / max(cond, 1e-6), 0.0, 1.0))

        # friction feed-forward, paper-style (eq 13): applied whenever the joint moves, direction
        # and taper from tanh(v/v0) alone - NO drive gate, NO sustain margins (relief is inherently
        # "sustained"). Passivity against calibration error is guaranteed by the Cartesian damping
        # D_v below (eq 37) + kd_drag + the fric_scale headroom: a worst-case over-relief Delta
        # injects at most Delta*v^2/v0 of power near rest, so joint damping >= Delta/v0 dissipates it.
        # (6) hybrid soft intent gate: full paper relief while the hand clearly drives a joint,
        # floor*relief when it does not (motor creep, bias, coasting) - see the constructor note.
        gate = np.clip(self.r * np.sign(v) / self.r0, 0.0, 1.0)
        s_int = self.gate_floor + (1.0 - self.gate_floor) * gate
        lvl = self.fric_curve(v) if self.fric_on else np.zeros(self.n)
        f_ff = np.tanh(v / self.v0) * lvl * s_int
        f_ff = f_ff * s_v * s_k
        # (1) viscous friction comp: cancel B*qd (paper eq 13). Self-zeroing at rest, so ungated -
        # but SATURATED at the identification range (the fv sweep only measured |qd| <= ~0.2 rad/s).
        # This term is anti-damping (+0.85*B*qd, positive velocity feedback); extrapolating it
        # linearly ran j6 (B 0.112 vs kd 0.05: net slope +0.045*qd) away to vel_abort at 4 rad/s.
        # Saturating caps the injection at 0.85*B*visc_vsat (~0.03 N.m) - a feel term, not a driver.
        if self.fric_on and self.visc_on and np.any(self.fric_viscous):
            vv = self.visc_vsat
            f_ff = f_ff + self.fric_scale * self.fric_viscous * vv * np.tanh(v / vv) * s_int * s_k
        # (3) breakaway assist: when the hand is pushing (r_db != 0) pre-pay a fraction of the
        #     static breakaway in that direction, decaying as the joint gets moving. Direction
        #     from sign(r_db) - no F/T sensor needed. Helps proximal joints break free.
        if self.fric_on and self.break_beta > 0.0:
            # direction-dependent breakaway (paper eq 28): the hand's intended direction sign(r_db)
            # selects which threshold to pre-pay - cheap gearboxes break away asymmetrically
            raw_dir = np.where(r_db >= 0.0, self._raw_static_pos, self._raw_static_neg)
            f_static = np.minimum(raw_dir + self._raw_static_mu * self._g_abs, self.level_max)
            decay = np.exp(-np.abs(v) / max(self.break_vs, 1e-3))
            f_ff = f_ff + self.break_beta * self.fric_scale * f_static * np.sign(r_db) * decay * s_k

        r_net = np.clip(r_db + f_ff, -self.r_net_max, self.r_net_max)

        # (2) Cartesian virtual damping D_v (passivity-aware): tau = -J^T D_v J qd. Strictly
        #     dissipative (qd . tau <= 0), so it can dominate friction-estimate error and keep
        #     the rendered interaction passive - THE guard for the ungated relief (eq 37).
        #     Two robustness deviations from the raw J^T D_v J: (a) DIAGONAL only - the
        #     off-diagonal terms torque *other* joints, and under worst-case over-relief each
        #     recruited joint gets its own relief, a positive-feedback channel (seen in sim);
        #     the diagonal keeps eq-37's Cartesian sizing while staying per-joint dissipative.
        #     (b) discrete-time cap: DELAYED damping (one cycle + the velocity-filter lag)
        #     destabilizes a joint once d*dt/M_jj approaches ~0.3 (light wrists first; seen on
        #     hardware as a wrist limit cycle whose amplitude is bounded by the saturation
        #     ceiling d_eff*vsat - raising the knee raised the lid). Cap at 0.15*M_jj/dt.
        #     (c) SATURATING, not linear: the guard is only needed near zero velocity (over-relief
        #     injection is capped at Delta*tanh(v/v0), so its danger zone is slow creep). A linear
        #     damper sized for that (~1 N.m.s/rad) costs ~1 N.m of drag at guiding speed - the arm
        #     feels rigid. tau = -d_eff * vsat * tanh(v/vsat): slope d_eff at rest (guard intact -
        #     dissipation >= injection for all v when vsat >= v0), felt drag capped at d_eff*vsat
        #     (~0.15-0.2 N.m). 'damp_vs <rad/s>' tunes the knee live.
        tau_damp = np.zeros(self.n)
        if self.damp_t > 0.0 or self.damp_r > 0.0:
            Dv = np.diag([self.damp_t] * 3 + [self.damp_r] * 3)
            Dj = J.T @ (Dv @ J)
            cap = 0.15 * np.maximum(np.diag(self._M), 1e-6) / max(dt, 1e-3)
            vs = self.damp_vsat
            tau_damp = -np.minimum(np.diag(Dj), cap) * vs * np.tanh(v / vs)

        # (5) latched detent: hold the pose whenever the hand is NOT driving the joint. The latch
        # follows q while you push (r_db != 0) and freezes when you release, so the spring holds
        # against an un-driven drift (the near-vertical lean) at any speed, but never fights guiding.
        # Gating on the hand estimate r (not velocity) is what lets it arrest a bias that would
        # otherwise accelerate past a velocity threshold.
        tau_detent = np.zeros(self.n)
        if self.detent_kp > 0.0:
            if self._q_latch is None:
                self._q_latch = q.copy()
            driven = np.abs(r_db) > 0.0
            self._q_latch = np.where(driven, q, self._q_latch)          # follow the hand; freeze when released
            tau_detent = self.detent_kp * (~driven) * (self._q_latch - q)

        self._runaway_check(v, r_db, dt)
        if self.shaping_on:
            K = self._shaping_matrix(J)
            sing = np.clip((40.0 - cond) / 15.0, 0.0, 1.0)  # fade shaping between cond(J) 25 and 40
            self.alpha = min(1.0, self._ramp_t / self.ramp_s) * sing * self._trip_scale
            self.tau_out = np.clip(self.alpha * (K @ r_net) + f_ff + tau_damp + tau_detent, -self.tau_cap, self.tau_cap)
        else:
            self.alpha = 0.0
            self.tau_out = np.clip(f_ff + tau_damp + tau_detent, -self.tau_cap, self.tau_cap)
        return self.tau_out

    def fric_curve(self, v) -> np.ndarray:
        """Stribeck magnitude per joint at the CURRENT pose: near-breakaway level at the
        onset of motion, decaying to the kinetic level as speed builds. Each level is
        base + mu * |g_cal(q)| (all fric_scale-scaled), so the compensation follows the
        gear load instead of a single all-pose median."""
        s = self.fric_scale
        vv = np.asarray(v, float)
        f_k = np.minimum((self._raw_kin + self.mu_on * self._raw_mu * self._g_abs) * s, self.level_max)
        raw_s = np.where(vv >= 0.0, self._raw_static_pos, self._raw_static_neg)   # eq 28 by motion dir
        f_s = np.minimum((raw_s + self._raw_static_mu * self._g_abs) * s, self.level_max)
        w = np.exp(-np.square(vv / self.v_stribeck))
        return f_k + (f_s - f_k) * w

    def toggle_mode(self, code: str) -> str:
        """Live mode keys: 'b' = inertia shaping, 'bf' = friction comp.
        Returns the message to print. Re-enabling shaping restarts the 2 s ramp."""
        if code == "b":
            self.shaping_on = not self.shaping_on
            if self.shaping_on:
                self._ramp_t = 0.0
                return f"inertia shaping ON (kappa {self.kappa:g}, ramping in over {self.ramp_s:g} s)"
            return "inertia shaping OFF (friction comp unchanged; b to re-enable)"
        if code == "bf":
            if self._fs <= 0.0:
                return "friction ff was launched at 0 (--balance-fric 0) - restart to enable it"
            self.fric_on = not self.fric_on
            return (f"friction compensation ON (x{self._fs:g} of the calibrated levels)"
                    if self.fric_on else "friction compensation OFF")
        return "unknown mode key (b = shaping, bf = friction)"

    def summary(self) -> str:
        s = "bal r=" + " ".join(f"{x:+.2f}" for x in self.r) + f" a={self.alpha:.2f}"
        off = [nm for nm, on in (("shape", self.shaping_on), ("fric", self.fric_on)) if not on]
        if off:
            s += " OFF:" + ",".join(off)
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

    def _shaping_matrix(self, J) -> np.ndarray:
        """K = M Md^-1 - I with eigenvalues clipped to [-resist, kappa] (J passed in)."""
        JJt = J @ J.T
        Jinv = J.T @ np.linalg.solve(JJt + self.lam2 * np.eye(6), np.eye(6))   # damped inverse
        Md_inv = Jinv @ np.diag(1.0 / self.lam_d) @ Jinv.T
        w, V = np.linalg.eigh(self._M)
        w = np.maximum(w, 1e-8)
        Mh = (V * np.sqrt(w)) @ V.T
        Mhi = (V / np.sqrt(w)) @ V.T
        S = Mh @ Md_inv @ Mh - np.eye(self.n)                # symmetric, similar to M Md^-1 - I
        ws, Vs = np.linalg.eigh(S)
        ws = np.clip(ws, -self.resist, self.kappa)
        return Mh @ ((Vs * ws) @ Vs.T) @ Mhi

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
