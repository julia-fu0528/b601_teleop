"""Gravity-compensated drag-teach controller (MIT mode, feed-forward only).

Phases
  RAMP    stiff PD hold at the start pose while the gravity feed-forward ramps 0 -> 100 % (ramp_s).
  FADE    feed-forward stays at 100 % while the PD hold fades to the drag gains (fade_s).
          Nobody touches the arm during RAMP/FADE. Any joint drifting > ramp_window rad from the
          start pose => HOLD (wrong sign / bad model).
  DRAG    tau = g(q) (+ optional light damping). Drag the arm by hand.
          Any joint faster than vel_abort => HOLD.
  HOLD    PD hold at the frozen pose + g(q) feed-forward. Never torque-off: a loaded arm free-falls.
  RELEASE torque fade over release_s with strong damping, then all motors disabled.
  CAPTURE (key c, from DRAG) hold the current pose, then sweep every joint +/- capture_amp rad as a slow
          triangle wave (period 2 s) so Coulomb friction changes sign and averages out; the mean torque
          the PD has to add per joint is the model error at this pose. Appended to the calibration CSV,
          then back to DRAG. Use scripts/fit_gravity.py on that CSV to fit per-joint scales.
  DONE

  FRICTION (key f, from DRAG) like CAPTURE but a larger, slower sweep (+/- 0.15 rad at ~0.15 rad/s) whose
          PD residual is split by direction of motion: kinetic Coulomb friction = (resid+ - resid-)/2 per joint.
  STATIC  (key s, from DRAG) breakaway-friction sweep (paper 4.3): joints one at a time, all others held
          stiff; the free joint's torque ramps slowly until |qd| stays above 0.03 rad/s for ~60 ms (the
          sustained window rejects backlash take-up), in both directions. Records the torque at the window
          ONSET (t+). Keeps tau+ and tau- separately (eq 28: direction-dependent breakaway) AND the
          symmetric average (gravity residual cancels). Prints paste-ready fric_static/-_pos/-_neg lines.
          Hands off during the sweep (~10 s per joint).
  VFRIC   (key fv, from DRAG) multi-speed CURRENT-BASED sweep: constant-velocity triangles at several speeds,
          reading the motor torque feedback (torq = K_t*iq, eq 22) minus gravity = friction(+/-v). Rows go to
          friction_v.csv; scripts/fit_friction.py regresses tau_c*tanh(v/eps) + B*v + c0 -> viscous B and a
          de-biased tau_c (c0 soaks up residual gravity/current offset). Hands off (~30-60 s). Repeat at 3-6 poses.
Keys while running (type + Enter): d = drag, h = hold, c = capture, f = friction sweep, fv = velocity-friction sweep, s = static-friction sweep, r = release, q! = disable NOW.
With --balance also: m <kg> = set virtual mass, i <kg.m^2> = set virtual rot. inertia, + / - = 25 % heavier / lighter;
b / bf / bs = toggle inertia shaping / friction compensation / j1 sustained relief live (state printed).
Ctrl+C: DRAG/RAMP -> HOLD, HOLD -> RELEASE, RELEASE -> disable now.
"""
from __future__ import annotations

import csv
import enum
import os
import queue
import signal
import sys
import threading
import time

import numpy as np

from .arm import KD_MAX, PositionSource, VelocityEstimator
from .config import Config
from .dynamics import ArmDynamics


class Phase(enum.Enum):
    RAMP = "ramp"
    FADE = "fade"
    DRAG = "drag"
    HOLD = "hold"
    RELEASE = "release"
    CAPTURE = "capture"
    STATIC = "static"
    DONE = "done"


class GravityDragController:
    def __init__(
        self,
        arm,
        dyn: ArmDynamics,
        cfg: Config,
        *,
        scale: float = 1.0,
        kd: np.ndarray | None = None,
        kp: float | None = None,
        fric_scale: float = 1.0,
        active: np.ndarray | None = None,
        gripper_hold: bool = True,
        duration: float | None = None,
        auto_release: bool = False,
        log_path: str | None = None,
        calib_path: str | None = None,
        fric_path: str | None = None,
        capture_s: float = 5.0,
        capture_amp: float = 0.05,
        vfric_path: str | None = None,
        vfric_speeds: tuple[float, ...] = (0.03, 0.06, 0.09, 0.12, 0.16, 0.20),
        vfric_period: float = 3.0,
        vfric_amp_max: float = 0.35,
        vfric_periods: int = 2,
        print_every: float = 0.5,
        interactive: bool = True,
        realtime: bool = True,
        hold_timeout: float | None = None,
        assist=None,
    ) -> None:
        self.arm, self.dyn, self.cfg = arm, dyn, cfg
        self.n = dyn.nq
        self.scale = float(scale)
        self.kp_drag = np.array([j.kp_drag for j in cfg.joints], float) if kp is None else np.full(len(cfg.joints), float(kp))
        self.kd_drag = np.array([j.kd_drag for j in cfg.joints], float) if kd is None else np.asarray(kd, float)
        self.tau_max = np.array([j.tau_max for j in cfg.joints], float)
        self.fric_comp = np.clip(np.array([j.fric_comp for j in cfg.joints], float) * float(fric_scale), 0.0, 0.6)
        self.fric_v0 = 0.08   # rad/s: tanh knee of the friction feed-forward (above the FD velocity noise)
        self.hold_kp = np.array([j.hold_kp for j in cfg.joints], float)
        self.hold_kd = np.array([j.hold_kd for j in cfg.joints], float)
        self.release_kd = np.minimum(KD_MAX, np.maximum(3.0 * self.hold_kd, self.hold_kd))
        self.active = np.ones(self.n, bool) if active is None else np.asarray(active, bool)
        self.gripper_hold = bool(gripper_hold) and cfg.gripper is not None and getattr(arm, "gripper", None) is not None
        self.duration = duration
        self.auto_release = auto_release
        self.log_path = log_path
        self.calib_path = calib_path
        self.fric_path = fric_path
        self.capture_s = float(capture_s)
        self.capture_amp = float(capture_amp)
        # multi-speed current-based (torq-g) friction sweep -> B (viscous) + de-biased tau_c (eq 22-25)
        self.vfric_path = vfric_path
        self._vf_speeds = [abs(float(s)) for s in vfric_speeds if float(s) > 0]
        self._vf_period = float(vfric_period)      # fixed triangle period; amp = speed*period/4 (const-velocity dwell)
        self._vf_amp_max = float(vfric_amp_max)    # amplitude safety cap (rad) -> shortens the period at high speed
        self._vf_nper = max(1, int(vfric_periods))
        self._vf_rows: list[tuple] = []
        self.captures: list[dict] = []
        self.print_every = print_every
        self.interactive = interactive
        self.realtime = realtime
        self.startup_failed = False
        self.hold_timeout = hold_timeout   # HOLD -> RELEASE automatically after this many s (None = wait for operator)
        self.t_hold: float | None = None
        self.assist = assist               # TorqueRebalance or None (b601/assist.py)

        self.loop = cfg.loop
        self.dt_nom = 1.0 / self.loop.rate_hz
        self.phase = Phase.RAMP
        self.freeze_reason: str | None = None
        self.events: list[tuple[float, str]] = []
        self.q_drag_end: np.ndarray | None = None
        self.q_now: np.ndarray | None = None
        self.telem: dict | None = None
        self.v_drag_end: np.ndarray | None = None
        self._cmds: "queue.Queue[str]" = queue.Queue()
        self._sim_t = 0.0

    # ---- helpers -------------------------------------------------------------------------
    def _now(self) -> float:
        return time.perf_counter() if self.realtime else self._sim_t

    def _say(self, msg: str) -> None:
        self.events.append((self._now(), msg))
        print(msg, flush=True)

    def _on_sigint(self, *_):
        self._cmds.put("<sigint>")

    def _stdin_thread(self) -> None:
        try:
            for line in sys.stdin:
                self._cmds.put(line.strip().lower())
        except Exception:
            pass

    def _freeze(self, q: np.ndarray, why: str) -> None:
        if self.phase in (Phase.HOLD, Phase.RELEASE, Phase.DONE):
            return
        if self.phase in (Phase.RAMP, Phase.FADE):
            self.startup_failed = True
        self.freeze_reason = why
        self.q_hold = q.copy()
        self.t_hold = self._now()
        self.phase = Phase.HOLD
        self._say(f"\n*** HOLD: {why}\n"
                  f"*** The arm is now STIFF ON PURPOSE (PD hold + gravity feed-forward, not torque-off).\n"
                  f"*** Type  d + Enter  to go back to DRAG,  r + Enter  to release (fade & disable),  q! + Enter  to disable NOW.")

    def _append_fric(self, kind: str, pose_q, vals, resid) -> None:
        """Append one sweep (kinetic 'f' or static 's') to the friction CSV so several
        poses can be aggregated with scripts/fit_friction.py."""
        if not self.fric_path:
            return
        new = not os.path.exists(self.fric_path)
        with open(self.fric_path, "a", newline="") as cf:
            w = csv.writer(cf)
            if new:
                w.writerow(["kind"] + [f"q{i+1}" for i in range(self.n)]
                           + [f"f{i+1}" for i in range(self.n)] + [f"resid{i+1}" for i in range(self.n)])
            w.writerow([kind] + [f"{x:.5f}" for x in pose_q]
                       + [f"{x:.4f}" for x in vals] + [f"{x:.4f}" for x in resid])
        self._say(f"  appended to {self.fric_path} (repeat at 4-6 spread poses, then scripts/fit_friction.py)")

    def _vf_est_s(self) -> float:
        """Rough wall-clock for one velocity-friction sweep (all speeds), for the operator prompt."""
        return float(sum(1.0 + self._vf_nper * min(self._vf_period, 4.0 * self._vf_amp_max / s)
                         for s in self._vf_speeds))

    def _append_vfric(self, rows) -> None:
        """Append a multi-speed current-based sweep to the velocity-friction CSV. One row per
        (speed, direction, pose): the signed commanded speed 'vset' (a pairing label), the pose,
        the MEASURED per-joint velocity mv (actual speed reached), and torq-g per joint. The
        fitter pairs +/-vset per pose, so mv (not vset) is the x-axis -> immune to tracking lag."""
        path = self.vfric_path or "friction_v.csv"
        new = not os.path.exists(path)
        with open(path, "a", newline="") as cf:
            w = csv.writer(cf)
            if new:
                w.writerow(["vset"] + [f"q{i+1}" for i in range(self.n)]
                           + [f"mv{i+1}" for i in range(self.n)] + [f"f{i+1}" for i in range(self.n)])
            for vset, q, mv, f in rows:
                w.writerow([f"{vset:.5f}"] + [f"{x:.5f}" for x in q]
                           + [f"{x:.5f}" for x in mv] + [f"{x:.4f}" for x in f])
        self._say(f"  appended {len(rows)} rows to {path}")

    def _static_init(self, q, t) -> None:
        """Key 's': measure breakaway (static) friction per active joint, both directions."""
        self._st_joints = [i for i in range(self.n) if self.active[i]]
        self._st_ji = 0
        self._st_stage = "settle"
        self._st_dir = 1
        self._st_tau = 0.0
        self._st_anchor = 0.0
        self._st_vcnt = 0          # consecutive cycles with |qd| above the breakaway threshold
        self._st_tau_onset = 0.0   # ramp torque at the onset of the sustained-velocity window (t+)
        self._st_hold = q.copy()
        self._st_t = t
        self._st_res: dict[int, dict[int, float]] = {}
        big = np.array([j.model == "rs-06" for j in self.cfg.joints])
        self._st_rate = np.where(big, 0.25, 0.12)   # N.m/s torque ramp
        self._st_cap = np.where(big, 2.0, 1.0)      # give up beyond this (limit / contact)
        self.phase = Phase.STATIC
        self._say(f"  static sweep: {self.cfg.joint_names[self._st_joints[0]]}")

    def _static_finish(self, names) -> None:
        self.static_result = np.full(self.n, np.nan)
        self.static_pos = np.full(self.n, np.nan)    # breakaway toward +q (paper eq 26)
        self.static_neg = np.full(self.n, np.nan)    # breakaway toward -q (paper eq 27)
        resid = np.full(self.n, np.nan)
        kin = np.array([j.fric_kinetic for j in self.cfg.joints], float)
        if getattr(self, "friction_result", None) is not None:
            kin = np.where(np.isfinite(self.friction_result), self.friction_result, kin)
        lines = ["static-friction sweep done (breakaway torque per joint, per direction):",
                 "  joint    tau+     tau-     f_static  g-resid   kinetic('f'/config)"]
        for i in self._st_joints:
            tp = self._st_res.get(i, {}).get(1, float("nan"))
            tm = self._st_res.get(i, {}).get(-1, float("nan"))
            self.static_pos[i], self.static_neg[i] = tp, tm
            self.static_result[i] = 0.5 * (tp + tm)
            resid[i] = 0.5 * (tm - tp)
            lines.append(f"  {names[i]:8s} {tp:+.3f}   {-tm:+.3f}   {self.static_result[i]:8.3f} "
                         f"{resid[i]:+8.3f}   {kin[i] if kin[i] > 0 else float('nan'):8.3f}")
        lines.append("paste into config/b601_rs.toml under each [[joint]] (raw values; the 85 % factor")
        lines.append("is applied at runtime by --balance-fric). NOTE: the per-direction values each")
        lines.append("carry the gravity residual at this pose; the symmetric average cancels it.")
        for i in self._st_joints:
            kin_s = f"  fric_kinetic = {kin[i]:.2f}" if kin[i] > 0 else "  # fric_kinetic: run the 'f' sweep"
            lines.append(f"  {names[i]:8s}: fric_static = {self.static_result[i]:.2f}"
                         f"  fric_static_pos = {self.static_pos[i]:.2f}"
                         f"  fric_static_neg = {self.static_neg[i]:.2f}{kin_s}")
        lines.append("sanity: f_static >= kinetic; a large |g-resid| means the gravity model is off at this pose")
        self._say("\n".join(lines))
        self._append_fric("static", self._st_hold, self.static_result, resid)
        self._append_fric("static_pos", self._st_hold, self.static_pos, resid)
        self._append_fric("static_neg", self._st_hold, self.static_neg, resid)

    # ---- main ----------------------------------------------------------------------------
    def run(self) -> Phase:
        arm, dyn, cfg = self.arm, self.dyn, self.cfg
        names = cfg.joint_names
        n = self.n

        q0 = arm.read_q()
        viol = dyn.limit_violation(q0)
        if np.any(viol > 0.05):
            raise RuntimeError(f"start pose outside URDF joint limits by {np.round(viol, 3)} rad: q={np.round(q0, 3)}")
        g0 = dyn.gravity(q0) * self.scale
        print("start pose q (rad):", np.round(q0, 3))
        print("feed-forward at start (N.m):", np.round(np.clip(g0, -self.tau_max, self.tau_max), 3),
              "| clamps:", self.tau_max, "| active:", [nm for nm, a in zip(names, self.active) if a])
        print(f"kd_drag={self.kd_drag}  kp_drag={self.kp_drag}  fric_comp={self.fric_comp}  scale={self.scale}  rate={self.loop.rate_hz} Hz  "
              f"ramp={self.loop.ramp_s}s  vel_abort={self.loop.vel_abort} rad/s")

        gq0 = arm.read_gripper_q() if self.gripper_hold else 0.0

        if self.interactive:
            signal.signal(signal.SIGINT, self._on_sigint)
            threading.Thread(target=self._stdin_thread, name="keys", daemon=True).start()

        logf = None
        writer = None
        if self.log_path:
            logf = open(self.log_path, "w", newline="")
            writer = csv.writer(logf)
            writer.writerow(["t", "phase"] + [f"q{i+1}" for i in range(n)] + [f"v{i+1}" for i in range(n)]
                            + [f"tau{i+1}" for i in range(n)] + [f"torq{i+1}" for i in range(n)]
                            + [f"velfb{i+1}" for i in range(n)] + ["t_rotor_max"]
                            + ([f"r{i+1}" for i in range(n)] if hasattr(self.assist, "r") else []))

        arm.prepare_mit(self.active, self.gripper_hold)
        arm.enable(self.active, self.gripper_hold)
        # first command immediately: stiff hold at the current pose (RAMP start), no torque-less gap after enable
        arm.send_mit(q0, np.zeros(n), self.hold_kp, self.hold_kd, np.zeros(n), self.active)
        self._say(f"motors enabled; RAMP {self.loop.ramp_s}s + FADE {self.loop.fade_s}s — hands off the arm")

        vel = VelocityEstimator(n)
        src = PositionSource(arm, self.active)
        self.src = src
        q = q0.copy()
        q_start = q0.copy()
        self.q_hold = q0.copy()
        t0 = self._now()
        t_phase = t0
        t_print = t0
        t_remind = t0
        t_prev = t0
        fails = 0
        cycles = 0
        tau_cmd = np.zeros(n)
        torq = np.full(n, np.nan)
        t_rot = np.nan

        try:
            while self.phase is not Phase.DONE:
                if not self.realtime:
                    self._sim_t += self.dt_nom
                t = self._now()
                dt = max(t - t_prev, 1e-4) if self.realtime else self.dt_nom
                t_prev = t

                # ---- read (MIT feedback for enabled joints, mechPos params for the rest)
                try:
                    q = src.update()
                    fails = 0
                    while src.warnings:
                        self._say("WARNING: " + src.warnings.pop(0))
                except Exception as e:  # bus timeout etc.
                    fails += 1
                    if fails >= self.loop.max_read_failures:
                        self._freeze(q, f"{fails} consecutive position read failures ({e})")
                v = vel.update(q, dt)
                self.q_now = q                    # latest joint config, for the --serve web panel
                g = dyn.gravity(q) * self.scale

                # ---- commands
                while True:
                    try:
                        c = self._cmds.get_nowait()
                    except queue.Empty:
                        break
                    if c == "<sigint>":
                        if self.phase in (Phase.RAMP, Phase.FADE, Phase.DRAG, Phase.CAPTURE, Phase.STATIC):
                            self._freeze(q, "Ctrl+C")
                        elif self.phase is Phase.HOLD:
                            self._say("Ctrl+C in HOLD -> RELEASE (torque fade, then disable)")
                            self.phase, t_phase = Phase.RELEASE, t
                        else:
                            self._say("Ctrl+C in RELEASE -> disabling NOW")
                            arm.disable_all()
                            self.phase = Phase.DONE
                    elif c in ("h", "hold"):
                        self._freeze(q, "user requested hold")
                    elif c in ("d", "drag"):
                        if self.phase is Phase.HOLD:
                            self.phase = Phase.DRAG
                            self.freeze_reason = None
                            vel.reset()
                            self._say("DRAG resumed (feed-forward only)")
                    elif c in ("c", "capture", "f", "friction"):
                        if self.phase is Phase.DRAG:
                            self.q_hold = q.copy()
                            self.phase, t_phase = Phase.CAPTURE, t
                            self._cap_mode = "friction" if c in ("f", "friction") else "calib"
                            self._cap_sum = np.zeros(n); self._cap_n = 0
                            self._cap_pos = np.zeros(n); self._cap_npos = 0
                            self._cap_neg = np.zeros(n); self._cap_nneg = 0
                            if self._cap_mode == "friction":
                                self._say("FRICTION sweep: +/-0.15 rad triangle at ~0.15 rad/s for 2 periods — hands off")
                            else:
                                self._say(f"CAPTURE: holding {self.capture_s}s, measuring the PD residual — hands off")
                        else:
                            self._say("capture/friction only work from DRAG (type d first)")
                    elif c in ("fv", "vfric"):
                        if self.phase is Phase.DRAG and self._vf_speeds:
                            self.q_hold = q.copy()
                            self.phase, t_phase = Phase.CAPTURE, t
                            self._cap_mode = "vfric"
                            self._vf_si = 0
                            self._vf_rows = []
                            self._vf_pos = np.zeros(n); self._vf_npos = np.zeros(n)
                            self._vf_neg = np.zeros(n); self._vf_nneg = np.zeros(n)
                            self._vf_vpos = np.zeros(n); self._vf_vneg = np.zeros(n)
                            spds = ", ".join(f"{s:g}" for s in self._vf_speeds)
                            self._say(f"VELOCITY-FRICTION sweep (current-based, torq-g): constant-velocity triangles "
                                      f"at [{spds}] rad/s (amp = v*{self._vf_period:g}/4, capped {self._vf_amp_max:g} rad), "
                                      f"{self._vf_nper} periods each -> fits B + de-biased tau_c. "
                                      f"HANDS OFF (~{self._vf_est_s():.0f}s total)")
                        elif not self._vf_speeds:
                            self._say("no vfric speeds configured (--vfric-speeds)")
                        else:
                            self._say("fv (velocity-friction) only works from DRAG (type d first)")
                    elif c in ("s", "static"):
                        if self.phase is Phase.DRAG:
                            self._static_init(q, t)
                            self._say("STATIC-FRICTION sweep: one joint at a time ramps torque until breakaway, "
                                      "both directions, others held stiff — hands OFF (~10 s per joint)")
                        else:
                            self._say("static sweep only works from DRAG (type d first)")
                    elif c in ("r", "release"):
                        if self.phase in (Phase.HOLD, Phase.DRAG):
                            self.q_hold = q.copy()
                            self.phase, t_phase = Phase.RELEASE, t
                            self._say(f"RELEASE: fading torques over {self.loop.release_s}s — arm should be resting!")
                    elif c == "q!":
                        self._say("EMERGENCY: disabling all motors NOW (arm may fall)")
                        arm.disable_all()
                        self.phase = Phase.DONE
                    elif self.assist is not None and hasattr(self.assist, "toggle_mode") \
                            and c in ("b", "bf", "bs"):
                        self._say(">>> " + self.assist.toggle_mode(c))
                    elif self.assist is not None and hasattr(self.assist, "set_kappa") \
                            and (c.startswith("bal ") or c.startswith("fric ") or c.startswith("sus ")
                                 or c.startswith("lam ") or c.startswith("detent ")
                                 or c.startswith("damp_t ") or c.startswith("damp_r ")
                                 or c.startswith("break ") or c.startswith("alphav ") or c.startswith("alphas ")
                                 or c.startswith("fmodel ") or c.startswith("mu ") or c.startswith("visc ")):
                        # live parameter changes (typed, or pushed by the --serve web panel);
                        # every change prints here so the CLI shows exactly what moved
                        try:
                            if c.startswith("bal "):
                                nk = self.assist.set_kappa(float(c[4:]))
                                self._say(f">>> balance kappa -> {nk:.2f}"
                                          + ("" if nk > 0 else "  (inertia shaping OFF)"))
                            elif c.startswith("fric "):
                                nf = self.assist.set_fric_scale(float(c[5:]))
                                self._say(f">>> friction compensation -> {nf:.0%} of calibrated"
                                          + ("" if nf > 0 else "  (OFF)"))
                            elif c.startswith("detent "):            # detent <kp>
                                nk = self.assist.set_detent(float(c.split()[1]))
                                self._say(f">>> latched detent -> {nk:g} N.m/rad" + ("" if nk > 0 else " (OFF)"))
                            elif c.startswith("damp_t "):            # damp_t <N.s/m>
                                dt, dr = self.assist.set_damp(d_t=float(c.split()[1]))
                                self._say(f">>> Cartesian damping trans -> {dt:g} N.s/m" + ("" if dt > 0 else " (OFF)"))
                            elif c.startswith("damp_r "):            # damp_r <N.m.s/rad>
                                dt, dr = self.assist.set_damp(d_r=float(c.split()[1]))
                                self._say(f">>> Cartesian damping rot -> {dr:g} N.m.s/rad" + ("" if dr > 0 else " (OFF)"))
                            elif c.startswith("break "):             # break <0..1>
                                nb = self.assist.set_break(float(c.split()[1]))
                                self._say(f">>> breakaway assist -> {nb:.0%}" + ("" if nb > 0 else " (OFF)"))
                            elif c.startswith("alphav "):            # alphav <sigma rad/s>
                                sv, k0 = self.assist.set_alpha(sigma_v=float(c.split()[1]))
                                self._say(f">>> alpha velocity knee -> {sv:g} rad/s" + ("" if sv > 0 else " (OFF, alpha=1)"))
                            elif c.startswith("alphas "):            # alphas <K0>
                                sv, k0 = self.assist.set_alpha(kappa0=float(c.split()[1]))
                                self._say(f">>> alpha singularity K0 -> {k0:g}" + ("" if k0 > 0 else " (OFF)"))
                            elif c.startswith("mu "):                # mu on|off - load term mu*|g(q)|
                                on = c.split()[1] in ("on", "1", "true")
                                m_on, _ = self.assist.set_fric_terms(mu=on)
                                self._say(">>> load term mu*|g(q)| -> " + ("ON (calibrated)" if m_on else "OFF (ablated)"))
                            elif c.startswith("visc "):              # visc on|off - viscous term B*qd
                                on = c.split()[1] in ("on", "1", "true")
                                _, v_on = self.assist.set_fric_terms(viscous=on)
                                self._say(">>> viscous term B*qd -> " + ("ON (calibrated)" if v_on else "OFF (ablated)"))
                            elif c.startswith("fmodel "):            # fmodel <viscous|load>
                                want = c.split()[1]
                                got = self.assist.set_fric_model(want)
                                if got == want:
                                    desc = {"viscous": "fv combined: tau_c + mu*|g(q)| + B*qd",
                                            "flat": "fv flat refit: pooled tau_c + B*qd (no load term)",
                                            "load": "friction_v2: tau_c + mu*|g(q)|"}.get(got, "")
                                    self._say(f">>> kinetic friction model -> {got} ({desc})")
                                else:
                                    known = ", ".join(self.assist.fric_models) or "none loaded"
                                    self._say(f"unknown friction model '{want}' (have: {known})")
                            elif c.startswith("lam "):                # lam <0..5> <value>
                                parts = c.split()
                                li = int(parts[1]); nv = self.assist.set_lam(li, float(parts[2]))
                                nm = ["trans-x", "trans-y", "trans-z", "rot-x", "rot-y", "rot-z"][li]
                                unit = "kg" if li < 3 else "kg.m^2"
                                self._say(f">>> Lambda_d {nm} -> {nv:g} {unit} (slews in ~0.5 s)")
                            else:                                   # sus <joint#> <on|off>
                                parts = c.split()
                                ji = int(parts[1]) - 1
                                on = parts[2] in ("on", "1", "true")
                                st = self.assist.set_sustain_joint(ji, on)
                                self._say(f">>> sustained relief joint{ji+1} -> {'ON' if st else 'OFF'}")
                        except (ValueError, IndexError):
                            self._say("usage: bal <0..2> | fric <0..0.85> | sus <1..6> <on|off> | lam <0..5> <val>")
                    elif self.assist is not None and hasattr(self.assist, "set_target") \
                            and (c in ("+", "-") or c[:1] in ("m", "i")):
                        try:
                            if c == "+":
                                md, ir = self.assist.set_target(scale=1.25)   # heavier
                            elif c == "-":
                                md, ir = self.assist.set_target(scale=0.8)    # lighter
                            else:
                                val = float(c[1:].lstrip(" ="))
                                md, ir = (self.assist.set_target(m_d=val) if c[0] == "m"
                                          else self.assist.set_target(i_rot=val))
                            self._say(f"virtual inertia target -> {md:.2f} kg / {ir:.3f} kg.m^2 "
                                      f"(slews in over ~0.5 s)")
                        except ValueError:
                            self._say("usage: m <kg> | i <kg.m^2> | + (heavier) | - (lighter), e.g. 'm 2.0'")
                    elif c == "":
                        self._status(t - t0, q, v, tau_cmd, torq, t_rot, cycles / max(t - t0, 1e-6))
                if self.phase is Phase.DONE:
                    break

                # ---- phase logic
                pos = q
                velcmd = np.zeros(n)
                if self.phase in (Phase.RAMP, Phase.FADE):
                    moved = np.abs(q - q_start)
                    if np.any(moved[self.active] > self.loop.ramp_window):
                        j = int(np.argmax(np.where(self.active, moved, 0)))
                        self._freeze(q, f"{names[j]} drifted {moved[j]:.3f} rad from the start pose during "
                                        f"{self.phase.value} (window {self.loop.ramp_window})")
                if self.phase is Phase.RAMP:
                    # stiff hold carries the arm; the feed-forward comes in gradually so a wrong sign
                    # shows up as a growing position error, not a runaway
                    alpha = min(1.0, (t - t_phase) / max(self.loop.ramp_s, 1e-3))
                    pos = q_start
                    kp = self.hold_kp
                    kd = self.hold_kd
                    tau_cmd = alpha * g
                    if alpha >= 1.0:
                        self.phase, t_phase = Phase.FADE, t
                        self._say(f"RAMP done -> FADE: hold gains fade out over {self.loop.fade_s}s")
                if self.phase is Phase.FADE:
                    beta = min(1.0, (t - t_phase) / max(self.loop.fade_s, 1e-3))
                    pos = q_start
                    kp = (1.0 - beta) * self.hold_kp + beta * self.kp_drag
                    kd = (1.0 - beta) * self.hold_kd + beta * self.kd_drag
                    tau_cmd = g.copy()
                    if beta >= 1.0:
                        self.phase, t_phase = Phase.DRAG, t
                        vel.reset()
                        self._say("FADE done -> DRAG: you can move the arm by hand now")
                if self.phase is Phase.DRAG:
                    fast = np.abs(v) > self.loop.vel_abort
                    if np.any(fast & self.active):
                        j = int(np.argmax(np.where(self.active, np.abs(v), 0)))
                        self._freeze(q, f"{names[j]} velocity {v[j]:+.2f} rad/s > {self.loop.vel_abort}")
                    elif self.duration is not None and (t - t_phase) >= self.duration:
                        self.q_drag_end = q.copy()
                        self.v_drag_end = v.copy()
                        if self.auto_release:
                            self.q_hold = q.copy()
                            self.phase, t_phase = Phase.RELEASE, t
                            self._say("duration reached -> RELEASE")
                        else:
                            self._freeze(q, "duration reached")
                    else:
                        kp = self.kp_drag
                        kd = self.kd_drag
                        # gravity + Coulomb-friction feed-forward in the direction of motion
                        tau_cmd = g + self.fric_comp * np.tanh(v / self.fric_v0)
                        if self.assist is not None:
                            tau_cmd = tau_cmd + self.assist.update(q, v, dt)
                if self.phase is Phase.CAPTURE and getattr(self, "_cap_mode", "calib") == "vfric":
                    # multi-speed CURRENT-BASED sweep: read the motor's torque feedback (torq = K_t*iq,
                    # eq 22) at several constant speeds, subtract gravity -> friction(+/-v). The fit
                    # (scripts/fit_friction.py) regresses tau_c*tanh(v/eps) + B*v + c0, giving B and a
                    # de-biased tau_c (c0 absorbs any residual gravity/current offset).
                    spd = self._vf_speeds[self._vf_si]
                    period = self._vf_period                     # fixed period -> a real constant-velocity dwell
                    amp = spd * period / 4.0                     # so |qd| = spd during each half-period
                    if amp > self._vf_amp_max:                   # cap the excursion; shorten the period instead
                        amp = self._vf_amp_max
                        period = 4.0 * amp / spd
                    total = 1.0 + self._vf_nper * period
                    el = t - t_phase
                    if el < 1.0:                                 # settle at this speed
                        tri, dtri = 0.0, 0.0
                    else:
                        ph_ = ((el - 1.0) % period) / period
                        tri = 4.0 * ph_ - 1.0 if ph_ < 0.5 else 3.0 - 4.0 * ph_
                        dtri = (4.0 / period) if ph_ < 0.5 else (-4.0 / period)
                    pos = self.q_hold + amp * tri
                    velcmd = np.full(n, amp * dtri)
                    kp = self.hold_kp
                    kd = self.hold_kd
                    tau_cmd = g.copy()
                    if el >= 1.0 and np.isfinite(src.torq).any():
                        f_meas = src.torq - g                    # current-based friction residual (eq 22)
                        frac = ((el - 1.0) % period) / period
                        # Per-joint STEADY-STATE gate: only average a sample once that joint has caught up
                        # to the commanded speed (|v - velcmd| small) -> excludes the inertial transient
                        # after each reversal, which otherwise biases B on the heavy joints. Record the
                        # MEASURED v so the fit's x-axis is the actual speed, not the commanded one.
                        steady = np.abs(v - velcmd) < (0.12 * spd + 0.015)
                        if 0.10 < frac < 0.45:                   # +qd half (transients gated out per joint)
                            m = steady.astype(float)
                            self._vf_pos += m * f_meas; self._vf_vpos += m * v; self._vf_npos += m
                        elif 0.60 < frac < 0.95:                 # -qd half
                            m = steady.astype(float)
                            self._vf_neg += m * f_meas; self._vf_vneg += m * v; self._vf_nneg += m
                    if el >= total:                              # this speed done -> bank the +/-v rows
                        fp = self._vf_pos / np.maximum(self._vf_npos, 1)
                        fn = self._vf_neg / np.maximum(self._vf_nneg, 1)
                        vp = self._vf_vpos / np.maximum(self._vf_npos, 1)
                        vn = self._vf_vneg / np.maximum(self._vf_nneg, 1)
                        self._vf_rows.append((+spd, self.q_hold.copy(), vp.copy(), fp.copy()))
                        self._vf_rows.append((-spd, self.q_hold.copy(), vn.copy(), fn.copy()))
                        self._say(f"  vfric ~{spd:.3f} rad/s: |qd|meas={np.round(0.5*(vp-vn),3)}  "
                                  f"(torq-g)+={np.round(fp, 2)} -={np.round(fn, 2)}")
                        self._vf_si += 1
                        self._vf_pos = np.zeros(n); self._vf_npos = np.zeros(n)
                        self._vf_neg = np.zeros(n); self._vf_nneg = np.zeros(n)
                        self._vf_vpos = np.zeros(n); self._vf_vneg = np.zeros(n)
                        t_phase = t
                        if self._vf_si >= len(self._vf_speeds):
                            self._append_vfric(self._vf_rows)
                            self.phase, t_phase = Phase.DRAG, t
                            vel.reset()
                            self._say("velocity-friction sweep done -> back to DRAG. Repeat at 3-6 spread "
                                      f"poses, then: python scripts/fit_friction.py {self.vfric_path or 'friction_v.csv'}")
                elif self.phase is Phase.CAPTURE:
                    el = t - t_phase
                    friction_mode = getattr(self, "_cap_mode", "calib") == "friction"
                    period = 4.0 if friction_mode else 2.0
                    amp = 0.15 if friction_mode else self.capture_amp
                    total = 1.0 + 2 * period if friction_mode else self.capture_s
                    if el < 1.0:                               # settle
                        tri, dtri = 0.0, 0.0
                    else:                                       # triangle wave in [-1, 1], slope 4/period
                        ph_ = ((el - 1.0) % period) / period
                        tri = 4.0 * ph_ - 1.0 if ph_ < 0.5 else 3.0 - 4.0 * ph_
                        dtri = (4.0 / period) if ph_ < 0.5 else (-4.0 / period)
                    pos = self.q_hold + amp * tri
                    velcmd = np.full(n, amp * dtri)
                    kp = self.hold_kp
                    kd = self.hold_kd
                    tau_cmd = g.copy()
                    if el >= 1.0:                               # average over whole periods only
                        r = kp * (pos - q) + kd * (velcmd - v)
                        self._cap_sum += r
                        self._cap_n += 1
                        # skip the 15 % of each half-period around the reversals (transients), split by direction
                        frac = ((el - 1.0) % period) / period
                        if 0.075 < frac < 0.425:
                            self._cap_pos += r; self._cap_npos += 1
                        elif 0.575 < frac < 0.925:
                            self._cap_neg += r; self._cap_nneg += 1
                    if friction_mode and el >= total:
                        rp = self._cap_pos / max(self._cap_npos, 1)
                        rn = self._cap_neg / max(self._cap_nneg, 1)
                        coulomb = 0.5 * (rp - rn)
                        ref = np.array([0.53, 0.53, 0.49, 0.30, 0.21, 0.21])   # Seeed 2026-07-17 (j1/j6 assumed like neighbours)
                        self._say("friction sweep at q(deg)=%s\n   Coulomb friction (N.m): %s\n   Seeed reference     : %s\n   gravity residual    : %s"
                                  "\n   paste into config/b601_rs.toml: fric_kinetic = <value> under each [[joint]]"
                                  % (np.round(np.degrees(self.q_hold), 0), np.round(coulomb, 2), ref, np.round(0.5 * (rp + rn), 2)))
                        self.friction_result = coulomb
                        self._append_fric("kinetic", self.q_hold, coulomb, 0.5 * (rp + rn))
                        self.phase, t_phase = Phase.DRAG, t
                        vel.reset()
                        self._say("back to DRAG")
                    elif (not friction_mode) and el >= 1.0 + period * max(1, int((self.capture_s - 1.0) // period)):
                        resid = self._cap_sum / max(self._cap_n, 1)
                        g_raw = dyn.gravity_raw(q)
                        rec = {"q": q.copy(), "g_raw": g_raw, "ff": g.copy(), "resid": resid}
                        self.captures.append(rec)
                        self._say("capture #%d at q(deg)=%s\n   model ff  = %s\n   PD resid  = %s   (+ = model too small; |resid| < friction is noise)"
                                  % (len(self.captures), np.round(np.degrees(q), 0), np.round(g, 2), np.round(resid, 2)))
                        if self.calib_path:
                            new = not os.path.exists(self.calib_path)
                            with open(self.calib_path, "a", newline="") as cf:
                                w = csv.writer(cf)
                                if new:
                                    w.writerow([f"q{i+1}" for i in range(n)] + [f"graw{i+1}" for i in range(n)]
                                               + [f"ff{i+1}" for i in range(n)] + [f"resid{i+1}" for i in range(n)])
                                w.writerow([f"{x:.5f}" for x in q] + [f"{x:.4f}" for x in g_raw]
                                           + [f"{x:.4f}" for x in g] + [f"{x:.4f}" for x in resid])
                        self.phase, t_phase = Phase.DRAG, t
                        vel.reset()
                        self._say("back to DRAG")
                if self.phase is Phase.STATIC:
                    pos = self._st_hold
                    kp = self.hold_kp.copy()
                    kd = self.hold_kd.copy()
                    tau_cmd = g.copy()
                    j = self._st_joints[self._st_ji]
                    el = t - self._st_t
                    if self._st_stage == "settle":
                        if el >= 0.8:
                            self._st_stage, self._st_t = "ramp", t
                            self._st_tau, self._st_anchor = 0.0, q[j]
                            self._st_hold[j] = q[j]
                    elif self._st_stage == "ramp":
                        kp[j] = 0.0                       # free joint: gravity ff + ramped extra torque only
                        kd[j] = self.kd_drag[j]           # light damping bounds the post-breakaway motion
                        self._st_tau += self._st_dir * self._st_rate[j] * dt
                        tau_cmd[j] += self._st_tau
                        # breakaway = |qd| above threshold for a SUSTAINED window (paper 4.3, t+): a
                        # plain position threshold fires on backlash take-up (the joint clicks through
                        # the gear slack) and under-reads. The recorded torque is the one at the ONSET
                        # of the sustained window, not when the window completes.
                        if abs(v[j]) > 0.03:
                            if self._st_vcnt == 0:
                                self._st_tau_onset = abs(self._st_tau)
                            self._st_vcnt += 1
                        else:
                            self._st_vcnt = 0
                        if self._st_vcnt >= 6:            # ~60 ms sustained at 100 Hz
                            self._st_res.setdefault(j, {})[self._st_dir] = self._st_tau_onset
                            self._st_hold[j] = q[j]
                            self._st_vcnt = 0
                            self._st_stage, self._st_t = "rehold", t
                        elif abs(self._st_tau) > self._st_cap[j]:
                            self._st_res.setdefault(j, {})[self._st_dir] = float("nan")
                            self._say(f"  {names[j]}: no breakaway at {self._st_cap[j]:.2f} N.m "
                                      f"(dir {self._st_dir:+d}) — near a limit or in contact?")
                            self._st_hold[j] = q[j]
                            self._st_vcnt = 0
                            self._st_stage, self._st_t = "rehold", t
                    elif self._st_stage == "rehold":
                        if el >= 0.8:
                            if self._st_dir > 0:
                                self._st_dir = -1
                                self._st_stage, self._st_t = "ramp", t
                                self._st_tau, self._st_anchor = 0.0, q[j]
                                self._st_hold[j] = q[j]
                            else:
                                self._st_dir = 1
                                self._st_ji += 1
                                if self._st_ji >= len(self._st_joints):
                                    self._static_finish(names)
                                    self.phase, t_phase = Phase.DRAG, t
                                    vel.reset()
                                    self._say("back to DRAG")
                                else:
                                    self._st_stage, self._st_t = "settle", t
                                    self._say(f"  static sweep: {names[self._st_joints[self._st_ji]]}")
                if self.phase is Phase.HOLD:
                    pos = self.q_hold
                    kp = self.hold_kp
                    kd = self.hold_kd
                    tau_cmd = g.copy()
                    if self.hold_timeout is not None and self.t_hold is not None and (t - self.t_hold) >= self.hold_timeout:
                        self._say(f"HOLD timeout ({self.hold_timeout}s) -> RELEASE")
                        self.phase, t_phase = Phase.RELEASE, t
                if self.phase is Phase.RELEASE:
                    beta = max(0.0, 1.0 - (t - t_phase) / max(self.loop.release_s, 1e-3))
                    pos = self.q_hold
                    kp = beta * self.hold_kp
                    kd = self.release_kd
                    tau_cmd = beta * g
                    if beta <= 0.0:
                        arm.disable_all()
                        self.phase = Phase.DONE
                        self._say("released: all motors disabled")
                        break

                if self.assist is not None and self.phase is not Phase.DRAG and hasattr(self.assist, "observe"):
                    self.assist.observe(q, v, dt)   # keep the estimator running while held

                # ---- send
                tau_cmd = np.clip(tau_cmd, -self.tau_max, self.tau_max)
                try:
                    arm.send_mit(pos, velcmd, kp, kd, tau_cmd, self.active)
                    if self.gripper_hold:
                        gc = cfg.gripper
                        arm.send_gripper_mit(gq0, gc.hold_kp, gc.hold_kd, 0.0)
                except Exception as e:
                    fails += 1
                    if fails >= self.loop.max_read_failures:
                        self._freeze(q, f"send failures ({e})")
                if self.assist is not None and hasattr(self.assist, "note_sent"):
                    self.assist.note_sent(tau_cmd, kp, kd, pos, velcmd)

                # ---- telemetry from the feedback frames
                torq = src.torq
                t_rot = np.nanmax(src.temp) if np.isfinite(src.temp).any() else np.nan
                self.telem = {                    # snapshot for the --serve web panel
                    "v": v, "tau": tau_cmd, "temp": t_rot,
                    "r": getattr(self.assist, "r", None),
                }
                if np.isfinite(t_rot) and t_rot > self.loop.temp_abort_c:
                    self._freeze(q, f"motor temperature {t_rot:.0f} C > {self.loop.temp_abort_c}")
                if writer is not None:
                    writer.writerow([f"{t - t0:.4f}", self.phase.value] + [f"{x:.5f}" for x in q]
                                    + [f"{x:.4f}" for x in v] + [f"{x:.4f}" for x in tau_cmd]
                                    + [f"{x:.3f}" for x in torq] + [f"{x:.4f}" for x in src.vel_fb] + [f"{t_rot:.1f}"]
                                    + ([f"{x:.4f}" for x in self.assist.r] if hasattr(self.assist, "r") else []))
                if self.print_every and (t - t_print) >= self.print_every:
                    t_print = t
                    self._status(t - t0, q, v, tau_cmd, torq, t_rot, cycles / max(t - t0, 1e-6))
                if self.phase is Phase.HOLD and self.interactive and (t - t_remind) >= 5.0:
                    t_remind = t
                    print(f">>> HOLD ({self.freeze_reason}) — stiff on purpose. d+Enter = drag, r+Enter = release", flush=True)

                cycles += 1
                if self.realtime:
                    remaining = self.dt_nom - (self._now() - t)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            if self.phase is not Phase.DONE:
                # abnormal exit (exception): freeze is impossible without the loop -> disable
                self._say("controller exiting abnormally: disabling all motors")
                try:
                    arm.disable_all()
                except Exception:
                    pass
                self.phase = Phase.DONE
            if logf is not None:
                logf.close()
        return self.phase

    def _status(self, t, q, v, tau, torq, t_rot, hz) -> None:
        qd = np.degrees(q)
        line = (f"[{t:7.2f}s {self.phase.value:7s} {hz:5.0f}Hz] "
                f"q(deg)=" + " ".join(f"{x:+6.1f}" for x in qd)
                + " | tau=" + " ".join(f"{x:+5.2f}" for x in tau)
                + " | v=" + " ".join(f"{x:+4.2f}" for x in v))
        if np.isfinite(torq).any():
            line += " | torq_fb=" + " ".join(f"{x:+5.2f}" for x in torq)
        if np.isfinite(t_rot):
            line += f" | Tmax={t_rot:.0f}C"
        src = getattr(self, "src", None)
        if src is not None:
            line += f" | {src.summary()}"
        if self.assist is not None:
            line += " | " + self.assist.summary()
        print(line, flush=True)
