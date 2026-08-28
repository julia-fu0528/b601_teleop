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
          PD residual is split by direction of motion: Coulomb friction = (resid+ - resid-)/2 per joint.
Keys while running (type + Enter): d = drag, h = hold, c = capture, f = friction sweep, r = release, q! = disable NOW.
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
        capture_s: float = 5.0,
        capture_amp: float = 0.05,
        print_every: float = 0.5,
        interactive: bool = True,
        realtime: bool = True,
        hold_timeout: float | None = None,
        lateral=None,
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
        self.capture_s = float(capture_s)
        self.capture_amp = float(capture_amp)
        self.captures: list[dict] = []
        self.print_every = print_every
        self.interactive = interactive
        self.realtime = realtime
        self.startup_failed = False
        self.hold_timeout = hold_timeout   # HOLD -> RELEASE automatically after this many s (None = wait for operator)
        self.t_hold: float | None = None
        self.lateral = lateral             # LateralAssist or None (b601/assist.py, step 2)

        self.loop = cfg.loop
        self.dt_nom = 1.0 / self.loop.rate_hz
        self.phase = Phase.RAMP
        self.freeze_reason: str | None = None
        self.events: list[tuple[float, str]] = []
        self.q_drag_end: np.ndarray | None = None
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
                            + [f"velfb{i+1}" for i in range(n)] + ["t_rotor_max"])

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
                g = dyn.gravity(q) * self.scale

                # ---- commands
                while True:
                    try:
                        c = self._cmds.get_nowait()
                    except queue.Empty:
                        break
                    if c == "<sigint>":
                        if self.phase in (Phase.RAMP, Phase.FADE, Phase.DRAG, Phase.CAPTURE):
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
                    elif c in ("r", "release"):
                        if self.phase in (Phase.HOLD, Phase.DRAG):
                            self.q_hold = q.copy()
                            self.phase, t_phase = Phase.RELEASE, t
                            self._say(f"RELEASE: fading torques over {self.loop.release_s}s — arm should be resting!")
                    elif c == "q!":
                        self._say("EMERGENCY: disabling all motors NOW (arm may fall)")
                        arm.disable_all()
                        self.phase = Phase.DONE
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
                        if self.lateral is not None:
                            tau_cmd = tau_cmd + self.lateral.update(q, v, dt)
                if self.phase is Phase.CAPTURE:
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
                                  % (np.round(np.degrees(self.q_hold), 0), np.round(coulomb, 2), ref, np.round(0.5 * (rp + rn), 2)))
                        self.friction_result = coulomb
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

                # ---- telemetry from the feedback frames
                torq = src.torq
                t_rot = np.nanmax(src.temp) if np.isfinite(src.temp).any() else np.nan
                if np.isfinite(t_rot) and t_rot > self.loop.temp_abort_c:
                    self._freeze(q, f"motor temperature {t_rot:.0f} C > {self.loop.temp_abort_c}")
                if writer is not None:
                    writer.writerow([f"{t - t0:.4f}", self.phase.value] + [f"{x:.5f}" for x in q]
                                    + [f"{x:.4f}" for x in v] + [f"{x:.4f}" for x in tau_cmd]
                                    + [f"{x:.3f}" for x in torq] + [f"{x:.4f}" for x in src.vel_fb] + [f"{t_rot:.1f}"])
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
        if self.lateral is not None:
            line += f" | lat j1 {self.lateral.tau1:+.2f}Nm"
        print(line, flush=True)
