"""Leader-follower teleoperation: reBot Arm 102 leader -> B601-RS follower (MIT tracking + gravity ff).

Phases
  ENGAGE  the follower target starts at the follower's *current* pose and glides toward the leader pose at
          engage_vel (rad/s). No jumps, whatever the leader is doing. -> TRACK when within 1 deg.
  TRACK   target = leader pose, rate-limited to max_vel. Leader read glitches (> jump_deg per sample) are
          ignored; 3 in a row, a stale leader (> stale_s), a fast joint, a hot motor or CAN failures -> HOLD.
  HOLD    PD hold at the frozen pose + gravity ff (never torque-off).   t = re-engage, r = release.
  RELEASE torque fade with strong damping, then all motors disabled.
  DONE

Keys (type + Enter): h = hold, t = track (re-engage from HOLD), r = release, q! = disable NOW, Enter = status.
Ctrl+C: ENGAGE/TRACK -> HOLD, HOLD -> RELEASE, RELEASE -> disable now.
"""
from __future__ import annotations

import csv
import enum
import queue
import signal
import sys
import threading
import time

import numpy as np

from .arm import KD_MAX, PositionSource, VelocityEstimator
from .config import Config
from .dynamics import ArmDynamics
from .leader import map_to_follower


class TPhase(enum.Enum):
    ENGAGE = "engage"
    TRACK = "track"
    HOLD = "hold"
    RELEASE = "release"
    DONE = "done"


class TeleopController:
    def __init__(
        self,
        arm,
        dyn: ArmDynamics,
        cfg: Config,
        leader,
        *,
        gripper: bool = True,
        kp_scale: float = 1.0,
        duration: float | None = None,
        auto_release: bool = False,
        hold_timeout: float | None = None,
        log_path: str | None = None,
        print_every: float = 0.5,
        interactive: bool = True,
        realtime: bool = True,
        force_engage: bool = False,
        vel_ff: bool | None = None,
        lead_s: float | None = None,
    ) -> None:
        if cfg.leader is None:
            raise ValueError("config has no [leader] section")
        self.arm, self.dyn, self.cfg, self.leader = arm, dyn, cfg, leader
        self.n = dyn.nq
        self.lc, self.tc = cfg.leader, cfg.teleop
        self.kp = np.asarray(self.tc.kp, float) * float(kp_scale)
        self.kd = np.asarray(self.tc.kd, float)
        self.tau_max = np.array([j.tau_max for j in cfg.joints], float)
        self.hold_kp = np.array([j.hold_kp for j in cfg.joints], float)
        self.hold_kd = np.array([j.hold_kd for j in cfg.joints], float)
        self.release_kd = np.minimum(KD_MAX, np.maximum(3.0 * self.hold_kd, self.hold_kd))
        self.lim_lo = np.radians([l[0] for l in self.tc.limits_deg])
        self.lim_hi = np.radians([l[1] for l in self.tc.limits_deg])
        self.active = np.ones(self.n, bool)
        self.use_gripper = bool(gripper) and cfg.gripper is not None and getattr(arm, "gripper", None) is not None
        self.duration, self.auto_release, self.hold_timeout = duration, auto_release, hold_timeout
        self.log_path, self.print_every = log_path, print_every
        self.interactive, self.realtime, self.force_engage = interactive, realtime, force_engage
        self.vel_ff = self.tc.vel_ff if vel_ff is None else bool(vel_ff)
        self.lead_s = self.tc.lead_s if lead_s is None else float(lead_s)
        self.loop = cfg.loop
        self.dt_nom = 1.0 / self.loop.rate_hz
        self.phase = TPhase.ENGAGE
        self.freeze_reason: str | None = None
        self.track_err_log: list[np.ndarray] = []
        self.max_engage_v = 0.0          # peak joint speed seen while engaging (diagnostic)
        self._cmds: "queue.Queue[str]" = queue.Queue()
        self._sim_t = 0.0
        self.t_hold: float | None = None

    # ---- helpers -------------------------------------------------------------------------
    def _now(self) -> float:
        return time.perf_counter() if self.realtime else self._sim_t

    def _say(self, msg: str) -> None:
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
        if self.phase in (TPhase.HOLD, TPhase.RELEASE, TPhase.DONE):
            return
        self.freeze_reason = why
        self.q_hold = q.copy()
        self.t_hold = self._now()
        self.phase = TPhase.HOLD
        self._say(f"\n*** HOLD: {why}\n*** Follower is STIFF ON PURPOSE (PD hold + gravity ff). "
                  f"t + Enter = re-engage the leader, r + Enter = release, q! + Enter = disable NOW.")

    def _leader_targets(self, deg7: np.ndarray) -> tuple[np.ndarray, float]:
        """Leader servo deg (7) -> follower joint targets (rad, 6) clipped to limits, gripper target (rad)."""
        mapped = map_to_follower(deg7, self.lc)
        q_t = np.clip(np.radians(mapped[: self.n]), self.lim_lo, self.lim_hi)
        g_lo, g_hi = self.tc.gripper_limits_deg
        g_t = np.radians(np.clip(mapped[self.n], g_lo, g_hi)) if len(mapped) > self.n else 0.0
        return q_t, float(g_t)

    # ---- main ----------------------------------------------------------------------------
    def run(self) -> TPhase:
        arm, dyn, cfg, n = self.arm, self.dyn, self.cfg, self.n
        names = cfg.joint_names

        q0 = arm.read_q()
        viol = dyn.limit_violation(q0)
        if np.any(viol > 0.05):
            raise RuntimeError(f"follower start pose outside URDF limits by {np.round(viol, 3)} rad")

        # wait for a leader sample
        t_wait = time.perf_counter()
        s = None
        while s is None and time.perf_counter() - t_wait < 3.0:
            s = self.leader.sample(self._now())
            if s is None:
                time.sleep(0.02)
        if s is None:
            raise RuntimeError("no data from the leader arm")
        q_lead, g_lead = self._leader_targets(s[1])
        gap = np.degrees(q_lead - q0)
        print("follower pose (deg):", np.round(np.degrees(q0), 1))
        print("leader target (deg):", np.round(np.degrees(q_lead), 1), " gripper:", round(np.degrees(g_lead), 1))
        print("engage distance (deg):", np.round(gap, 1))
        if np.abs(gap).max() > self.tc.max_engage_deg and not self.force_engage:
            raise RuntimeError(f"a joint is {np.abs(gap).max():.0f} deg away from the leader (> {self.tc.max_engage_deg}); "
                               f"move the leader closer to the follower's pose, or check the leader zero (scripts/teleop.py compare)")

        if self.interactive:
            signal.signal(signal.SIGINT, self._on_sigint)
            threading.Thread(target=self._stdin_thread, name="keys", daemon=True).start()

        writer = None
        logf = None
        if self.log_path:
            logf = open(self.log_path, "w", newline="")
            writer = csv.writer(logf)
            writer.writerow(["t", "phase"] + [f"lead{i+1}" for i in range(n)] + ["lead_grip"]
                            + [f"tgt{i+1}" for i in range(n)] + [f"q{i+1}" for i in range(n)]
                            + [f"tau{i+1}" for i in range(n)] + ["grip_tgt", "grip_q", "grip_tau"])

        arm.prepare_mit(self.active, self.use_gripper)
        arm.enable(self.active, self.use_gripper)
        # first command immediately: hold the current pose + gravity ff (no torque-less gap after enable)
        arm.send_mit(q0, np.zeros(n), self.kp, self.kd, np.clip(dyn.gravity(q0), -self.tau_max, self.tau_max), self.active)
        self._say(f"motors enabled; ENGAGE at {self.tc.engage_vel} rad/s toward the leader pose")

        src = PositionSource(arm, self.active)
        self.src = src
        vel = VelocityEstimator(n)
        q = q0.copy()
        target = q0.copy()                  # commanded target (rad); starts at the follower pose -> no jump
        target_prev = q0.copy()
        v_des = np.zeros(n)                 # filtered target velocity (MIT velocity feed-forward)
        g_target = None
        g_prev_target = None
        g_prev_vel_f = 0.0
        t0 = self._now()
        t_phase = t0
        t_prev = t0
        t_print = t0
        t_remind = t0
        t_track_start = None
        fails = 0
        glitches = 0
        last_lead = s[1].copy()
        last_lead_t = t0
        tau_cmd = np.zeros(n)
        g_tau = 0.0
        g_q = g_lead
        cycles = 0
        try:
            while self.phase is not TPhase.DONE:
                if not self.realtime:
                    self._sim_t += self.dt_nom
                t = self._now()
                dt = max(t - t_prev, 1e-4) if self.realtime else self.dt_nom
                t_prev = t

                # ---- follower state
                try:
                    q = src.update()
                    fails = 0
                    while src.warnings:
                        self._say("WARNING: " + src.warnings.pop(0))
                except Exception as e:
                    fails += 1
                    if fails >= self.loop.max_read_failures:
                        self._freeze(q, f"{fails} consecutive CAN read failures ({e})")
                v = vel.update(q, dt)
                g = dyn.gravity(q)

                # ---- leader sample (glitch + stale guards)
                smp = self.leader.sample(t)
                if smp is not None:
                    t_s, deg = smp
                    if t_s != last_lead_t:
                        jump = np.abs(deg[:n] - last_lead[:n]).max()
                        if jump > self.tc.jump_deg and self.phase is TPhase.TRACK:
                            glitches += 1
                            if glitches >= 3:
                                self._freeze(q, f"leader jumped {jump:.0f} deg in one sample, 3 times in a row")
                        else:
                            glitches = 0
                            last_lead = deg.copy()
                            last_lead_t = t_s
                stale = (t - last_lead_t) if self.realtime else 0.0
                if stale > self.lc.stale_s and self.phase in (TPhase.ENGAGE, TPhase.TRACK):
                    self._freeze(q, f"leader data stale for {stale:.2f}s")
                q_lead, g_lead = self._leader_targets(last_lead)

                # ---- commands
                while True:
                    try:
                        c = self._cmds.get_nowait()
                    except queue.Empty:
                        break
                    if c == "<sigint>":
                        if self.phase in (TPhase.ENGAGE, TPhase.TRACK):
                            self._freeze(q, "Ctrl+C")
                        elif self.phase is TPhase.HOLD:
                            self._say("Ctrl+C in HOLD -> RELEASE (torque fade, then disable)")
                            self.phase, t_phase = TPhase.RELEASE, t
                        else:
                            self._say("Ctrl+C in RELEASE -> disabling NOW")
                            arm.disable_all()
                            self.phase = TPhase.DONE
                    elif c in ("h", "hold"):
                        self._freeze(q, "user requested hold")
                    elif c in ("t", "track", "d"):
                        if self.phase is TPhase.HOLD:
                            self.phase, t_phase = TPhase.ENGAGE, t
                            target = q.copy()
                            self.freeze_reason = None
                            glitches = 0
                            self._say("re-ENGAGE: gliding to the leader pose")
                    elif c in ("r", "release"):
                        if self.phase in (TPhase.HOLD, TPhase.TRACK, TPhase.ENGAGE):
                            self.q_hold = q.copy()
                            self.phase, t_phase = TPhase.RELEASE, t
                            self._say(f"RELEASE: fading torques over {self.loop.release_s}s — follower should be resting!")
                    elif c == "q!":
                        self._say("EMERGENCY: disabling all motors NOW (arm may fall)")
                        arm.disable_all()
                        self.phase = TPhase.DONE
                    elif c == "":
                        self._status(t - t0, q, q_lead, tau_cmd, g_q, g_tau, cycles / max(t - t0, 1e-6))
                if self.phase is TPhase.DONE:
                    break

                # ---- phase logic: compute pos/kp/kd/tau for the arm joints
                pos = target
                velcmd = np.zeros(n)
                if self.phase is TPhase.ENGAGE:
                    step = self.tc.engage_vel * dt
                    target = target + np.clip(q_lead - target, -step, step)
                    pos, kp, kd, tau_cmd = target, self.kp, self.kd, g.copy()
                    self.max_engage_v = max(self.max_engage_v, float(np.abs(v).max()))
                    if np.abs(q_lead - target).max() < np.radians(1.0):
                        self.phase, t_phase = TPhase.TRACK, t
                        t_track_start = t
                        self._say("ENGAGED -> TRACK: the follower now follows the leader")
                if self.phase is TPhase.TRACK:
                    fast = np.abs(v) > self.loop.vel_abort
                    if np.any(fast):
                        j = int(np.argmax(np.abs(v)))
                        self._freeze(q, f"{names[j]} velocity {v[j]:+.2f} rad/s > {self.loop.vel_abort}")
                    elif self.duration is not None and (t - t_track_start) >= self.duration:
                        if self.auto_release:
                            self.q_hold = q.copy()
                            self.phase, t_phase = TPhase.RELEASE, t
                            self._say("duration reached -> RELEASE")
                        else:
                            self._freeze(q, "duration reached")
                    else:
                        step = self.tc.max_vel * dt
                        target = target + np.clip(q_lead - target, -step, step)
                        pos, kp, kd, tau_cmd = target, self.kp, self.kd, g.copy()
                        self.track_err_log.append(q_lead - q)
                if self.phase in (TPhase.ENGAGE, TPhase.TRACK):
                    # velocity feed-forward + latency prediction on the rate-limited target
                    v_raw = (target - target_prev) / dt
                    a = dt / (self.tc.vel_tau + dt)
                    v_des = v_des + a * (v_raw - v_des)
                    v_des = np.clip(v_des, -self.tc.max_vel, self.tc.max_vel)
                    if self.vel_ff:
                        velcmd = v_des
                        pos = np.clip(target + v_des * self.lead_s, self.lim_lo, self.lim_hi)
                target_prev = target.copy()
                if self.phase is TPhase.HOLD:
                    v_des[:] = 0.0
                    pos, kp, kd, tau_cmd = self.q_hold, self.hold_kp, self.hold_kd, g.copy()
                    if self.hold_timeout is not None and self.t_hold is not None and (t - self.t_hold) >= self.hold_timeout:
                        self._say(f"HOLD timeout ({self.hold_timeout}s) -> RELEASE")
                        self.phase, t_phase = TPhase.RELEASE, t
                if self.phase is TPhase.RELEASE:
                    beta = max(0.0, 1.0 - (t - t_phase) / max(self.loop.release_s, 1e-3))
                    pos, kp, kd, tau_cmd = self.q_hold, beta * self.hold_kp, self.release_kd, beta * g
                    if beta <= 0.0:
                        arm.disable_all()
                        self.phase = TPhase.DONE
                        self._say("released: all motors disabled")
                        break

                # ---- send arm
                tau_cmd = np.clip(tau_cmd, -self.tau_max, self.tau_max)
                try:
                    arm.send_mit(pos, velcmd, kp, kd, tau_cmd, self.active)
                except Exception as e:
                    fails += 1
                    if fails >= self.loop.max_read_failures:
                        self._freeze(q, f"send failures ({e})")

                # ---- gripper: host-side impedance with force limit (Seeed's scheme)
                if self.use_gripper:
                    gs = arm.gripper_state()
                    if gs is not None:
                        g_q, g_v = gs
                        if self.phase in (TPhase.ENGAGE, TPhase.TRACK):
                            g_target = g_lead if g_target is None else g_target + np.clip(g_lead - g_target, -2.0 * dt, 2.0 * dt)
                        elif g_target is None:
                            g_target = g_q
                        tv = 0.0 if g_prev_target is None else (g_target - g_prev_target) / dt
                        g_prev_target = g_target
                        g_prev_vel_f = 0.3 * tv + 0.7 * g_prev_vel_f
                        tv = float(np.clip(g_prev_vel_f, -3.0, 3.0))
                        g_tau = self.tc.gripper_kp * (g_target - g_q) + self.tc.gripper_kd * (tv - g_v)
                        lim = self.tc.gripper_tau_max if abs(g_v) > 0.25 else self.tc.gripper_tau_hold
                        if self.phase is TPhase.RELEASE:
                            lim *= beta
                        g_tau = float(np.clip(g_tau, -lim, lim))
                        try:
                            arm.send_gripper_mit(0.0, 0.0, 1.5, g_tau)
                        except Exception:
                            pass

                # ---- temperature / telemetry
                t_rot = np.nanmax(src.temp) if np.isfinite(src.temp).any() else np.nan
                if np.isfinite(t_rot) and t_rot > self.loop.temp_abort_c:
                    self._freeze(q, f"motor temperature {t_rot:.0f} C > {self.loop.temp_abort_c}")
                if writer is not None:
                    writer.writerow([f"{t - t0:.4f}", self.phase.value] + [f"{x:.2f}" for x in last_lead[:n]] + [f"{last_lead[n]:.2f}"]
                                    + [f"{x:.5f}" for x in target] + [f"{x:.5f}" for x in q] + [f"{x:.4f}" for x in tau_cmd]
                                    + [f"{(g_target or 0.0):.4f}", f"{g_q:.4f}", f"{g_tau:.3f}"])
                if self.print_every and (t - t_print) >= self.print_every:
                    t_print = t
                    self._status(t - t0, q, q_lead, tau_cmd, g_q, g_tau, cycles / max(t - t0, 1e-6))
                if self.phase is TPhase.HOLD and self.interactive and (t - t_remind) >= 5.0:
                    t_remind = t
                    print(f">>> HOLD ({self.freeze_reason}) — stiff on purpose. t+Enter = re-engage, r+Enter = release", flush=True)

                cycles += 1
                if self.realtime:
                    remaining = self.dt_nom - (self._now() - t)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            if self.phase is not TPhase.DONE:
                self._say("controller exiting abnormally: disabling all motors")
                try:
                    arm.disable_all()
                except Exception:
                    pass
                self.phase = TPhase.DONE
            if logf is not None:
                logf.close()
        return self.phase

    def _status(self, t, q, q_lead, tau, g_q, g_tau, hz) -> None:
        err = np.degrees(q_lead - q)
        line = (f"[{t:7.2f}s {self.phase.value:7s} {hz:5.0f}Hz] q(deg)=" + " ".join(f"{x:+6.1f}" for x in np.degrees(q))
                + " | lead-q(deg)=" + " ".join(f"{x:+5.1f}" for x in err)
                + " | tau=" + " ".join(f"{x:+5.2f}" for x in tau)
                + f" | grip q={np.degrees(g_q):+6.1f} tau={g_tau:+.2f}")
        src = getattr(self, "src", None)
        if src is not None:
            line += f" | {src.summary()}"
        print(line, flush=True)
