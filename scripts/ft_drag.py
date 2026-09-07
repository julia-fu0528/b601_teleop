#!/usr/bin/env python3
"""Hand-guide the B601-RS with an ATI Nano25 F/T sensor at the wrist, and read the wrench.

Standalone: adds NO changes to the existing code - it reuses the safe GravityDragController for
the arm and streams the F/T sensor alongside. Use it once the Nano25 is mounted between joint6
and the gripper (your printed mounts).

  # bench test, no hardware at all (stub sensor + simulator arm):
  python scripts/ft_drag.py --sim --no-ft

  # real arm, real sensor (plain gravity-comp drag) - set your Net F/T box IP:
  python scripts/ft_drag.py --ft-ip 192.168.1.1 --cpf 1000000 --cpt 1000000

  # same, with the balance layer on:
  python scripts/ft_drag.py --ft-ip 192.168.1.1 --balance 1

Keys/behaviour of the arm are exactly the existing `gravity_drag.py drag` (RAMP/FADE/DRAG/HOLD,
d/h/r/q!, Ctrl+C, and the balance keys when --balance is used). The F/T stream prints on its own
line and, with --log-ft, is written with wall-clock timestamps you can align against the arm
--log CSV.

IMPORTANT (deliberately deferred, per your note): the mounted sensor adds ~63 g and ~22 mm of
reach at the wrist that the current gravity/inertia model does NOT know about, so gravity comp
will be slightly under (the wrist may sag a little) and FK to the gripper is ~22 mm short. That's
fine for reading the sensor now; recalibrate gravity/friction and add the sensor offset later.
The wrench is raw+tared in the SENSOR frame - no tool-gravity or frame transform yet.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b601 import ArmDynamics, GravityDragController, load_config  # noqa: E402
from b601 import ft_sensor  # noqa: E402


def build_assist(cfg, dyn, kappa):
    """Compact BalancedDrag builder (mirrors the essentials of gravity_drag.py's --balance)."""
    from b601.balance import BalancedDrag, FRIC as FRIC_DEFAULT
    f_kin = np.array([j.fric_kinetic if j.fric_kinetic > 0 else FRIC_DEFAULT[i]
                      for i, j in enumerate(cfg.joints)])
    f_sta = np.array([j.fric_static if j.fric_static > 0 else f_kin[i]
                      for i, j in enumerate(cfg.joints)])
    mu_kin = np.array([j.fric_kinetic_mu for j in cfg.joints])
    mu_sta = np.array([j.fric_static_mu if j.fric_static > 0 else mu_kin[i]
                       for i, j in enumerate(cfg.joints)])
    return BalancedDrag(dyn, kappa=kappa, fric_scale=(0.0 if kappa == 0 else 0.85),
                        fric=f_kin, f_static=f_sta, fric_mu=mu_kin, f_static_mu=mu_sta,
                        sustain_joints=np.ones(len(cfg.joints), bool),
                        tau_cap=0.4 * np.array([j.tau_max for j in cfg.joints]))


class FTStreamer:
    """Runs the F/T read in a thread: prints at a modest rate and optionally logs with timestamps."""
    def __init__(self, sensor, hz=5.0, log_path=None):
        self.s = sensor
        self.dt = 1.0 / max(hz, 0.1)
        self.log_path = log_path
        self._stop = threading.Event()
        self._t = None
        self._f = None

    def start(self):
        if self.log_path:
            self._f = open(self.log_path, "w")
            self._f.write("wall_time,Fx,Fy,Fz,Tx,Ty,Tz,status\n")
        self._t = threading.Thread(target=self._loop, name="ft-print", daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            w = self.s.read()
            st = getattr(self.s, "status", 0)
            if self._f is not None:
                self._f.write(f"{time.time():.4f}," + ",".join(f"{x:.4f}" for x in w) + f",{st}\n")
                self._f.flush()
            print("  F/T  F=[%+6.2f %+6.2f %+6.2f] N   T=[%+6.3f %+6.3f %+6.3f] N.m%s"
                  % (*w, "" if getattr(self.s, "connected", True) else "   (no data)"), flush=True)
            time.sleep(self.dt)

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)
        if self._f is not None:
            self._f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "b601_rs.toml"))
    ap.add_argument("--scale", type=float, default=1.0, help="global multiplier on gravity feed-forward")
    ap.add_argument("--kd", type=float, default=None, help="MIT damping override, all joints (default: config kd_drag)")
    ap.add_argument("--balance", type=float, default=None, metavar="KAPPA",
                    help="run the balance layer at this kappa (0..2); omit for plain gravity-comp drag")
    ap.add_argument("--joints", default=None, help="comma list of joints to energize (others disabled)")
    ap.add_argument("--gripper", choices=["hold", "off"], default="hold")
    ap.add_argument("--log", default=None, help="arm telemetry CSV (GravityDragController --log)")
    # ---- F/T sensor ----
    ap.add_argument("--ft-ip", default=None, help="ATI Net F/T box IP (required for a real sensor)")
    ap.add_argument("--ft-port", type=int, default=ft_sensor.RDT_PORT)
    ap.add_argument("--cpf", type=float, default=1_000_000.0, help="Net F/T counts-per-force (VERIFY on your box)")
    ap.add_argument("--cpt", type=float, default=1_000_000.0, help="Net F/T counts-per-torque (VERIFY on your box)")
    ap.add_argument("--no-ft", action="store_true", help="don't use a sensor (stub reads zero)")
    ap.add_argument("--ft-hz", type=float, default=5.0, help="F/T print/log rate")
    ap.add_argument("--log-ft", default=None, help="F/T CSV with wall-clock timestamps (align vs --log)")
    ap.add_argument("--no-tare", action="store_true", help="skip the resting tare of the sensor")
    # ---- arm backend ----
    ap.add_argument("--sim", action="store_true", help="use the simulator instead of hardware")
    ap.add_argument("--sim-truth", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true", help="with --sim: don't sleep")
    ap.add_argument("--duration", type=float, default=None, help="seconds of DRAG before auto-HOLD")
    ap.add_argument("--auto-release", action="store_true")
    ap.add_argument("--no-keys", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dyn = ArmDynamics(cfg.urdf, cfg.joint_names, cfg.lock_joints,
                      [j.g_scale for j in cfg.joints], [j.g_bias for j in cfg.joints])

    # ---- arm backend
    n = dyn.nq
    active = np.ones(n, bool)
    if args.joints:
        want = [s.strip() for s in args.joints.split(",")]
        bad = [w for w in want if w not in cfg.joint_names]
        if bad:
            raise SystemExit(f"unknown joints {bad}; choose from {cfg.joint_names}")
        active = np.array([nm in want for nm in cfg.joint_names])
    if args.sim:
        from b601.sim import SimArm
        arm = SimArm(dyn, np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0]), 1.0 / cfg.loop.rate_hz, truth_scale=args.sim_truth)
        print(f"SIMULATION arm (truth gravity scale {args.sim_truth})")
    else:
        from b601.arm import RobstrideArm
        arm = RobstrideArm(cfg)

    assist = build_assist(cfg, dyn, args.balance) if args.balance is not None else None
    if assist is not None:
        print(f"balance layer ON (kappa {args.balance})")

    kd = None if args.kd is None else np.full(n, args.kd)
    ctrl = GravityDragController(
        arm, dyn, cfg, assist=assist, scale=args.scale, kd=kd, active=active,
        gripper_hold=(args.gripper == "hold"),
        duration=args.duration, auto_release=args.auto_release, log_path=args.log,
        interactive=not args.no_keys, realtime=not (args.sim and args.fast),
    )

    # ---- F/T sensor
    if args.no_ft or (args.sim and not args.ft_ip):
        sensor = ft_sensor.StubFTSensor()
        print("F/T sensor: STUB (reads zero)")
    else:
        if not args.ft_ip:
            raise SystemExit("give --ft-ip (the Net F/T box IP), or pass --no-ft to run without a sensor")
        sensor = ft_sensor.NetFTSensor(args.ft_ip, args.ft_port, args.cpf, args.cpt)
        print(f"F/T sensor: ATI Net F/T at {args.ft_ip}:{args.ft_port} (cpf {args.cpf:g}, cpt {args.cpt:g})")

    streamer = FTStreamer(sensor, hz=args.ft_hz, log_path=args.log_ft)
    try:
        sensor.start()
        if not args.no_tare and hasattr(sensor, "n_recv"):
            print("taring F/T sensor (keep the arm still, nothing touching the gripper)...")
            b = sensor.tare(200)
            print("  bias:", np.round(b, 3))
        streamer.start()
        ctrl.run()
    finally:
        streamer.stop()
        try:
            sensor.stop()
        except Exception:
            pass
        arm.close()
    print("done.")


if __name__ == "__main__":
    main()
