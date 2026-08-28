#!/usr/bin/env python3
"""B601-RS gravity-compensated drag-teach.

  read   no torque: stream joint angles + the model feed-forward that *would* be sent (safe sign probe:
         move joints by hand with motors off; lifting joint2/joint3 from rest must read positive)
  drag   MIT feed-forward only (kp=0): the arm floats, drag it by hand.  Ctrl+C = hold, again = release.

Examples
  python scripts/gravity_drag.py read
  python scripts/gravity_drag.py drag --sim                       # no hardware, exercise the safety logic
  python scripts/gravity_drag.py drag --joints joint3             # single joint first
  python scripts/gravity_drag.py drag --kd 0                      # pure gravity feed-forward, no damping
  python scripts/gravity_drag.py drag --log run.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b601 import ArmDynamics, GravityDragController, load_config  # noqa: E402


def parse_kd(s: str | None, n: int) -> np.ndarray | None:
    if s is None:
        return None
    parts = [float(x) for x in s.split(",")]
    if len(parts) == 1:
        return np.full(n, parts[0])
    if len(parts) != n:
        raise SystemExit(f"--kd needs 1 or {n} values")
    return np.array(parts)


def cmd_read(cfg, dyn, args) -> None:
    from b601.arm import RobstrideArm
    arm = RobstrideArm(cfg)
    names = cfg.joint_names
    print("read-only: motors stay disabled. Ctrl+C to stop.")
    print("hint: from the rest pose, lifting joint2 or joint3 by hand must read POSITIVE (URDF j2/j3 in [0, pi]).")
    t_end = time.monotonic() + args.seconds if args.seconds else None
    try:
        while t_end is None or time.monotonic() < t_end:
            q = arm.read_q()
            g = dyn.gravity(q) * args.scale
            gq = arm.read_gripper_q() if arm.gripper is not None else float("nan")
            viol = dyn.limit_violation(q)
            print("q(rad) " + " ".join(f"{n}={x:+.3f}" for n, x in zip(names, q))
                  + f" grip={gq:+.3f} | ff(N.m) " + " ".join(f"{x:+.2f}" for x in g)
                  + ("  LIMIT!" if np.any(viol > 0.05) else ""), flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()


def cmd_drag(cfg, dyn, args) -> None:
    n = dyn.nq
    names = cfg.joint_names
    active = np.ones(n, bool)
    if args.joints:
        want = [s.strip() for s in args.joints.split(",")]
        bad = [w for w in want if w not in names]
        if bad:
            raise SystemExit(f"unknown joints {bad}; choose from {names}")
        active = np.array([nm in want for nm in names])
    kd = parse_kd(args.kd, n)

    if args.sim:
        from b601.sim import SimArm
        q0 = np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0])
        arm = SimArm(dyn, q0, 1.0 / cfg.loop.rate_hz, truth_scale=args.sim_truth)
        print(f"SIMULATION (no hardware): truth gravity scale {args.sim_truth}")
    else:
        from b601.arm import RobstrideArm
        arm = RobstrideArm(cfg)

    assist = None
    if args.assist and args.dyn_assist:
        raise SystemExit("--assist and --dyn-assist are alternatives; pick one")
    if args.assist:
        from b601.assist import TorqueRebalance
        if not 0.0 < args.assist <= 2.0:
            raise SystemExit("--assist: per-joint torque cap in (0, 2] N.m (config tau_max clamps are 3+)")
        kd_all = kd if kd is not None else np.array([j.kd_drag for j in cfg.joints])
        assist = TorqueRebalance(dyn, tau_max=args.assist, gain=args.assist_gain, kd_down=kd_all[3:])
        print(f"torque rebalance ACTIVE: sensed hand force moved to joints 1-3 (J^T size, J+ direction, "
              f"gain x{args.assist_gain}, cap {args.assist} N.m/joint); status line shows tau_dn -> tau_up, w, tx")
    elif args.dyn_assist:
        from b601.assist import DynamicAssist
        if not 0.0 < args.dyn_assist <= 2.0:
            raise SystemExit("--dyn-assist: per-joint torque cap in (0, 2] N.m")
        assist = DynamicAssist(dyn, tau_max=args.dyn_assist, gain=args.assist_gain)
        print(f"dynamic assist ACTIVE (inverse-dynamics observer): all joints carry gain x the felt EE force "
              f"(gain x{args.assist_gain}, cap {args.dyn_assist} N.m/joint); status line shows F and tx")

    ctrl = GravityDragController(
        arm, dyn, cfg, assist=assist,
        scale=args.scale, kd=kd, kp=args.kp, fric_scale=args.fric, active=active,
        gripper_hold=(args.gripper == "hold"),
        duration=args.duration, auto_release=args.auto_release,
        log_path=args.log, calib_path=args.calib, capture_s=args.capture_s, capture_amp=args.capture_amp, print_every=args.print_every,
        interactive=not args.no_keys, realtime=not (args.sim and args.fast),
        hold_timeout=args.hold_timeout,
    )
    try:
        phase = ctrl.run()
    finally:
        arm.close()
    print("final phase:", phase.value, "| freeze reason:", ctrl.freeze_reason)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "b601_rs.toml"))
    ap.add_argument("--scale", type=float, default=1.0, help="global multiplier on the gravity feed-forward")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("read", help="stream positions + model feed-forward, motors disabled")
    r.add_argument("--hz", type=float, default=10.0)
    r.add_argument("--seconds", type=float, help="stop after this many seconds (default: until Ctrl+C)")

    d = sub.add_parser("drag", help="gravity-compensated drag (MIT feed-forward)")
    d.add_argument("--joints", help="comma list of joints to energize (others stay disabled), e.g. joint3")
    d.add_argument("--kd", help="MIT damping: one value or per-joint list; 0 = pure feed-forward (default: config kd_drag)")
    d.add_argument("--kp", type=float, help="MIT stiffness while dragging, all joints (default: config kp_drag, normally 0 = free)")
    d.add_argument("--fric", type=float, default=1.0, help="multiplier on the config fric_comp values (0 = no friction compensation)")
    d.add_argument("--gripper", choices=["hold", "off"], default="hold", help="hold the gripper at its current position, or leave it unpowered")
    d.add_argument("--duration", type=float, help="seconds of DRAG before automatically going to HOLD")
    d.add_argument("--auto-release", action="store_true", help="with --duration: release (fade + disable) instead of waiting in HOLD")
    d.add_argument("--log", help="CSV telemetry file")
    d.add_argument("--calib", default="calib.csv", help="CSV that the 'c' key appends pose/residual captures to")
    d.add_argument("--capture-s", type=float, default=5.0, help="capture duration: 1 s settle + whole 2 s sweep periods")
    d.add_argument("--capture-amp", type=float, default=0.05, help="sweep amplitude per joint during capture (rad)")
    d.add_argument("--print-every", type=float, default=0.5)
    d.add_argument("--no-keys", action="store_true", help="no stdin/SIGINT handling (scripts, tests)")
    d.add_argument("--hold-timeout", type=float, help="leave HOLD automatically (release) after this many seconds")
    d.add_argument("--vel-abort", type=float, help="override loop.vel_abort (rad/s)")
    d.add_argument("--ramp-window", type=float, help="override loop.ramp_window (rad)")
    d.add_argument("--rate", type=float, help="override loop.rate_hz")
    d.add_argument("--assist", "--lateral", dest="assist", type=float, metavar="TAU",
                   help="isotropic takeover: sense the hand force from the backdriving wrist and transform it "
                        "(Jacobian transpose) to joints 1-3, so dragging feels the same in every direction; "
                        "TAU = per-joint torque cap (N.m), above breakaway on purpose, try 1.2")
    d.add_argument("--assist-gain", "--lateral-gain", dest="assist_gain", type=float, default=2.0, metavar="G",
                   help="force multiplication wrist -> base: base joints get G x the torque the sensed hand "
                        "force exerts on them, dividing their apparent stiction by ~(1+G)")
    d.add_argument("--dyn-assist", type=float, metavar="TAU",
                   help="alternative to --assist: inverse-dynamics observer (M qdd + C qd + g + fric - tau_motor "
                        "= J^T F) senses the hand force at the EE, and ALL joints carry gain x it - whole-arm "
                        "power assist; TAU = per-joint torque cap (N.m), try 1.2")
    d.add_argument("--sim", action="store_true", help="run against the built-in simulator instead of hardware")
    d.add_argument("--sim-truth", type=float, default=1.0, help="simulator's true gravity scale (e.g. -1 to test the runaway guard)")
    d.add_argument("--fast", action="store_true", help="with --sim: don't sleep (faster than realtime)")

    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "drag":
        import dataclasses
        over = {k: v for k, v in (("vel_abort", args.vel_abort), ("ramp_window", args.ramp_window), ("rate_hz", args.rate)) if v is not None}
        if over:
            cfg = dataclasses.replace(cfg, loop=dataclasses.replace(cfg.loop, **over))
    dyn = ArmDynamics(cfg.urdf, cfg.joint_names, cfg.lock_joints, [j.g_scale for j in cfg.joints], [j.g_bias for j in cfg.joints])
    if args.cmd == "read":
        cmd_read(cfg, dyn, args)
    else:
        cmd_drag(cfg, dyn, args)


if __name__ == "__main__":
    main()
