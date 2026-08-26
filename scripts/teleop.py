#!/usr/bin/env python3
"""B601-RS teleoperation with the Seeed reBot Arm 102 leader.

  compare    read-only: leader (mapped) vs follower, side by side; motors stay off. Move joints on the
             leader and check the follower column moves the same way later / verify the zero.
  calibrate  read-only: both arms in the SAME pose -> store leader offsets so mapped == follower now.
  run        teleoperate (ENGAGE glides to the leader pose first). Ctrl+C = hold, again = release.

  python scripts/teleop.py compare
  python scripts/teleop.py run --log teleop.csv
  python scripts/teleop.py run --sim            # simulator follower + scripted leader
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b601 import ArmDynamics, load_config  # noqa: E402
from b601.leader import map_to_follower  # noqa: E402
from b601.teleop import TeleopController  # noqa: E402

NAMES = ["j1 shoulder_pan", "j2 shoulder_lift", "j3 elbow", "j4 wrist_flex", "j5 wrist_yaw", "j6 wrist_roll", "gripper"]


def make_dyn(cfg):
    return ArmDynamics(cfg.urdf, cfg.joint_names, cfg.lock_joints, [j.g_scale for j in cfg.joints], [j.g_bias for j in cfg.joints])


def read_both(cfg):
    from b601.arm import RobstrideArm
    from b601.leader import RebotLeader
    arm = RobstrideArm(cfg)
    leader = RebotLeader(cfg.leader)
    return arm, leader


def cmd_compare(cfg, args):
    arm, leader = read_both(cfg)
    print("read-only (motors off). Ctrl+C to stop. Move a leader joint: 'mapped' must move the way the follower should.")
    try:
        while True:
            q = np.degrees(np.append(arm.read_q(), arm.read_gripper_q()))
            ld = leader.read_deg()
            mapped = map_to_follower(ld, cfg.leader)
            print("\033[2J\033[H" + f"{'joint':18s} {'leader raw':>10s} {'mapped':>8s} {'follower':>9s} {'delta':>7s}")
            for nm, l, m, f in zip(NAMES, ld, mapped, q):
                print(f"{nm:18s} {l:10.1f} {m:8.1f} {f:9.1f} {f - m:7.1f}")
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        leader.close(); arm.close()


def cmd_calibrate(cfg, args):
    arm, leader = read_both(cfg)
    try:
        input("Put the FOLLOWER and the LEADER in the same pose (e.g. both at rest), then press Enter...")
        q = np.degrees(np.append(arm.read_q(), arm.read_gripper_q()))
        lds = np.array([leader.read_deg() for _ in range(10)]).mean(0)
        dirs = np.asarray(cfg.leader.directions, float)
        offsets = lds - q / dirs           # so that direction * (leader - offset) == follower now
        out = ROOT / "config" / "leader_calib.json"
        out.write_text(json.dumps({"offsets": [round(float(x), 3) for x in offsets], "note": "leader deg at follower zero, per joint"}, indent=1))
        print("offsets (deg):", np.round(offsets, 2)); print("saved", out)
    finally:
        leader.close(); arm.close()


def cmd_run(cfg, args):
    dyn = make_dyn(cfg)
    if args.sim:
        from b601.sim import SimArm
        from b601.leader import ScriptedLeader
        q0 = np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0])
        arm = SimArm(dyn, q0, 1.0 / cfg.loop.rate_hz)
        dirs = np.asarray(cfg.leader.directions, float)
        base = np.degrees(np.append(q0, 0.0)) / dirs

        def script(t):  # leader wiggles the elbow +/-20 deg at 0.3 Hz, shoulder +/-10 deg at 0.2 Hz
            d = base.copy()
            d[2] += 20.0 * np.sin(2 * np.pi * 0.3 * max(0.0, t - 3.0)) / dirs[2]
            d[1] += 10.0 * np.sin(2 * np.pi * 0.2 * max(0.0, t - 3.0)) / dirs[1]
            return d
        leader = ScriptedLeader(script)
        print("SIMULATION: scripted leader, simulated follower")
    else:
        from b601.arm import RobstrideArm
        from b601.leader import LeaderReader, RebotLeader
        arm = RobstrideArm(cfg)
        leader = LeaderReader(RebotLeader(cfg.leader)).start()
    ctrl = TeleopController(
        arm, dyn, cfg, leader, gripper=not args.no_gripper, kp_scale=args.kp_scale,
        duration=args.duration, auto_release=args.auto_release, hold_timeout=args.hold_timeout,
        log_path=args.log, print_every=args.print_every, interactive=not args.no_keys,
        realtime=not (args.sim and args.fast), force_engage=args.force_engage,
        vel_ff=(False if args.no_vel_ff else None), lead_s=args.lead,
    )
    try:
        phase = ctrl.run()
    finally:
        try:
            leader.close()
        finally:
            arm.close()
    if ctrl.track_err_log:
        e = np.degrees(np.array(ctrl.track_err_log))
        print("tracking error while TRACK (deg): rms", np.round(np.sqrt((e ** 2).mean(0)), 2), " max", np.round(np.abs(e).max(0), 1))
    print("final phase:", phase.value, "| freeze reason:", ctrl.freeze_reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "b601_rs.toml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.add_argument("--hz", type=float, default=5.0)
    sub.add_parser("calibrate")
    r = sub.add_parser("run")
    r.add_argument("--log"); r.add_argument("--no-gripper", action="store_true")
    r.add_argument("--kp-scale", type=float, default=1.0, help="multiply the tracking kp (start lower, e.g. 0.5, for a softer follower)")
    r.add_argument("--duration", type=float); r.add_argument("--auto-release", action="store_true")
    r.add_argument("--hold-timeout", type=float); r.add_argument("--print-every", type=float, default=0.5)
    r.add_argument("--no-keys", action="store_true"); r.add_argument("--sim", action="store_true"); r.add_argument("--fast", action="store_true")
    r.add_argument("--force-engage", action="store_true", help="engage even if a joint is > max_engage_deg from the leader")
    r.add_argument("--no-vel-ff", action="store_true", help="disable the target-velocity feed-forward (laggy; for A/B only)")
    r.add_argument("--lead", type=float, help="override teleop.lead_s (s of target prediction)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.leader is None:
        raise SystemExit("config has no [leader] section")
    {"compare": cmd_compare, "calibrate": cmd_calibrate, "run": cmd_run}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
