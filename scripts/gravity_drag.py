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
    if args.observe and args.balance is None:
        args.balance = 0.0
    if args.assist and args.balance is not None:
        raise SystemExit("--assist and --balance are separate experiments; pick one")
    if args.balance is not None:
        from b601.balance import BalancedDrag
        if not 0.0 <= args.balance <= 2.0:
            raise SystemExit("--balance: kappa in [0, 2] (stability ceiling of the ~90 Hz loop; "
                             "0 = observer only). See the balanced-drag derivation.")
        from b601.balance import FRIC as FRIC_DEFAULT
        # friction levels: calibrated config values ('f' and 's' sweeps) when present,
        # module defaults otherwise; f_static falls back to kinetic (no extra breakaway kick)
        f_kin = np.array([j.fric_kinetic if j.fric_kinetic > 0 else FRIC_DEFAULT[i]
                          for i, j in enumerate(cfg.joints)])
        f_sta = np.array([j.fric_static if j.fric_static > 0 else f_kin[i]
                          for i, j in enumerate(cfg.joints)])
        mu_kin = np.array([j.fric_kinetic_mu for j in cfg.joints])
        # static falls back to the kinetic slope when only the kinetic model was fitted
        mu_sta = np.array([j.fric_static_mu if j.fric_static > 0 else mu_kin[i]
                           for i, j in enumerate(cfg.joints)])
        sus_arg = str(args.balance_sustain).strip().lower()
        if args.observe or sus_arg in ("none", "0", "off"):
            want = []
        elif sus_arg in ("all", "1", "on", "default"):
            want = list(cfg.joint_names)
        else:
            want = [w.strip() for w in args.balance_sustain.split(",")]
        bad = [w for w in want if w not in cfg.joint_names]
        if bad:
            raise SystemExit(f"--balance-sustain: unknown joints {bad}; choose from {cfg.joint_names} or 'none'")
        sustain_mask = np.array([nm in want for nm in cfg.joint_names])
        # three independent toggles: kappa (inertia shaping), fric scale, sustain.
        # --observe zeroes them all: pure estimator logging.
        fs_raw = {"on": 0.85, "off": 0.0}.get(str(args.balance_fric).strip().lower(), None)
        if fs_raw is None:
            try:
                fs_raw = float(args.balance_fric)
            except ValueError:
                raise SystemExit("--balance-fric: fraction in [0..1], or on/off")
        fscale = 0.0 if args.observe else min(max(fs_raw, 0.0), 1.0)
        if args.observe:
            args.balance = 0.0
        assist = BalancedDrag(
            dyn, kappa=args.balance, m_d=args.balance_md, i_rot=args.balance_irot,
            f_o=args.balance_fo, resist=args.balance_resist,
            fric_scale=fscale, fric=f_kin, f_static=f_sta,
            fric_mu=mu_kin, f_static_mu=mu_sta,
            sustain_joints=sustain_mask,
            fric_viscous=np.array([j.fric_viscous for j in cfg.joints]),
            damp_t=args.balance_damp[0], damp_r=args.balance_damp[1],
            break_beta=args.balance_breakaway,
            alpha_sigma_v=args.balance_alpha_vel, alpha_kappa0=args.balance_alpha_sing,
            detent_kp=args.balance_detent,
            tau_cap=0.4 * np.array([j.tau_max for j in cfg.joints]))
        # two calibrated KINETIC friction models for live A/B ('fmodel <name>' or the web toggle);
        # statics (breakaway) are shared. "viscous" = whatever the config holds (fv COMBINED fit
        # 2026-09-07: de-biased tau_c + mu*|g(q)| + B*qd). "load" = the previous friction_v2 fit
        # 2026-09-05 (PD-effort): higher tau_c + mu*|g(q)| on j2, no viscous.
        assist.fric_models = {
            "viscous": {"kin": f_kin.tolist(), "mu": mu_kin.tolist(),
                        "viscous": [j.fric_viscous for j in cfg.joints]},
            # "flat": the SAME fv sweep refitted WITHOUT the load term (2026-09-07 am) - the pooled
            # tau_c absorbs average load (j2 0.27 vs combined 0.18+mu). Refit-style ablation baseline:
            # NOT reproducible by toggling mu off (that keeps the combined fit's lower intercept).
            "flat": {"kin": [0.43, 0.27, 0.36, 0.14, 0.22, 0.16],
                     "mu": [0.0] * len(cfg.joints),
                     "viscous": [0.1086, 0.0, 0.0, 0.0723, 0.0, 0.1121]},
            "load": {"kin": [0.52, 0.48, 0.68, 0.28, 0.24, 0.25],
                     "mu": [0.0, 0.018, 0.0, 0.0, 0.0, 0.0],
                     "viscous": [0.0] * len(cfg.joints)},
        }
        assist.set_fric_model(args.balance_fric_model)
        assist.set_fric_terms(mu=args.balance_mu == "on", viscous=args.balance_viscous == "on")
        if fscale > 0.0:
            if args.fric is None:
                args.fric = 0.0
                print("note: config fric_comp disabled (--fric 0) - the calibrated gated friction ff replaces it; "
                      "pass --fric explicitly to force both")
            elif args.fric > 0.0:
                raise SystemExit("--fric > 0 together with the balance friction ff would compensate the same "
                                 "friction twice (e.g. j3: 0.35 + 0.62 N.m vs ~0.73 real). Use --balance-fric "
                                 "alone (calibrated, gated), or --balance-fric 0 to fall back to fric_comp.")
        calibrated = any(j.fric_kinetic > 0 or j.fric_static > 0 for j in cfg.joints)
        fric_src = ("config (f/s sweeps)" if calibrated
                    else "module defaults - calibrate with the 'f' and 's' keys")
        if np.any(mu_kin > 0) or np.any(mu_sta > 0):
            fric_src += ", load-dependent (f0 + mu*|g(q)|)"
        if np.any(sustain_mask) and fscale > 0:
            sus_names = [n for n, m in zip(cfg.joint_names, sustain_mask) if m]
            fric_src += ("; sustained relief on ALL joints (margin-capped)" if len(sus_names) == len(cfg.joint_names)
                         else f"; sustained relief on {sus_names} (margin-capped)")
        if args.balance == 0.0 and fscale == 0.0:
            mode = "OBSERVE-ONLY (zero output; r on the status line / log)"
        elif args.balance == 0.0:
            mode = "friction compensation only (inertia shaping OFF)"
        else:
            mode = f"kappa {args.balance} (arm up to {1.0 + args.balance:.0f}x lighter, resist floor -{assist.resist:.2f})"
        print(f"balanced drag ACTIVE: momentum observer @ {args.balance_fo} Hz + Cartesian shaping, {mode}; "
              f"target {args.balance_md} kg / {args.balance_irot} kg.m^2 at the gripper, "
              f"fric ff x{fscale} of {fric_src}; "
              f"live keys: m <kg> | i <kg.m^2> | + | - retarget the virtual inertia")
    if args.assist:
        from b601.assist import TorqueRebalance
        if not 0.0 < args.assist <= 2.0:
            raise SystemExit("--assist: per-joint torque cap in (0, 2] N.m (config tau_max clamps are 3+)")
        kd_all = kd if kd is not None else np.array([j.kd_drag for j in cfg.joints])
        assist = TorqueRebalance(dyn, tau_max=args.assist, gain=args.assist_gain, kd_down=kd_all[3:])
        print(f"torque rebalance ACTIVE: sensed hand force moved to joints 1-3 (J^T size, J+ direction, "
              f"gain x{args.assist_gain}, cap {args.assist} N.m/joint); status line shows tau_dn -> tau_up, w, tx")

    ctrl = GravityDragController(
        arm, dyn, cfg, assist=assist,
        scale=args.scale, kd=kd, kp=args.kp,
        fric_scale=1.0 if args.fric is None else args.fric, active=active,
        gripper_hold=(args.gripper == "hold"),
        duration=args.duration, auto_release=args.auto_release,
        log_path=args.log, calib_path=args.calib, fric_path=args.fric_csv,
        capture_s=args.capture_s, capture_amp=args.capture_amp,
        vfric_path=args.vfric_csv, vfric_speeds=args.vfric_speeds,
        vfric_period=args.vfric_period, vfric_amp_max=args.vfric_amp_max,
        vfric_periods=args.vfric_periods,
        print_every=args.print_every,
        interactive=not args.no_keys, realtime=not (args.sim and args.fast),
        hold_timeout=args.hold_timeout,
    )
    srv = None
    if args.serve is not None:
        if not hasattr(assist, "set_kappa"):
            raise SystemExit("--serve needs --balance (the web panel controls the balance layer)")
        from b601 import webserve
        html = Path(args.serve_html) if args.serve_html else ROOT / "balance_panel.html"
        srv = webserve.start(ctrl, html, port=args.serve)
        print(f"web control panel: http://127.0.0.1:{args.serve}  (open it; moving a slider prints + "
              f"applies here live). Serving {html.name}.")

    try:
        phase = ctrl.run()
    finally:
        if srv is not None:
            srv.shutdown()
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
    d.add_argument("--fric", type=float, default=None,
                   help="multiplier on the config fric_comp values (default 1.0; 0 = off). With --balance and "
                        "its own friction ff active, fric_comp defaults to OFF instead - the calibrated gated "
                        "path replaces it, and running both would over-compensate")
    d.add_argument("--gripper", choices=["hold", "off"], default="hold", help="hold the gripper at its current position, or leave it unpowered")
    d.add_argument("--duration", type=float, help="seconds of DRAG before automatically going to HOLD")
    d.add_argument("--auto-release", action="store_true", help="with --duration: release (fade + disable) instead of waiting in HOLD")
    d.add_argument("--log", help="CSV telemetry file")
    d.add_argument("--calib", default="calib.csv", help="CSV that the 'c' key appends pose/residual captures to")
    d.add_argument("--fric-csv", default="friction.csv",
                   help="CSV that the 'f' (kinetic) and 's' (static) sweeps append to, one row per sweep per "
                        "pose; aggregate with scripts/fit_friction.py")
    d.add_argument("--capture-s", type=float, default=5.0, help="capture duration: 1 s settle + whole 2 s sweep periods")
    d.add_argument("--capture-amp", type=float, default=0.05, help="sweep amplitude per joint during capture (rad)")
    d.add_argument("--vfric-csv", default="friction_v.csv",
                   help="CSV the 'fv' velocity-friction sweep appends to (v, q, torq-g per joint); "
                        "fit B + de-biased tau_c with scripts/fit_friction.py")
    d.add_argument("--vfric-speeds", type=lambda s: tuple(float(x) for x in s.split(",")),
                   default=(0.03, 0.06, 0.09, 0.12, 0.16, 0.20), metavar="V1,V2,...",
                   help="constant speeds (rad/s) for the 'fv' current-based sweep "
                        "(default 0.03,0.06,0.09,0.12,0.16,0.2 - fine, low-speed range)")
    d.add_argument("--vfric-period", type=float, default=3.0,
                   help="'fv' triangle period (s); amplitude = speed*period/4 gives a real const-velocity dwell")
    d.add_argument("--vfric-amp-max", type=float, default=0.35, help="'fv' amplitude safety cap per joint (rad)")
    d.add_argument("--vfric-periods", type=int, default=2, help="'fv' whole triangle periods averaged per speed")
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
    d.add_argument("--observe", action="store_true",
                   help="estimator logging only: implies --balance with inertia shaping, friction comp and "
                        "sustain all OFF; r per joint on the status line / --log CSV. The safe first run")
    d.add_argument("--balance", dest="balance", type=float, metavar="KAPPA",
                   help="balanced drag (momentum observer + Cartesian inertia shaping): estimate the hand "
                        "torque from motion, then shape the felt inertia toward an isotropic mass at the "
                        "gripper - joints heavier than the target assist, lighter ones resist the wrist "
                        "running away. KAPPA = max lightening ratio minus 1, in [0, 2]; 0 = inertia shaping "
                        "OFF (friction comp / sustain still follow their own flags; use --observe for a "
                        "zero-output estimator run)")
    d.add_argument("--balance-md", type=float, default=1.8, metavar="KG",
                   help="target translational mass at the gripper (kg, default 1.8) - THE 'how heavy does it feel' knob")
    d.add_argument("--balance-irot", type=float, default=0.06, metavar="KGM2",
                   help="target rotational inertia about the tool pitch/yaw axes (kg.m^2, default 0.06; "
                        "roll stays natural until the rotor inertia is identified)")
    d.add_argument("--balance-fo", type=float, default=3.0, metavar="HZ",
                   help="observer bandwidth (Hz, default 3; higher is snappier but lowers the stable KAPPA)")
    d.add_argument("--balance-sustain", default="all", metavar="JOINTS",
                   help="1/0 (on/off) or a comma list of joints: friction relief STAYS ON while a joint is "
                        "clearly moving (> 0.1 rad/s), capped at (real kinetic level at the pose - margin) so "
                        "residual pushes below the margin still decelerate it (margins per joint, sized above "
                        "the measured model residuals). Default all; 'none'/0 = drive-gated everywhere "
                        "(relief only while out-pushing full friction)")
    d.add_argument("--balance-fric", default="0.85", metavar="S",
                   help="fraction [0..1] of the measured friction to compensate while moving, or on/off "
                        "(default 0.85; 0 = off). Uses fric_static/fric_kinetic from the config (Stribeck: breakaway level "
                        "at motion onset decaying to kinetic) - calibrate with the 's' and 'f' keys. The gate "
                        "on the estimated drive makes it creep-proof; ignored with --balance 0 (observe-only)")
    d.add_argument("--balance-fric-model", choices=["viscous", "flat", "load"], default="viscous",
                   help="which calibrated KINETIC friction model to start with: 'viscous' = fv combined fit "
                        "(tau_c + mu*|g| + B*qd, 2026-09-07 config), 'flat' = same sweep refit without "
                        "the load term (pooled tau_c + B), 'load' = previous friction_v2 PD-effort fit "
                        "(tau_c + mu*|g|). Toggle live with 'fmodel <name>' or the web panel")
    d.add_argument("--balance-mu", choices=["on", "off"], default="on",
                   help="load term mu*|g(q)| of the kinetic friction model: on = calibrated (default), "
                        "off = 0 (ablate load-dependence). Live: 'mu on|off' or the web toggle")
    d.add_argument("--balance-viscous", choices=["on", "off"], default="on",
                   help="viscous term B*qd of the kinetic friction model: on = calibrated (default), "
                        "off = 0 (ablate velocity-dependence). Live: 'visc on|off' or the web toggle")
    d.add_argument("--balance-resist", type=float, default=None, metavar="R",
                   help="resist floor in [0, 0.7]: directions may be made at most 1/(1-R)x heavier "
                        "(default kappa/(1+kappa), the mirror of the assist ratio)")
    d.add_argument("--serve", nargs="?", type=int, const=8730, default=None, metavar="PORT",
                   help="with --balance: serve the Balance Console web panel on localhost:PORT (default 8730). "
                        "Moving a slider / toggling a joint on the page changes kappa, friction, or per-joint "
                        "sustain in THIS running session live, and prints the change here. A sandboxed "
                        "claude.ai artifact can't reach the process - this local server is how the page drives it.")
    d.add_argument("--serve-html", default=None, help="path to the panel HTML to serve (default: repo balance_panel.html)")
    d.add_argument("--balance-damp", type=float, nargs=2, default=[0.0, 0.0], metavar=("D_T", "D_R"),
                   help="paper 5.1: Cartesian virtual damping (D_v) at the hand, tool frame - translational "
                        "N.s/m and rotational N.m.s/rad. Strictly dissipative (passive); dominates friction "
                        "uncertainty so it can replace the sustain margins. Default 0 0 (off); try 4 0.6")
    d.add_argument("--balance-breakaway", type=float, default=0.0, metavar="BETA",
                   help="paper 5.2: breakaway-assist fraction [0..1]; pre-pays BETA * static friction in the "
                        "estimated push direction (sign of r), decaying as the joint moves - helps break "
                        "stiction from rest without a F/T sensor. Default 0 (off); try 0.35")
    d.add_argument("--balance-alpha-vel", type=float, default=0.0, metavar="SIGMA",
                   help="paper eq 38: taper friction comp near zero velocity, scale = 1-exp(-(qd/SIGMA)^2) "
                        "(rad/s). Default 0 (off); try 0.05 (anti-chatter at standstill)")
    d.add_argument("--balance-alpha-sing", type=float, default=0.0, metavar="K0",
                   help="paper eq 39: taper friction comp near singularities, scale = min(1, K0/cond(J)). "
                        "Default 0 (off); try 25")
    d.add_argument("--balance-detent", type=float, default=0.0, metavar="KP",
                   help="latched low-speed restoring spring (N.m/rad): when a joint goes quiet it holds the "
                        "pose it stopped at, faded out while you guide. Unlike damping this HOLDS against a "
                        "static bias (fixes the near-vertical j2 lean-back). Default 0 (off); try 3-5. "
                        "Live: type 'detent <kp>'.")
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
