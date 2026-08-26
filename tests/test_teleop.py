"""Teleop tests on the simulator: mapping, engage-without-jump, tracking, glitch guard, refusal."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b601 import ArmDynamics, load_config  # noqa: E402
from b601.leader import ScriptedLeader, map_to_follower, unwrap  # noqa: E402
from b601.sim import SimArm  # noqa: E402
from b601.teleop import TPhase, TeleopController  # noqa: E402

CFG = load_config(ROOT / "config" / "b601_rs.toml")
DIRS = np.asarray(CFG.leader.directions, float)


def dyn():
    return ArmDynamics(CFG.urdf, CFG.joint_names, CFG.lock_joints, [j.g_scale for j in CFG.joints], [j.g_bias for j in CFG.joints])


def leader_deg_for(q_rad6, grip_deg=0.0):
    """Leader servo degrees that map exactly onto a follower pose."""
    return np.append(np.degrees(q_rad6), grip_deg) / DIRS + np.asarray(CFG.leader.offsets)


def run(q0, script, *, duration=3.0, **kw):
    d = dyn()
    arm = SimArm(d, np.array(q0, float), 1.0 / CFG.loop.rate_hz)
    ctrl = TeleopController(arm, d, CFG, ScriptedLeader(script), duration=duration, auto_release=True,
                            hold_timeout=1.0, interactive=False, realtime=False, print_every=0, **kw)
    return arm, ctrl, ctrl.run()


def test_unwrap_and_mapping():
    assert unwrap(370.0, -150, 150) == 10.0 and unwrap(-350.0, -150, 150) == 10.0
    assert unwrap(-190.0, -200, 1) == -190.0            # elbow range window is centred at -99.5
    deg = np.array([10, 20, -30, -40, -50, 60, 45.0])
    m = map_to_follower(deg, CFG.leader)
    assert np.allclose(m, [10, 20, 30, 40, 50, 60, 270.0])


def test_engage_glides_without_jump_and_tracks():
    q0 = np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0])
    q_far = q0 + np.array([0.3, 0.2, -0.3, 0.2, 0.0, 0.3])          # leader 17 deg away on several joints
    lead = leader_deg_for(q_far)
    arm, ctrl, phase = run(q0, lambda t: lead, duration=2.0)
    assert phase is TPhase.DONE and ctrl.freeze_reason is None
    assert ctrl.max_engage_v < 0.8, f"engage was too fast: {ctrl.max_engage_v:.2f} rad/s (limit 0.4 + PD overshoot)"
    err = np.degrees(np.array(ctrl.track_err_log))
    assert np.abs(err[-50:]).max() < 3.0, f"steady tracking error {np.abs(err[-50:]).max():.2f} deg"


def test_tracks_moving_leader():
    q0 = np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0])
    base = leader_deg_for(q0)

    def script(t):
        d = base.copy()
        d[2] += 20.0 * np.sin(2 * np.pi * 0.3 * max(0.0, t - 2.0)) / DIRS[2]
        return d
    arm, ctrl, phase = run(q0, script, duration=6.0)
    assert phase is TPhase.DONE and ctrl.freeze_reason is None
    err = np.degrees(np.array(ctrl.track_err_log))
    assert np.sqrt((err[:, 2] ** 2).mean()) < 4.0, "elbow should follow a 0.3 Hz +/-20 deg leader motion"


def test_velocity_feedforward_cuts_lag():
    q0 = np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0])
    base = leader_deg_for(q0)

    def script(t):
        d = base.copy()
        d[2] += 25.0 * np.sin(2 * np.pi * 0.4 * max(0.0, t - 2.0)) / DIRS[2]
        return d
    rms = {}
    for ff in (False, True):
        _, ctrl, phase = run(q0, script, duration=6.0, vel_ff=ff, lead_s=0.03 if ff else 0.0)
        assert phase is TPhase.DONE and ctrl.freeze_reason is None
        err = np.degrees(np.array(ctrl.track_err_log))[100:]
        rms[ff] = float(np.sqrt((err[:, 2] ** 2).mean()))
    assert rms[True] < 0.5 * rms[False], f"velocity ff should cut the elbow lag: {rms}"


def test_leader_glitch_is_ignored_then_holds():
    q0 = np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0])
    base = leader_deg_for(q0)

    def script(t):
        d = base.copy()
        if t > 3.0:
            d[1] += 100.0          # persistent 100 deg jump on the shoulder reading
        return d
    arm, ctrl, phase = run(q0, script, duration=5.0)
    assert ctrl.freeze_reason is not None and "jumped" in ctrl.freeze_reason
    assert arm.max_abs_v < 1.5, "the glitch must not be followed"
    assert phase is TPhase.DONE


def test_refuses_far_leader():
    q0 = np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0])
    lead = leader_deg_for(q0 + np.array([0, 0, 0, 0, 0, 2.5]))    # 143 deg away on wrist roll
    try:
        run(q0, lambda t: lead)
    except RuntimeError as e:
        assert "away from the leader" in str(e)
    else:
        raise AssertionError("expected refusal")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed"); sys.exit(1 if failed else 0)
