"""Offline tests: dynamics sanity + controller phase machine on the simulator.
Run:  python tests/test_gravity_drag.py   (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b601 import ArmDynamics, GravityDragController, Phase, load_config  # noqa: E402
from b601.sim import SimArm  # noqa: E402

CFG = load_config(ROOT / "config" / "b601_rs.toml")


def make_dyn(scaled: bool = True) -> ArmDynamics:
    return ArmDynamics(CFG.urdf, CFG.joint_names, CFG.lock_joints,
                       [j.g_scale for j in CFG.joints] if scaled else None,
                       [j.g_bias for j in CFG.joints] if scaled else None)


def run_sim(q0, *, truth=1.0, scale=1.0, duration=2.0, kd=None, external=None, rate=None):
    dyn = make_dyn()
    dt = 1.0 / (rate or CFG.loop.rate_hz)
    arm = SimArm(dyn, np.array(q0, float), dt, truth_scale=truth, external=external)
    ctrl = GravityDragController(arm, dyn, CFG, scale=scale, kd=kd, duration=duration, auto_release=True,
                                 interactive=False, realtime=False, print_every=0, hold_timeout=1.0)
    phase = ctrl.run()
    return arm, ctrl, phase


def test_gravity_at_rest_matches_reference():
    dyn = make_dyn(scaled=False)
    g0 = dyn.gravity_raw(np.zeros(6))
    assert np.allclose(g0, [0.0, 1.545, 6.764, 2.001, 0.0, 0.0], atol=2e-3), g0
    assert dyn.nq == 6 and dyn.model.names[2] == "joint2"


def test_scale_applies_per_joint():
    dyn = make_dyn(scaled=True)
    raw, sc = dyn.gravity_raw(np.zeros(6)), dyn.gravity(np.zeros(6))
    k = np.array([j.g_scale for j in CFG.joints]); c = np.array([j.g_bias for j in CFG.joints])
    assert np.allclose(sc, k * raw + c) and k[0] == 1.0 and c[0] == 0.0


def test_sim_holds_pose_with_correct_feedforward():
    q0 = [0.0, 0.7, 1.1, 0.0, 0.0, 0.0]
    arm, ctrl, phase = run_sim(q0, truth=1.0, scale=1.0, duration=2.0)
    assert phase is Phase.DONE and ctrl.freeze_reason is None
    assert ctrl.q_drag_end is not None
    drift = np.abs(ctrl.q_drag_end - np.array(q0))
    assert np.all(drift < 0.05), f"arm drifted {drift}"
    assert not arm.enabled.any()  # released -> disabled


def test_sim_wrong_sign_is_caught_during_ramp():
    q0 = [0.0, 0.7, 1.1, 0.0, 0.0, 0.0]
    arm, ctrl, phase = run_sim(q0, truth=1.0, scale=-1.0, duration=2.0)
    assert ctrl.freeze_reason is not None, "runaway not detected"
    assert ctrl.startup_failed or "velocity" in ctrl.freeze_reason
    assert arm.max_abs_v < 6.0  # caught before it got violent


def test_sim_drag_by_external_torque_then_stops():
    q0 = [0.0, 0.7, 1.1, 0.0, 0.0, 0.0]
    # a "hand" pushes joint3 with 1.0 N.m for 0.4 s after ramp+fade (5 s), then lets go
    def hand(t, q):
        ext = np.zeros(6)
        if 5.5 < t < 5.9:
            ext[2] = 1.0
        return ext
    arm, ctrl, phase = run_sim(q0, duration=3.0, external=hand)
    assert phase is Phase.DONE and ctrl.freeze_reason is None
    assert ctrl.q_drag_end[2] > q0[2] + 0.05, "joint3 should have been moved by the hand"
    assert abs(ctrl.v_drag_end[2]) < 0.05, "arm should come to rest after the hand lets go"


def test_start_outside_limits_refuses():
    dyn = make_dyn()
    arm = SimArm(dyn, np.zeros(6), 0.01)
    arm.read_q = lambda: np.array([0.0, -0.5, 1.0, 0, 0, 0])   # bypass the sim's hard stops
    ctrl = GravityDragController(arm, dyn, CFG, interactive=False, realtime=False, print_every=0)
    try:
        ctrl.run()
    except RuntimeError as e:
        assert "outside URDF joint limits" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert not arm.enabled.any()


def test_friction_sweep_recovers_sim_coulomb():
    """The 'f' sweep must measure the simulator's Coulomb friction (0.3 N.m) on the loaded joints."""
    dyn = make_dyn()
    dt = 1.0 / CFG.loop.rate_hz
    arm = SimArm(dyn, np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0]), dt, coulomb=0.3, viscous=0.0)
    ctrl = GravityDragController(arm, dyn, CFG, interactive=False, realtime=False, print_every=0,
                                 duration=12.0, auto_release=True, hold_timeout=1.0)
    ctrl._cmds.put("f")            # queued before the loop: processed once DRAG is reached? no - it is consumed in RAMP
    # so queue it via a scripted hook: process commands only in DRAG by re-queuing when not in DRAG
    orig_freeze = ctrl._freeze
    ran = {"done": False}
    def run_with_key():
        # push 'f' 0.5 s after DRAG starts by wrapping _say
        orig_say = ctrl._say
        def say(msg):
            orig_say(msg)
            if "-> DRAG" in msg and not ran["done"]:
                ran["done"] = True
                ctrl._cmds.put("f")
        ctrl._say = say
        return ctrl.run()
    phase = run_with_key()
    assert phase is Phase.DONE
    fr = getattr(ctrl, "friction_result", None)
    assert fr is not None, "friction sweep did not run"
    assert np.all(np.abs(fr[1:4] - 0.3) < 0.1), f"Coulomb estimate off: {fr}"


class _FakeArm:
    """Feedback for enabled joints, params for the rest; joint 1's feedback can be made wrong."""
    def __init__(self, n=6, bad_fb_joint=None):
        self.n, self.bad = n, bad_fb_joint
        self.q_true = np.linspace(0.1, 0.6, n)
        self.enabled = np.array([True, True, False, False, False, False])
        self.param_reads = 0
    def read_q(self):
        return self.q_true.copy()
    def read_q_param(self, idx):
        self.param_reads += len(idx)
        return np.array([self.q_true[i] for i in idx])
    def poll_states(self):
        from b601.sim import SimState
        out = []
        for i in range(self.n):
            if not self.enabled[i]:
                out.append(None)
            else:
                pos = self.q_true[i] + (0.5 if i == self.bad else 0.0)
                out.append(SimState(pos, 0.0, 1.0))
        return out


def test_position_source_uses_feedback_and_falls_back_on_mismatch():
    from b601.arm import PositionSource
    arm = _FakeArm(bad_fb_joint=1)
    src = PositionSource(arm, arm.enabled, verify_every=2, tol=0.03, max_mismatch=2)
    for _ in range(40):
        q = src.update()
    assert src.src[0] == "fb" and src.src[2] == "param"
    assert src.fallback[1], "joint 1 feedback disagreed with mechPos and must fall back"
    assert np.allclose(q, arm.q_true, atol=1e-9), q
    assert arm.param_reads < 40 * 6, "must not read every joint every cycle"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
