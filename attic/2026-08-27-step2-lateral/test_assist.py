"""Offline tests for drag rebalancing step 2: LateralAssist takeover servo (b601/assist.py).

A sideways push at the gripper spins the low-friction wrist while joint1 stays stuck;
the assist must servo joint1 to carry that lateral motion - including when joint1's
stiction is above the torque a hand-push can ever build through the yielding wrist
(the real-arm case step 1's below-breakaway cap could not handle). It must also do
nothing on its own, nothing for non-lateral pushes, and stop joint1 with the hand.
Run:  python tests/test_assist.py   (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b601 import ArmDynamics, GravityDragController, Phase, load_config  # noqa: E402
from b601.assist import LateralAssist  # noqa: E402
from b601.sim import SimArm  # noqa: E402

CFG = load_config(ROOT / "config" / "b601_rs.toml")
# no g_scale/g_bias: these tests exercise the assist algorithm in a world where the
# gravity model is exact (sim truth = raw URDF gravity), so the hardware-fitted
# calibration in the config must not leak a ff-vs-truth mismatch into the sim
DYN = ArmDynamics(CFG.urdf, CFG.joint_names, CFG.lock_joints)
# real-arm-like friction: proximal RS-06 sticky, wrist RS-00 light -> the anisotropy under test
COULOMB = np.array([0.5, 0.5, 0.5, 0.25, 0.2, 0.2])
T_DRAG = CFG.loop.ramp_s + CFG.loop.fade_s          # sim time when DRAG starts
# a pose like the one that provoked the request: forearm out, wrist folded (j4 bent)
Q_WRIST = np.array([0.0, 1.08, 1.22, -1.0, 0.0, 0.0])


def push_at_ee(F, t0=T_DRAG + 0.3, t1=T_DRAG + 2.3, v_stop=0.25):
    """A hand pushing at gripper_end during [t0, t1]: the force fades as the EE moves along
    with it (a real hand cannot keep full force on something running away at v_stop m/s)."""
    F = np.asarray(F, float)
    fdir = F / (np.linalg.norm(F) + 1e-9)
    state = {"t": None, "p": None}

    def ext(t, q):
        if not (t0 < t < t1):
            state["t"] = None
            return np.zeros(6)
        Jv = DYN.ee_jacobian(q)[:3]
        p = _ee_p(q)
        scale = 1.0
        if state["t"] is not None and t > state["t"]:
            along = float(((p - state["p"]) / (t - state["t"])) @ fdir)
            scale = float(np.clip(1.0 - along / v_stop, 0.0, 1.0))
        state["t"], state["p"] = t, p
        return Jv.T @ (F * scale)
    return ext


def _ee_p(q):
    """gripper_end position in the base frame."""
    import pinocchio as pin
    pin.forwardKinematics(DYN.model, DYN.data, np.asarray(q, float))
    pin.updateFramePlacements(DYN.model, DYN.data)
    return np.array(DYN.data.oMf[DYN._ee_fid].translation)


def run_drag(q0, *, external=None, lateral=None, duration=3.0, coulomb=COULOMB):
    dt = 1.0 / CFG.loop.rate_hz
    arm = SimArm(DYN, np.array(q0, float), dt, coulomb=np.asarray(coulomb, float), external=external)
    ctrl = GravityDragController(arm, DYN, CFG, lateral=lateral, duration=duration, auto_release=True,
                                 interactive=False, realtime=False, print_every=0, hold_timeout=1.0)
    phase = ctrl.run()
    assert phase is Phase.DONE and ctrl.freeze_reason is None, \
        f"phase={phase} freeze={ctrl.freeze_reason}"
    return arm, ctrl


def test_lateral_push_recruits_j1():
    """The step-1 scenario: sideways push spins only j5 in plain drag; with assist j1 carries it."""
    q0 = Q_WRIST
    # force sized to sit under j1's breakaway but over the wrist's: the premise of the problem
    lever1 = abs((DYN.ee_jacobian(q0).T @ [0, 1, 0, 0, 0, 0])[0])
    F = [0.0, 0.85 * COULOMB[0] / lever1, 0.0]
    tau0 = DYN.ee_jacobian(q0)[:3].T @ F
    assert abs(tau0[0]) < COULOMB[0], "premise: the push alone must not break j1 loose"
    assert np.any(np.abs(tau0[3:]) > COULOMB[3:] * 1.1), "premise: the push must move some wrist joint"

    arm_p, ctrl_p = run_drag(q0, external=push_at_ee(F))
    arm_a, ctrl_a = run_drag(q0, external=push_at_ee(F), lateral=LateralAssist(DYN))
    d1_plain = abs(ctrl_p.q_drag_end[0] - q0[0])
    d1_assist = abs(ctrl_a.q_drag_end[0] - q0[0])
    assert d1_plain < 0.02, f"premise: plain drag moved j1 by {d1_plain:.3f}"
    assert d1_assist > max(0.06, 3 * d1_plain), \
        f"assist must recruit j1: plain {d1_plain:.4f} rad, assisted {d1_assist:.4f} rad"
    dy = _ee_p(ctrl_a.q_drag_end)[1] - _ee_p(q0)[1]
    assert dy > 0.03, f"gripper should truly translate sideways, dy={dy:.3f}"


def test_recruits_j1_when_stiction_exceeds_old_cap():
    """The real-arm failure: j1 breakaway above the old tip-the-balance cap (~0.4-0.5 N.m).
    The servo's cap exceeds breakaway, so it must still break j1 loose - and j1 must stop
    once the hand does (safety comes from the velocity target, not the torque cap)."""
    sticky = COULOMB.copy()
    sticky[0] = 0.8
    lever1 = abs((DYN.ee_jacobian(Q_WRIST).T @ [0, 1, 0, 0, 0, 0])[0])
    F = [0.0, 0.85 * COULOMB[0] / lever1, 0.0]      # same hand push as the base scenario
    arm, ctrl = run_drag(Q_WRIST, external=push_at_ee(F), lateral=LateralAssist(DYN), coulomb=sticky)
    d1 = abs(ctrl.q_drag_end[0] - Q_WRIST[0])
    assert d1 > 0.06, f"servo must break j1 loose at 0.8 N.m stiction, moved {d1:.4f} rad"
    assert abs(ctrl.v_drag_end[0]) < 0.2, f"j1 must stop with the hand, v1={ctrl.v_drag_end[0]:+.2f} rad/s"


def test_no_self_drive():
    """With no hand, the assist must not move anything (engage = 0 -> zero torque),
    even though its torque cap is above joint1's breakaway."""
    arm, ctrl = run_drag(Q_WRIST, lateral=LateralAssist(DYN))
    drift = np.abs(ctrl.q_drag_end - Q_WRIST)
    assert np.all(drift < 0.05), f"assist alone moved the arm: {np.round(drift, 3)}"


def test_vertical_push_is_ignored():
    """Selectivity: an up-push is perpendicular to the j1 direction -> j1 stays put."""
    arm, ctrl = run_drag(Q_WRIST, external=push_at_ee([0.0, 0.0, 5.0]),
                         lateral=LateralAssist(DYN))
    assert abs(ctrl.q_drag_end[0] - Q_WRIST[0]) < 0.03, \
        f"vertical push must not rotate j1, moved {ctrl.q_drag_end[0] - Q_WRIST[0]:+.3f} rad"


def test_plain_drag_unchanged_without_assist():
    """lateral=None must leave the existing behavior untouched: floats and holds."""
    q0 = [0.0, 0.7, 1.1, 0.0, 0.0, 0.0]
    arm, ctrl = run_drag(q0)
    drift = np.abs(ctrl.q_drag_end - np.array(q0))
    assert np.all(drift < 0.05), drift


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
