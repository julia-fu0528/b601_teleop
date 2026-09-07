"""Offline tests for the torque-rebalance pipeline (b601/assist.py), built step by step:
  1. measure the external torque on joints 4-6          [tested here]
  2. infer the external torque on joints 1-3 via the Jacobian
  3. compare the halves: downstream torque-loading vs upstream motion-carriage
  4. move torque from the yielding half to the stuck half (J^T size, J+ direction)
Run:  python tests/test_assist.py   (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b601 import ArmDynamics, GravityDragController, Phase, load_config  # noqa: E402
from b601.assist import TorqueRebalance  # noqa: E402
from b601.sim import SimArm  # noqa: E402

CFG = load_config(ROOT / "config" / "b601_rs.toml")
# no g_scale/g_bias: these tests exercise the assist algorithms in a world where the
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
    with it (a real hand cannot keep full force on something running away at v_stop m/s).
    The force actually applied each instant is exposed as ext.state["Fs"]."""
    F = np.asarray(F, float)
    fdir = F / (np.linalg.norm(F) + 1e-9)
    state = {"t": None, "p": None, "Fs": np.zeros(3)}

    def ext(t, q):
        if not (t0 < t < t1):
            state["t"] = None
            state["Fs"] = np.zeros(3)
            return np.zeros(6)
        Jv = DYN.ee_jacobian(q)[:3]
        p = _ee_p(q)
        scale = 1.0
        if state["t"] is not None and t > state["t"]:
            along = float(((p - state["p"]) / (t - state["t"])) @ fdir)
            scale = float(np.clip(1.0 - along / v_stop, 0.0, 1.0))
        state["t"], state["p"] = t, p
        state["Fs"] = F * scale
        return Jv.T @ (F * scale)
    ext.state = state
    return ext


def _ee_p(q):
    """gripper_end position in the base frame."""
    import pinocchio as pin
    pin.forwardKinematics(DYN.model, DYN.data, np.asarray(q, float))
    pin.updateFramePlacements(DYN.model, DYN.data)
    return np.array(DYN.data.oMf[DYN._ee_fid].translation)


def run_drag(q0, *, external=None, assist=None, duration=3.0, coulomb=COULOMB):
    dt = 1.0 / CFG.loop.rate_hz
    arm = SimArm(DYN, np.array(q0, float), dt, coulomb=np.asarray(coulomb, float), external=external)
    ctrl = GravityDragController(arm, DYN, CFG, assist=assist, duration=duration, auto_release=True,
                                 interactive=False, realtime=False, print_every=0, hold_timeout=1.0)
    # Pin the friction feed-forward like the sim's COULOMB is pinned: these tests exercise the
    # drag/assist mechanisms, and must not shift when the hardware calibration in the config does.
    ctrl.fric_comp = np.array([0.35, 0.35, 0.35, 0.0, 0.0, 0.0])
    phase = ctrl.run()
    assert phase is Phase.DONE and ctrl.freeze_reason is None, \
        f"phase={phase} freeze={ctrl.freeze_reason}"
    return arm, ctrl


class _MeasureProbe(TorqueRebalance):
    """Records (measured tau_down, true external torque on j4-6) each control cycle."""

    def __init__(self, dyn, ext, **kw):
        super().__init__(dyn, **kw)
        self.ext = ext
        self.hist = []

    def update(self, q, v, dt=0.01, **kw):
        tau = super().update(q, v, dt, **kw)
        true_down = (DYN.ee_jacobian(q)[:3].T @ self.ext.state["Fs"])[3:]
        self.hist.append((self.tau_down.copy(), true_down))
        return tau


def test_step1_downstream_measurement():
    """A radial push backdrives j4 (its lever is ~2x j2/j3's): the measured external
    torque on j4 must track the true one, and the non-yielding j5/j6 must read ~0."""
    q0 = Q_WRIST
    lev = DYN.ee_jacobian(q0)[:3].T @ [1.0, 0.0, 0.0]
    F = [1.8 * COULOMB[3] / abs(lev[3]), 0.0, 0.0]
    ext = push_at_ee(F)
    probe = _MeasureProbe(DYN, ext, gain=0.0)
    arm, ctrl = run_drag(q0, external=ext, assist=probe)

    h_meas = np.array([m for m, t in probe.hist])
    h_true = np.array([t for m, t in probe.hist])
    steady = np.abs(h_true[:, 0]) > 0.1          # samples where the push truly loads j4
    assert steady.sum() > 50, "premise: the push must load j4 for a sustained stretch"
    m4, t4 = h_meas[steady, 0].mean(), h_true[steady, 0].mean()
    assert np.sign(m4) == np.sign(t4), f"measured j4 torque has the wrong sign: {m4:+.2f} vs {t4:+.2f}"
    assert 0.4 < m4 / t4 < 1.6, f"measured j4 torque off: {m4:+.3f} N.m vs true {t4:+.3f} N.m"
    assert np.abs(h_meas[:, 1:]).max() < 0.08, \
        f"j5/j6 do not yield here and must read ~0, got {np.round(np.abs(h_meas[:, 1:]).max(0), 3)}"
    assert np.abs(ctrl.q_drag_end[:3] - q0[:3]).max() < 0.02, \
        "with gain 0 the probe must inject nothing: the upstream joints may not move"


class _InferProbe(TorqueRebalance):
    """Records (inferred tau_up, true upstream torque, inferred F, applied F) each cycle."""

    def __init__(self, dyn, ext, **kw):
        super().__init__(dyn, **kw)
        self.ext = ext
        self.hist = []

    def update(self, q, v, dt=0.01, **kw):
        tau = super().update(q, v, dt, **kw)
        true_up = (DYN.ee_jacobian(q)[:3].T @ self.ext.state["Fs"])[:3]
        self.hist.append((self.tau_up.copy(), true_up, self.F.copy(),
                          self.ext.state["Fs"].copy(), self.tau_down.copy(), q.copy()))
        return tau


def test_step2_lateral_inference():
    """Good observability: a lateral push backdrives j5, whose lever spans the push
    direction, so the inferred j1 torque must track the true one (sign + magnitude)."""
    q0 = Q_WRIST
    lev = DYN.ee_jacobian(q0)[:3].T @ [0.0, 1.0, 0.0]
    ext = push_at_ee([0.0, 0.85 * COULOMB[0] / abs(lev[0]), 0.0])
    probe = _InferProbe(DYN, ext, gain=0.0)
    arm, ctrl = run_drag(q0, external=ext, assist=probe)
    h = probe.hist
    s = [i for i, r in enumerate(h) if np.linalg.norm(r[3]) > 0.3]
    assert len(s) > 50, "premise: sustained push"
    m1 = np.mean([h[i][0][0] for i in s])
    t1 = np.mean([h[i][1][0] for i in s])
    assert np.sign(m1) == np.sign(t1), f"inferred j1 torque wrong sign: {m1:+.2f} vs {t1:+.2f}"
    assert 0.5 < m1 / t1 < 1.5, f"inferred j1 torque off: {m1:+.3f} vs true {t1:+.3f} N.m"
    assert np.abs(ctrl.q_drag_end[:3] - q0[:3]).max() < 0.02, "with gain 0 the probe must inject nothing"


def test_step2_fit_explains_measurement():
    """Limited observability: with only j4 yielding (radial push) the force DIRECTION is
    ambiguous (min-norm pick), but the fit must at least explain what was measured:
    J_down^T F_hat ~ tau_down."""
    q0 = Q_WRIST
    lev = DYN.ee_jacobian(q0)[:3].T @ [1.0, 0.0, 0.0]
    ext = push_at_ee([1.8 * COULOMB[3] / abs(lev[3]), 0.0, 0.0])
    probe = _InferProbe(DYN, ext, gain=0.0)
    run_drag(q0, external=ext, assist=probe)
    errs = []
    for tau_up, true_up, F, Fs, tau_dn, q in probe.hist:
        if np.linalg.norm(Fs) > 0.3 and np.abs(tau_dn).max() > 0.1:
            pred = (DYN.ee_jacobian(q)[:3][:, 3:]).T @ F
            errs.append(np.abs(pred - tau_dn).max())
    assert len(errs) > 50, "premise: sustained push"
    assert np.mean(errs) < 0.05, f"F_hat fails to explain tau_down, mean err {np.mean(errs):.3f} N.m"


class _CompareProbe(TorqueRebalance):
    """Records (r_down, r_up, weight, applied F) each cycle."""

    def __init__(self, dyn, ext, **kw):
        super().__init__(dyn, **kw)
        self.ext = ext
        self.hist = []

    def update(self, q, v, dt=0.01, **kw):
        tau = super().update(q, v, dt, **kw)
        self.hist.append((self.r_down, self.carry_up, self.weight,
                          float(np.linalg.norm(self.ext.state["Fs"]))))
        return tau


def test_step3_gate_opens_under_push():
    """While the wrist carries a push and the upstream half sits under its breakaway,
    the transfer gate must open; without a hand it must stay shut."""
    q0 = Q_WRIST
    lev = DYN.ee_jacobian(q0)[:3].T @ [1.0, 0.0, 0.0]
    ext = push_at_ee([1.8 * COULOMB[3] / abs(lev[3]), 0.0, 0.0])
    probe = _CompareProbe(DYN, ext, gain=0.0)
    run_drag(q0, external=ext, assist=probe)
    h = np.array(probe.hist)
    s = h[:, 3] > 0.3                          # samples with the hand truly pushing
    assert s.sum() > 50, "premise: sustained push"
    assert h[s, 0].mean() > 0.6, f"r_down should show the wrist loaded, got {h[s, 0].mean():.2f}"
    assert h[s, 1].mean() < 0.3, f"upstream is stuck, carry_up should be ~0, got {h[s, 1].mean():.2f}"
    assert h[s, 2].mean() > 0.4, f"gate should open under the push, got {h[s, 2].mean():.2f}"

    probe0 = _CompareProbe(DYN, push_at_ee([0.0, 0.0, 0.0]), gain=0.0)
    run_drag(q0, assist=probe0)
    h0 = np.array(probe0.hist)
    assert h0[:, 2].max() < 0.15, f"gate must stay shut with no hand, got {h0[:, 2].max():.2f}"


def test_step1_zero_at_rest():
    """No hand: the measurement must read ~0 and nothing may move."""
    probe = _MeasureProbe(DYN, push_at_ee([0.0, 0.0, 0.0]))
    arm, ctrl = run_drag(Q_WRIST, assist=probe)
    h_meas = np.array([m for m, t in probe.hist])
    assert np.abs(h_meas).max() < 0.05, f"phantom torque at rest: {np.round(np.abs(h_meas).max(0), 3)}"
    drift = np.abs(ctrl.q_drag_end - Q_WRIST)
    assert np.all(drift < 0.05), f"moved with no hand: {np.round(drift, 3)}"


def test_step4_lateral_recruits_j1():
    """A sideways push spins only j5 in plain drag; the transfer must recruit j1 so the
    gripper truly translates."""
    q0 = Q_WRIST
    lever1 = abs((DYN.ee_jacobian(q0).T @ [0, 1, 0, 0, 0, 0])[0])
    F = [0.0, 0.85 * COULOMB[0] / lever1, 0.0]     # under j1's direct breakaway, over the wrist's
    arm_p, ctrl_p = run_drag(q0, external=push_at_ee(F))
    arm_a, ctrl_a = run_drag(q0, external=push_at_ee(F), assist=TorqueRebalance(DYN))
    d1_plain = abs(ctrl_p.q_drag_end[0] - q0[0])
    d1_assist = abs(ctrl_a.q_drag_end[0] - q0[0])
    assert d1_plain < 0.02, f"premise: plain drag moved j1 by {d1_plain:.3f}"
    assert d1_assist > max(0.06, 3 * d1_plain), \
        f"transfer must recruit j1: plain {d1_plain:.4f} rad, assisted {d1_assist:.4f} rad"
    dy = _ee_p(ctrl_a.q_drag_end)[1] - _ee_p(q0)[1]
    assert dy > 0.03, f"gripper should truly translate sideways, dy={dy:.3f}"


def test_step4_high_stiction_j1():
    """j1 breakaway at 0.8 N.m (above anything a hand-push can build through the yielding
    wrist): the transfer must still break it loose, and j1 must stop with the hand."""
    sticky = COULOMB.copy()
    sticky[0] = 0.8
    lever1 = abs((DYN.ee_jacobian(Q_WRIST).T @ [0, 1, 0, 0, 0, 0])[0])
    F = [0.0, 0.85 * COULOMB[0] / lever1, 0.0]
    arm, ctrl = run_drag(Q_WRIST, external=push_at_ee(F), assist=TorqueRebalance(DYN), coulomb=sticky)
    d1 = abs(ctrl.q_drag_end[0] - Q_WRIST[0])
    assert d1 > 0.05, f"transfer must break j1 loose at 0.8 N.m stiction, moved {d1:.4f} rad"
    assert abs(ctrl.v_drag_end[0]) < 0.2, f"j1 must stop with the hand, v1={ctrl.v_drag_end[0]:+.2f} rad/s"


def test_step4_radial_recruits_shoulder():
    """A radial push bends only j4 in plain drag; the transfer must recruit the shoulder,
    never press a joint against the motion (the J+ direction guarantee), and leave j1 put."""
    q0 = Q_WRIST
    lev = DYN.ee_jacobian(q0)[:3].T @ [1.0, 0.0, 0.0]
    F = [1.8 * COULOMB[3] / abs(lev[3]), 0.0, 0.0]
    arm_p, ctrl_p = run_drag(q0, external=push_at_ee(F))
    arm_a, ctrl_a = run_drag(q0, external=push_at_ee(F), assist=TorqueRebalance(DYN))
    dsh_plain = np.abs(ctrl_p.q_drag_end[1:3] - q0[1:3]).max()
    d2, d3 = ctrl_a.q_drag_end[1] - q0[1], ctrl_a.q_drag_end[2] - q0[2]
    dx_plain = _ee_p(ctrl_p.q_drag_end)[0] - _ee_p(q0)[0]
    dx_assist = _ee_p(ctrl_a.q_drag_end)[0] - _ee_p(q0)[0]
    assert max(abs(d2), abs(d3)) > max(0.03, 2 * dsh_plain), \
        f"transfer must recruit the shoulder: plain {dsh_plain:.4f}, assisted dq2={d2:+.4f} dq3={d3:+.4f}"
    assert d2 > -0.01 and d3 > -0.01, \
        f"no joint may be pressed against the outward motion: dq2={d2:+.4f} dq3={d3:+.4f}"
    assert dx_assist > dx_plain + 0.02, \
        f"gripper should truly translate outward: plain dx={dx_plain:.3f}, assisted dx={dx_assist:.3f}"
    assert abs(ctrl_a.q_drag_end[0] - q0[0]) < 0.03, \
        f"radial push must not rotate j1, moved {ctrl_a.q_drag_end[0] - q0[0]:+.3f} rad"


def test_plain_drag_unchanged_without_assist():
    """assist=None must leave the existing behavior untouched: floats and holds."""
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
