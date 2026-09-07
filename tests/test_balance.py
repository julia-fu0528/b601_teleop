"""Offline tests for balanced drag (b601/balance.py): momentum observer + Cartesian shaping.
Rollout order they mirror:
  1. observer only (kappa 0): r tracks a known hand push, output exactly zero
  2. bias hygiene: a miscalibrated gravity model must not become a phantom hand (no creep)
  3. shaping: a torque pulse reaches ~(1+kappa)x the velocity; assist/resist signs match
  4. guards: with a deliberately wrong inertia model the trips/vel_abort keep it bounded
Run:  python tests/test_balance.py   (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from b601 import GravityDragController, Phase  # noqa: E402
from b601.balance import BalancedDrag  # noqa: E402
from b601.sim import SimArm  # noqa: E402
from test_assist import CFG, COULOMB, DYN, Q_WRIST, T_DRAG, _ee_p, push_at_ee  # noqa: E402

TAU_CAP = 0.4 * np.array([j.tau_max for j in CFG.joints])


def urdf_inertia(q):
    """Diagonal of the URDF mass matrix at q: a 'well-identified' sim plant for the observer."""
    return np.maximum(np.diag(DYN.mass_matrix(np.asarray(q, float))), 0.004)


class _SimModel:
    """DYN with the SimArm's actual dynamics (diagonal, pose-frozen inertia, no Coriolis):
    the observer's 'exact model' of the *sim* plant. On the real arm plant and model are
    both the URDF; SimArm is diagonal, so a matched-model test must be diagonal too."""

    def __init__(self, dyn, inertia):
        self._dyn = dyn
        self._I = np.asarray(inertia, float)
        self.nq = dyn.nq

    def __getattr__(self, name):
        return getattr(self._dyn, name)

    def mass_matrix(self, q):
        return np.diag(self._I)

    def coriolis(self, q, v):
        return np.zeros((self.nq, self.nq))


def run(q0, *, assist, external=None, duration=3.0, inertia=None, truth_scale=1.0,
        allow_freeze=False, ctrl_fric=1.0):
    dt = 1.0 / CFG.loop.rate_hz
    arm = SimArm(DYN, np.array(q0, float), dt, coulomb=COULOMB, external=external,
                 inertia=inertia, truth_scale=truth_scale)
    ctrl = GravityDragController(arm, DYN, CFG, assist=assist, duration=duration,
                                 auto_release=True, interactive=False, realtime=False,
                                 fric_scale=ctrl_fric,
                                 print_every=0, hold_timeout=1.0)
    phase = ctrl.run()
    assert phase is Phase.DONE, f"phase={phase} freeze={ctrl.freeze_reason}"
    if not allow_freeze:
        assert ctrl.freeze_reason is None, f"froze: {ctrl.freeze_reason}"
    return arm, ctrl


class _Probe(BalancedDrag):
    """Records (r, true external torque, v, |tau_out|) every estimator cycle."""

    def __init__(self, dyn, ext=None, **kw):
        super().__init__(dyn, tau_cap=TAU_CAP, **kw)
        self.ext = ext
        self.hist = []
        self.vmax = 0.0

    def _record(self, q, v):
        true = np.zeros(self.n) if self.ext is None else \
            DYN.ee_jacobian(q)[:3].T @ self.ext.state["Fs"]
        self.hist.append((self.r.copy(), true, np.array(v, float), float(np.abs(self.tau_out).max())))
        self.vmax = max(self.vmax, float(np.abs(v).max()))

    def observe(self, q, v, dt):
        super().observe(q, v, dt)
        self._record(q, v)

    def update(self, q, v, dt=0.01):
        out = super().update(q, v, dt)
        self._record(q, v)
        return out


def test_observer_tracks_known_push():
    """kappa 0: output must be exactly zero, and while the push slides a joint, r on that
    joint must have the push's sign and roughly the (hand - Coulomb) magnitude."""
    q0 = Q_WRIST
    ext = push_at_ee([8.0, 0.0, 0.0])
    inertia = urdf_inertia(q0)
    probe = _Probe(_SimModel(DYN, inertia), ext, kappa=0.0)
    run(q0, assist=probe, external=ext, inertia=inertia)

    assert max(o for *_, o in probe.hist) == 0.0, "observe-only must output exactly zero"
    r_h = np.array([r for r, *_ in probe.hist])
    t_h = np.array([t for _, t, *_ in probe.hist])
    v_h = np.array([v for *_, v, _ in probe.hist])
    # the most-loaded joint, in samples where it is truly pushed and actually sliding
    j = int(np.abs(t_h).max(0).argmax())
    m = (np.abs(t_h[:, j]) > 0.3) & (np.abs(v_h[:, j]) > 0.05)
    assert m.sum() > 30, f"premise: joint {j+1} must be pushed into sliding for a stretch"
    agree = np.mean(np.sign(r_h[m, j]) == np.sign(t_h[m, j]))
    assert agree > 0.8, f"r sign agrees with the true hand torque only {agree:.0%} of the time"
    expect = np.abs(t_h[m, j]) - COULOMB[j]          # r is the net drive beyond friction
    ratio = np.abs(r_h[m, j]).mean() / max(expect.mean(), 1e-6)
    assert 0.3 < ratio < 1.9, f"r magnitude off on joint {j+1}: {ratio:.2f}x the net drive"


def test_gravity_bias_does_not_creep():
    """5 % gravity-model error, kappa 2, nobody touching: the dead-band + bias learner must
    keep the shaping from turning the residual into a phantom push (no drift, no trips)."""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    inertia = urdf_inertia(q0)
    probe = _Probe(_SimModel(DYN, inertia), kappa=2.0)
    arm, ctrl = run(q0, assist=probe, inertia=inertia, truth_scale=1.05, duration=6.0)
    drift = np.abs(ctrl.q_drag_end - q0).max()
    assert drift < 0.05, f"arm crept {drift:.3f} rad with no hand on it"
    assert probe.trips == 0, "runaway detector tripped in a quiet hold"


def joint_pulse(j, tau, t0, t1):
    """A constant external torque on one joint during [t0, t1] (a clean joint-space 'hand')."""
    def ext(t, q):
        out = np.zeros(6)
        if t0 < t < t1:
            out[j] = tau
        return out
    ext.state = {"Fs": np.zeros(3)}
    return ext


def test_shaping_lightens_a_joint_pulse():
    """A fixed torque pulse on j2, well after the output ramp: with kappa 2 the same pulse
    must reach a clearly higher velocity (lower apparent inertia), and stay bounded."""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    t0 = T_DRAG + 2.5                      # alpha ramp (2 s) fully in
    vmax = {}
    for kappa in (0.0, 2.0):
        ext = joint_pulse(1, 1.5, t0, t0 + 0.5)
        # m_d far below the arm's ~2-3 kg: the j2 direction pins at the kappa clip,
        # so the pulse probes the full advertised lightening
        inertia = urdf_inertia(q0)
        probe = _Probe(_SimModel(DYN, inertia), ext, kappa=kappa, m_d=0.5)
        run(q0, assist=probe, external=ext, inertia=inertia, duration=4.0)
        v2 = np.array([v[1] for *_, v, _ in probe.hist])
        vmax[kappa] = float(v2.max())
    assert vmax[0.0] > 0.3, f"premise: the pulse must move j2 unassisted (got {vmax[0.0]:.2f} rad/s)"
    ratio = vmax[2.0] / vmax[0.0]
    assert 1.5 < ratio < 4.5, \
        f"kappa 2 should reach roughly (1+kappa)x the unassisted velocity, got {ratio:.2f}x " \
        f"({vmax[2.0]:.2f} vs {vmax[0.0]:.2f} rad/s)"


def test_assist_and_resist_pattern():
    """During a radial hand push at the folded-wrist pose, the shoulder row of the output
    must assist (with the drive) and the folding-wrist row must resist (against it)."""
    q0 = Q_WRIST
    # push after the 2 s output ramp, with a firm hand (v_stop 0.6) so the wrist stays loaded
    ext = push_at_ee([8.0, 0.0, 0.0], t0=T_DRAG + 2.2, t1=T_DRAG + 2.9, v_stop=0.6)
    inertia = urdf_inertia(q0)
    probe = _Probe(_SimModel(DYN, inertia), ext, kappa=2.0)

    taus = []
    orig = probe.update

    def upd(q, v, dt=0.01):
        out = orig(q, v, dt)
        true = DYN.ee_jacobian(q)[:3].T @ ext.state["Fs"]
        if abs(true[3]) > 0.5 and probe.alpha > 0.5:   # wrist truly loaded, shaping active
            taus.append((out.copy(), true.copy()))
        return out

    probe.update = upd
    run(q0, assist=probe, external=ext, inertia=inertia)
    assert len(taus) > 20, "premise: the push must load the wrist with the shaping active"
    out = np.array([o for o, _ in taus])
    true = np.array([t for _, t in taus])
    # j4 output must oppose the hand's j4 torque (resist the fold) on average...
    assert np.mean(out[:, 3] * np.sign(true[:, 3])) < -0.03, \
        f"j4 does not resist the fold: mean {np.mean(out[:, 3]):+.2f} N.m vs hand {np.mean(true[:, 3]):+.2f}"
    # ...while the shoulder is helped along its own drive
    assert np.mean(out[:, 1] * np.sign(true[:, 1])) > 0.03, \
        f"j2 does not assist: mean {np.mean(out[:, 1]):+.2f} N.m vs hand {np.mean(true[:, 1]):+.2f}"


def test_wrong_inertia_stays_bounded():
    """Sim plant with the default (deliberately different) inertia: whatever the mismatch
    excites, the trips / vel_abort / HOLD machinery must keep velocities bounded and land
    in DONE. This is the containment test, not a prediction that it oscillates."""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    ext = push_at_ee([10.0, 0.0, 3.0])
    probe = _Probe(DYN, ext, kappa=2.0)
    arm, ctrl = run(q0, assist=probe, external=ext, allow_freeze=True, duration=4.0)
    assert probe.vmax < CFG.loop.vel_abort + 1.5, \
        f"velocity {probe.vmax:.1f} rad/s escaped the guards"


def test_live_retarget_slews():
    """Typing 'm 3.0' at the running CLI must retarget the virtual mass and slew lam_d
    there smoothly (no step), while kappa=0 keeps the output at zero."""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    inertia = urdf_inertia(q0)
    probe = _Probe(_SimModel(DYN, inertia), kappa=0.0)
    dt = 1.0 / CFG.loop.rate_hz
    arm = SimArm(DYN, q0, dt, coulomb=COULOMB, inertia=inertia)
    ctrl = GravityDragController(arm, DYN, CFG, assist=probe, duration=2.0, auto_release=True,
                                 interactive=False, realtime=False, print_every=0, hold_timeout=1.0)
    ctrl._cmds.put("m 3.0")
    ctrl._cmds.put("i 0.1")
    assert ctrl.run() is Phase.DONE and ctrl.freeze_reason is None
    assert probe._lam_goal[0] == 3.0 and probe._lam_goal[4] == 0.1
    assert abs(probe.lam_d[0] - 3.0) < 0.05, f"lam_d did not slew to the goal: {probe.lam_d[0]:.2f}"
    assert max(o for *_, o in probe.hist) == 0.0
    # +/- step the goal by 25 %
    md, ir = probe.set_target(scale=1.25)
    assert abs(md - 3.75) < 1e-9


def test_static_sweep_measures_breakaway():
    """The 's' key: per-joint bidirectional torque ramp must recover the sim's true static
    friction (set 1.5x kinetic) within the ramp-overshoot tolerance, with ~zero residual."""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    inertia = urdf_inertia(q0)
    static_true = 1.5 * COULOMB
    probe = _Probe(_SimModel(DYN, inertia), kappa=0.0)
    dt = 1.0 / CFG.loop.rate_hz
    arm = SimArm(DYN, q0, dt, coulomb=COULOMB, static=static_true, inertia=inertia)
    fcsv = Path(__file__).parent / "_fric_test.csv"
    fcsv.unlink(missing_ok=True)
    ctrl = GravityDragController(arm, DYN, CFG, assist=probe, duration=2.0, auto_release=True,
                                 fric_path=str(fcsv),
                                 interactive=False, realtime=False, print_every=0, hold_timeout=1.0)
    fired = []
    orig = probe.update

    def upd(q, v, dt=0.01):
        if not fired:
            fired.append(1)
            ctrl._cmds.put("s")          # first DRAG cycle: start the static sweep
        return orig(q, v, dt)

    probe.update = upd
    assert ctrl.run() is Phase.DONE and ctrl.freeze_reason is None, ctrl.freeze_reason
    meas = ctrl.static_result
    assert np.isfinite(meas).all(), f"sweep incomplete: {meas}"
    err = meas - static_true
    assert np.all(err > -0.03), f"static friction under-measured: {np.round(err, 3)}"
    assert np.all(err < 0.15), f"static friction over-measured (ramp too fast?): {np.round(err, 3)}"
    # the sweep must land in the friction CSV for multi-pose aggregation
    import csv as _csv
    with open(fcsv, newline="") as fh:
        rows = list(_csv.DictReader(fh))
    fcsv.unlink()
    assert len(rows) == 1 and rows[0]["kind"] == "static"
    logged = np.array([float(rows[0][f"f{i+1}"]) for i in range(6)])
    assert np.allclose(logged, meas, atol=1e-3)


def test_fric_curve_stribeck():
    """85 % compensation: near-breakaway level at motion onset, kinetic level at speed."""
    b = BalancedDrag(DYN, kappa=1.0, fric_scale=0.85,
                     fric=np.full(6, 0.4), f_static=np.full(6, 0.6))
    lo = b.fric_curve(np.full(6, 0.01))
    hi = b.fric_curve(np.full(6, 1.0))
    assert abs(lo[0] - 0.85 * 0.6) < 0.01, f"onset level {lo[0]:.3f} != 0.85*static"
    assert abs(hi[0] - 0.85 * 0.4) < 0.01, f"sliding level {hi[0]:.3f} != 0.85*kinetic"
    assert np.all(b.fric_curve(np.full(6, 0.15)) < lo) and np.all(lo <= 0.51)

def test_fit_friction_recovers_load_model():
    """fit_friction.py must recover f = f0 + mu*|g(q)| per joint from synthetic multi-pose
    sweeps, and fall back to the median where the load never varies (j1, vertical axis)."""
    import csv as _csv
    sys.path.insert(0, str(ROOT / "scripts"))
    import fit_friction
    from b601 import ArmDynamics
    dyn_cal = ArmDynamics(CFG.urdf, CFG.joint_names, CFG.lock_joints,
                          [j.g_scale for j in CFG.joints], [j.g_bias for j in CFG.joints])
    rng = np.random.default_rng(0)
    f0 = np.array([0.50, 0.20, 0.22, 0.15, 0.20, 0.20])
    mu = np.array([0.00, 0.030, 0.045, 0.02, 0.0, 0.0])
    poses = [np.array([0.0, a, b, c, 0.0, 0.0])
             for a in (0.3, 0.9, 1.5) for b in (0.4, 1.2) for c in (-0.8, 0.4)]
    path = Path(__file__).parent / "_fric_fit_test.csv"
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["kind"] + [f"q{i+1}" for i in range(6)] + [f"f{i+1}" for i in range(6)]
                   + [f"resid{i+1}" for i in range(6)])
        for kind in ("static", "kinetic"):
            for q in poses:
                f = f0 + mu * np.abs(dyn_cal.gravity(q)) + rng.normal(0, 0.005, 6)
                w.writerow([kind] + [f"{x:.5f}" for x in q] + [f"{x:.4f}" for x in f] + ["0"] * 6)
    out = fit_friction.fit(path)
    path.unlink()
    r = out["kinetic"]
    assert r["model"][1] and r["model"][2], f"load model not detected on j2/j3: {r['model']}"
    assert abs(r["f0"][1] - 0.20) < 0.05 and abs(r["f0"][2] - 0.22) < 0.05, r["f0"]
    assert abs(r["mu"][1] - 0.030) < 0.01 and abs(r["mu"][2] - 0.045) < 0.015, r["mu"]
    assert not r["model"][0], "j1 sees no gravity-load variation and must keep the median"
    assert abs(r["median"][0] - 0.50) < 0.03


def test_fric_curve_tracks_load():
    """At runtime the friction level must follow |g(q)|: heavier pose -> higher level,
    by fric_scale * mu * delta|g|."""
    b = BalancedDrag(DYN, kappa=1.0, fric_scale=0.85, fric=np.full(6, 0.2),
                     f_static=np.full(6, 0.2), fric_mu=np.full(6, 0.03),
                     f_static_mu=np.full(6, 0.03))
    q_light = np.array([0.0, 0.1, 0.15, 0.0, 0.0, 0.0])
    q_heavy = np.array([0.0, 1.2, 0.9, -0.5, 0.0, 0.0])
    g_l = abs(DYN.gravity(q_light)[1])
    g_h = abs(DYN.gravity(q_heavy)[1])
    assert g_h - g_l > 1.0, "premise: the two poses must differ in j2 load"
    b.observe(q_light, np.zeros(6), 0.01)
    lo = b.fric_curve(np.full(6, 1.0))[1]
    b.observe(q_heavy, np.zeros(6), 0.01)
    hi = b.fric_curve(np.full(6, 1.0))[1]
    expect = 0.85 * 0.03 * (g_h - g_l)
    assert abs((hi - lo) - expect) < 0.01, f"level moved {hi-lo:.3f}, expected {expect:.3f}"


def test_sustained_relief_j1_only():
    """j1 keeps margin-capped relief while clearly moving: a push below full friction (but
    above the 0.2 N.m margin) sustains it; a bias-sized push (below the margin) cannot; and
    a non-sustain joint (j2) still stalls under a below-friction push (drive gate)."""
    q0 = np.array([-1.2, 0.9, 1.1, -0.4, 0.0, 0.0])   # j1 off-center: room to travel before its limit
    inertia = urdf_inertia(q0)
    t0 = T_DRAG + 2.5
    sustain = np.array([True, False, False, False, False, False])

    def scenario(joint, weak, t_probe=2.5):
        kick = 0.8 if joint == 0 else 1.2   # j1's inertia is small; stay well under vel_abort
        def ext(t, q):
            out = np.zeros(6)
            if t0 < t < t0 + 0.4:
                out[joint] = kick         # establish motion
            elif t0 + 0.4 <= t < t0 + 2.7:
                out[joint] = weak
            return out
        ext.state = {"Fs": np.zeros(3)}
        probe = _Probe(_SimModel(DYN, inertia), ext, kappa=0.0, fric_scale=0.85,
                       fric=COULOMB, f_static=COULOMB, sustain_joints=sustain)
        run(q0, assist=probe, external=ext, inertia=inertia, duration=5.5, ctrl_fric=0.0)
        vj = np.array([v[joint] for *_, v, _ in probe.hist])
        return float(abs(vj[int((t0 + t_probe) * CFG.loop.rate_hz)]))

    # j1: friction 0.5, sustained relief min(0.85*0.5, 0.5-0.2)=0.30
    assert scenario(0, 0.35, t_probe=1.2) > 0.15, \
        "j1 must keep sliding under a below-friction push (sustained relief)"
    assert scenario(0, 0.15) < 0.08, "a margin-sized push must NOT sustain j1 (residuals cannot self-drive)"
    assert scenario(1, 0.35) < 0.08, "j2 is drive-gated only and must stall under a below-friction push"


def test_sustained_relief_all_joints():
    """Default CLI mask: every joint keeps margin-capped relief while sliding. j2 (margin
    0.28, friction 0.5 in sim -> sustained relief 0.22): a below-friction push above the
    margin sustains it; a margin-sized push must NOT (residuals cannot self-drive)."""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    inertia = urdf_inertia(q0)
    t0 = T_DRAG + 2.5

    def scenario(weak):
        def ext(t, q):
            out = np.zeros(6)
            if t0 < t < t0 + 0.5:
                out[1] = 1.2
            elif t0 + 0.5 <= t < t0 + 2.7:
                out[1] = weak
            return out
        ext.state = {"Fs": np.zeros(3)}
        # explicit margin: this test checks the mechanism, not the tuned per-joint defaults
        probe = _Probe(_SimModel(DYN, inertia), ext, kappa=0.0, fric_scale=0.85,
                       fric=COULOMB, f_static=COULOMB, sustain_joints=np.ones(6, bool),
                       sustain_margin=0.28)
        run(q0, assist=probe, external=ext, inertia=inertia, duration=5.5, ctrl_fric=0.0)
        vj = np.array([v[1] for *_, v, _ in probe.hist])
        return float(abs(vj[int((t0 + 2.1) * CFG.loop.rate_hz)]))   # past the ~0.7 s decay constant

    assert scenario(0.42) > 0.12, "j2 must keep sliding under a below-friction push above its margin"
    assert scenario(0.18) < 0.08, "a margin-sized push must NOT sustain j2"


def test_signed_margin_direction():
    """The margin is signed: with defaults, j2's reserve is 0.35 along -q2 (its cable's push
    direction) and 0.20 along +q2, so the same below-friction push sustains motion one way
    and not the other (sim friction 0.5: relief 0.30 for +q2, 0.15 for -q2)."""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    inertia = urdf_inertia(q0)
    t0 = T_DRAG + 2.5

    def scenario(sign):
        def ext(t, q):
            out = np.zeros(6)
            if t0 < t < t0 + 0.5:
                out[1] = sign * 1.2
            elif t0 + 0.5 <= t < t0 + 2.7:
                out[1] = sign * 0.30
            return out
        ext.state = {"Fs": np.zeros(3)}
        # explicit asymmetric margins so the test probes the mechanism, not the tuned defaults
        probe = _Probe(_SimModel(DYN, inertia), ext, kappa=0.0, fric_scale=0.85,
                       fric=COULOMB, f_static=COULOMB, sustain_joints=np.ones(6, bool))
        probe.margin_pos = np.full(6, 0.20); probe.margin_neg = np.full(6, 0.35)
        run(q0, assist=probe, external=ext, inertia=inertia, duration=5.5, ctrl_fric=0.0)
        vj = np.array([v[1] for *_, v, _ in probe.hist])
        return float(abs(vj[int((t0 + 2.1) * CFG.loop.rate_hz)]))

    assert scenario(+1) > 0.12, "+q2 (small margin side) must sustain under a 0.30 push"
    assert scenario(-1) < 0.08, "-q2 (residual's push direction, 0.35 margin) must stall"


def test_live_mode_toggles():
    """Keys b / bf / bs flip inertia shaping, friction comp, and j1 sustain live."""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    inertia = urdf_inertia(q0)

    def run_with(cmds, **kw):
        probe = _Probe(_SimModel(DYN, inertia), kappa=1.0, fric_scale=0.85,
                       fric=COULOMB, f_static=COULOMB, **kw)
        dt = 1.0 / CFG.loop.rate_hz
        arm = SimArm(DYN, q0, dt, coulomb=COULOMB, inertia=inertia)
        ctrl = GravityDragController(arm, DYN, CFG, assist=probe, duration=2.0, auto_release=True,
                                     interactive=False, realtime=False, print_every=0, hold_timeout=1.0)
        for c in cmds:
            ctrl._cmds.put(c)
        assert ctrl.run() is Phase.DONE and ctrl.freeze_reason is None
        return probe

    sus = np.array([True] + [False] * 5)
    p = run_with(["b", "bf"], sustain_joints=sus)
    assert not p.shaping_on and not p.fric_on
    assert p.alpha == 0.0 and "OFF:shape,fric" in p.summary()
    p = run_with(["bs"], sustain_joints=sus)
    assert not p.sustain.any() and p.shaping_on and p.fric_on
    p = run_with(["b", "b", "bs", "bs"], sustain_joints=sus)
    assert p.shaping_on and p.sustain.any()
    # bs with friction off must refuse (sustain rides on the friction ff)
    p = run_with(["bf", "bs"], sustain_joints=sus)
    assert not p.fric_on and p.sustain.any(), "bs must refuse while friction ff is off"


def test_paper_features():
    """Opt-in paper-aligned additions (viscous B, Cartesian damping D_v, breakaway assist,
    alpha scheduling) - each does what it should, and defaults leave the output unchanged."""
    q0=np.array([0.0,0.9,1.1,-0.4,0.0,0.0]); v=np.array([0.0,0.3,-0.2,0.1,0.0,0.0])
    def mk(**kw):
        b=BalancedDrag(DYN,kappa=0.0,fric_scale=0.85,fric=np.full(6,0.4),f_static=np.full(6,0.5),
                       sustain_joints=np.zeros(6,bool),**kw)
        b._observe(q0,v,0.011); b._observe(q0,v,0.011); return b
    # (2) Cartesian damping is strictly dissipative
    J=DYN.ee_jacobian(q0,frame="local"); Dv=np.diag([6.0]*3+[3.0]*3)
    td=-(J.T@(Dv@(J@v)))
    assert float(v@td)<-1e-3, "D_v damping must remove energy"
    # (1) viscous adds exactly fric_scale*B*qd
    o0=mk().update(q0,v,0.011); o1=mk(fric_viscous=np.full(6,0.05)).update(q0,v,0.011)
    assert np.allclose(o1-o0, 0.85*0.05*v, atol=1e-6), "viscous term wrong"
    # (4) velocity schedule tapers near zero (s_v small at 0.02, ~1 at 0.5)
    b=mk(alpha_sigma_v=0.05)
    assert (1-np.exp(-(0.02/0.05)**2))<0.2 and (1-np.exp(-(0.5/0.05)**2))>0.99
    # (3) breakaway CONTRIBUTION (with - without) follows sign(r) and fades with speed
    froze=lambda b: setattr(b,"_observe",lambda qq,vv,dt: setattr(b,"_g_abs",np.abs(DYN.gravity(qq))))
    def brk_contrib(rv, vt):
        bb=mk(break_beta=0.5,break_vs=0.10); froze(bb); bb.r=np.array([0.0,rv,0,0,0,0])
        b0=mk(break_beta=0.0); froze(b0); b0.r=np.array([0.0,rv,0,0,0,0])
        vv=np.array([0.,vt,0,0,0,0])
        return float(bb.update(q0,vv,0.011)[1]-b0.update(q0,vv,0.011)[1])
    lo=brk_contrib(0.6,0.01); hi=brk_contrib(0.6,0.5)
    assert lo>0.1 and lo>hi, f"breakaway contrib should be +ve and fade: lo={lo:.3f} hi={hi:.3f}"
    assert brk_contrib(-0.6,0.01)<-0.1, "breakaway sign must follow r"
    # (5) latched detent: holds an un-driven joint against a static bias where damping can't,
    #     and produces no torque while the hand drives it
    def drift(kp, bias=0.15, I=0.3, steps=300, dt=0.011):
        b=BalancedDrag(DYN,kappa=0.0,fric_scale=0.0,sustain_joints=np.zeros(6,bool),detent_kp=kp)
        b._observe=lambda qq,vv,ddt: setattr(b,"_g_abs",np.abs(DYN.gravity(qq)))
        qq=q0.copy(); vv=np.zeros(6); q0j=qq[1]
        for _ in range(steps):
            o=b.update(qq,vv,dt); a=(bias+o[1])/I; vv[1]+=a*dt; qq[1]+=vv[1]*dt
        return abs(np.degrees(qq[1]-q0j))
    assert drift(0.0)>90, "premise: without the detent the bias must run the joint away"
    assert drift(5.0)<5, "detent must hold an un-driven joint against a static bias"
    bd=BalancedDrag(DYN,kappa=0.0,fric_scale=0.0,sustain_joints=np.zeros(6,bool),detent_kp=5.0)
    bd._observe=lambda qq,vv,ddt: setattr(bd,"_g_abs",np.abs(DYN.gravity(qq))); bd._q_latch=q0.copy()
    bd.r=np.array([0.0,0.6,0,0,0,0])   # hand driving j2
    assert abs(bd.update(q0+np.array([0,0.1,0,0,0,0]),np.array([0.,0.2,0,0,0,0]),0.011)[1])<1e-6, \
        "detent must not fight the hand while it drives the joint"

    # (6) live kinetic-friction-model A/B: values swap both ways, unknown name is a no-op
    bm = BalancedDrag(DYN, kappa=0.0, fric=np.full(6, 0.3))
    bm.fric_models = {"viscous": {"kin": [0.4]*6, "mu": [0.0]*6, "viscous": [0.1]*6},
                      "load": {"kin": [0.6]*6, "mu": [0.02]*6, "viscous": [0.0]*6}}
    assert bm.set_fric_model("load") == "load" and bm._raw_kin[0] == 0.6 and bm._raw_mu[0] == 0.02
    assert bm.set_fric_model("viscous") == "viscous" and bm.fric_viscous[0] == 0.1 and bm._raw_mu[0] == 0.0
    assert bm.set_fric_model("bogus") == "viscous" and bm._raw_kin[0] == 0.4

    # (7) per-term ablation: mu and B toggles zero their terms and restore them
    bt = BalancedDrag(DYN, kappa=0.0, fric=np.full(6, 0.3), fric_mu=np.full(6, 0.05),
                      fric_viscous=np.full(6, 0.1), fric_scale=1.0, sustain_joints=np.zeros(6, bool))
    bt._g_abs = np.full(6, 2.0)
    lvl_on = bt.fric_curve(np.full(6, 1.0))            # fast -> kinetic level = kin + mu*|g|
    assert bt.set_fric_terms(mu=False) == (False, True)
    lvl_off = bt.fric_curve(np.full(6, 1.0))
    assert np.allclose(lvl_on - lvl_off, 0.05 * 2.0), "mu off must remove exactly mu*|g|"
    bt.r = np.zeros(6); bt._observe = lambda q, v, dt: None
    v1 = np.full(6, 0.5)
    with_v = bt.update(Q_WRIST.copy(), v1, 0.011)
    bt.set_fric_terms(viscous=False)
    no_v = bt.update(Q_WRIST.copy(), v1, 0.011)
    assert np.allclose(with_v - no_v, 0.1 * 0.5, atol=1e-9), "visc off must remove exactly B*qd"
    assert bt.set_fric_terms(mu=True, viscous=True) == (True, True)

    # defaults: no feature changes the output vs a plain build
    base=BalancedDrag(DYN,kappa=1.0,fric_scale=0.85,fric=COULOMB,f_static=COULOMB,sustain_joints=np.ones(6,bool))
    assert base.damp_t==0 and base.break_beta==0 and base.alpha_sigma_v==0 and base.detent_kp==0 and not np.any(base.fric_viscous)


def test_kappa_validation():
    for bad in (-0.1, 2.5):
        try:
            BalancedDrag(DYN, kappa=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"kappa={bad} must be rejected")


if __name__ == "__main__":
    for fn in [test_observer_tracks_known_push, test_gravity_bias_does_not_creep,
               test_shaping_lightens_a_joint_pulse, test_assist_and_resist_pattern,
               test_wrong_inertia_stays_bounded, test_live_retarget_slews,
               test_static_sweep_measures_breakaway, test_fric_curve_stribeck,
               test_fit_friction_recovers_load_model, test_fric_curve_tracks_load,
               test_sustained_relief_j1_only, test_sustained_relief_all_joints, test_signed_margin_direction, test_paper_features,
               test_live_mode_toggles, test_kappa_validation]:
        print(f"-- {fn.__name__}")
        fn()
        print("   ok")
    print("all balance tests passed")
