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
    # per-direction values (paper eq 28) recorded alongside the symmetric average
    sp, sn = ctrl.static_pos, ctrl.static_neg
    assert np.isfinite(sp).all() and np.isfinite(sn).all(), f"pos/neg incomplete: {sp} {sn}"
    assert np.allclose(0.5 * (sp + sn), meas, atol=1e-9), "average must equal (tau+ + tau-)/2"
    # the sweep must land in the friction CSV for multi-pose aggregation: 3 rows (avg, pos, neg)
    import csv as _csv
    with open(fcsv, newline="") as fh:
        rows = list(_csv.DictReader(fh))
    fcsv.unlink()
    by_kind = {r["kind"]: r for r in rows}
    assert set(by_kind) == {"static", "static_pos", "static_neg"}, f"kinds: {sorted(by_kind)}"
    logged = np.array([float(by_kind["static"][f"f{i+1}"]) for i in range(6)])
    assert np.allclose(logged, meas, atol=1e-3)
    logged_p = np.array([float(by_kind["static_pos"][f"f{i+1}"]) for i in range(6)])
    assert np.allclose(logged_p, sp, atol=1e-3)


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


def test_ungated_relief_sustains_motion():
    """Paper-style relief (no drive gate, no margins): once a joint is moving, a push well
    below full friction keeps it sliding, because the relief pays 85 % of the friction.
    (Under the old margin scheme a 0.2 N.m push was below j2's margin and had to stall.)"""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    inertia = urdf_inertia(q0)
    t0 = T_DRAG + 2.5

    def ext(t, q):
        out = np.zeros(6)
        if t0 < t < t0 + 0.3:
            out[1] = 0.6                    # establish motion
        elif t0 + 0.3 <= t < t0 + 2.0:
            out[1] = 0.2                    # far below friction 0.5, above the residual 0.85*0.5 leaves
        return out
    ext.state = {"Fs": np.zeros(3)}
    probe = _Probe(_SimModel(DYN, inertia), ext, kappa=0.0, fric_scale=0.85,
                   fric=COULOMB, f_static=COULOMB)
    run(q0, assist=probe, external=ext, inertia=inertia, duration=5.5, ctrl_fric=0.0)
    vj = np.array([v[1] for *_, v, _ in probe.hist])
    v_late = float(abs(vj[int((t0 + 1.4) * CFG.loop.rate_hz)]))
    assert v_late > 0.12, f"a below-friction push must keep j2 sliding (ungated relief), got {v_late:.3f}"


def test_passivity_damping_bounds_over_relief():
    """Paper eq 37: with the gate and margins gone, D_v is the passivity guard. Calibrate the
    relief WELL ABOVE the sim's true friction (0.85*0.75 = 0.64 vs 0.5, Delta ~ 0.14 - the
    sim's sharp friction knee masks smaller over-relief, so this probes the saturated regime): after a kick with the hand off,
    the over-relief alone would keep the joint coasting; with D_v (+ kd_drag) it must decay.
    The tanh knee bounds the injected power at Delta*v^2/v0, so damping >= Delta/v0 wins."""
    q0 = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    inertia = urdf_inertia(q0)
    t0 = T_DRAG + 2.5

    def coast_speed(**damp):
        def ext(t, q):
            out = np.zeros(6)
            if t0 < t < t0 + 0.4:
                out[1] = 0.8                # kick, then hands off entirely
            return out
        ext.state = {"Fs": np.zeros(3)}
        probe = _Probe(_SimModel(DYN, inertia), ext, kappa=0.0, fric_scale=0.85,
                       fric=np.full(6, 0.75), f_static=np.full(6, 0.75), **damp)
        run(q0, assist=probe, external=ext, inertia=inertia, duration=6.0, ctrl_fric=0.0)
        vj = np.array([v[1] for *_, v, _ in probe.hist])
        return float(abs(vj[int((t0 + 2.4) * CFG.loop.rate_hz)]))   # 2 s after the hand left

    v_undamped = coast_speed()
    v_damped = coast_speed(damp_t=4.0, damp_r=0.6)
    assert v_undamped > 0.06,         f"premise: over-calibrated relief must keep the joint coasting without D_v (got {v_undamped:.3f})"
    assert v_damped < 0.5 * v_undamped and v_damped < 0.05,         f"D_v must dissipate the over-relief (damped {v_damped:.3f} vs undamped {v_undamped:.3f})"


def test_live_mode_toggles():
    """Keys b / bf flip inertia shaping and friction comp live."""
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

    p = run_with(["b", "bf"])
    assert not p.shaping_on and not p.fric_on
    assert p.alpha == 0.0 and "OFF:shape,fric" in p.summary()
    p = run_with(["b", "b"])
    assert p.shaping_on and p.fric_on


def test_paper_features():
    """Opt-in paper-aligned additions (viscous B, Cartesian damping D_v, breakaway assist,
    alpha scheduling) - each does what it should, and defaults leave the output unchanged."""
    q0=np.array([0.0,0.9,1.1,-0.4,0.0,0.0]); v=np.array([0.0,0.3,-0.2,0.1,0.0,0.0])
    def mk(**kw):
        b=BalancedDrag(DYN,kappa=0.0,fric_scale=0.85,fric=np.full(6,0.4),f_static=np.full(6,0.5),
                       **kw)
        b._observe(q0,v,0.011); b._observe(q0,v,0.011); return b
    # (2) Cartesian damping is strictly dissipative
    J=DYN.ee_jacobian(q0,frame="local"); Dv=np.diag([6.0]*3+[3.0]*3)
    td=-(J.T@(Dv@(J@v)))
    assert float(v@td)<-1e-3, "D_v damping must remove energy"
    # (1) viscous adds fric_scale*B*qd SATURATED at the identification range (visc_vsat)
    o0=mk().update(q0,v,0.011); bv=mk(fric_viscous=np.full(6,0.05)); o1=bv.update(q0,v,0.011)
    exp_v = 0.85*0.05*bv.visc_vsat*np.tanh(v/bv.visc_vsat)
    assert np.allclose(o1-o0, exp_v, atol=1e-6), "viscous term wrong"
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
        b=BalancedDrag(DYN,kappa=0.0,fric_scale=0.0,detent_kp=kp)
        b._observe=lambda qq,vv,ddt: setattr(b,"_g_abs",np.abs(DYN.gravity(qq)))
        qq=q0.copy(); vv=np.zeros(6); q0j=qq[1]
        for _ in range(steps):
            o=b.update(qq,vv,dt); a=(bias+o[1])/I; vv[1]+=a*dt; qq[1]+=vv[1]*dt
        return abs(np.degrees(qq[1]-q0j))
    assert drift(0.0)>90, "premise: without the detent the bias must run the joint away"
    assert drift(5.0)<5, "detent must hold an un-driven joint against a static bias"
    bd=BalancedDrag(DYN,kappa=0.0,fric_scale=0.0,detent_kp=5.0)
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
                      fric_viscous=np.full(6, 0.1), fric_scale=1.0)
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
    exp = 0.1 * bt.visc_vsat * np.tanh(0.5 / bt.visc_vsat)   # saturated B comp (identification range)
    assert np.allclose(with_v - no_v, exp, atol=1e-9), "visc off must remove exactly the saturated B term"
    assert bt.set_fric_terms(mu=True, viscous=True) == (True, True)
    # the B comp is anti-damping and must stay BOUNDED at speed: at 4 rad/s it may inject at most
    # ~B*vsat (~0.03 N.m), never the linear 0.4 N.m that ran j6 away to vel_abort on hardware
    v4 = np.full(6, 4.0)
    with4 = bt.update(Q_WRIST.copy(), v4, 0.011)
    bt.set_fric_terms(viscous=False)
    no4 = bt.update(Q_WRIST.copy(), v4, 0.011)
    bt.set_fric_terms(viscous=True)
    assert np.all(np.abs(with4 - no4) < 0.05), f"B comp must saturate at speed: {np.round(with4 - no4, 3)}"

    # (8) direction-dependent breakaway (paper eq 28): fric_curve near rest and the breakaway
    # assist must select tau+ or tau- by direction, and fall back to symmetric when not given
    ba = BalancedDrag(DYN, kappa=0.0, fric=np.full(6, 0.3), f_static=np.full(6, 0.4),
                      f_static_pos=np.full(6, 0.6), f_static_neg=np.full(6, 0.2),
                      fric_scale=1.0)
    ba._g_abs = np.zeros(6)
    fp = ba.fric_curve(np.full(6, +1e-4))            # ~rest, moving + -> static_pos level
    fn = ba.fric_curve(np.full(6, -1e-4))
    assert np.allclose(fp, 0.6, atol=0.01) and np.allclose(fn, 0.2, atol=0.01), (fp, fn)
    sym = BalancedDrag(DYN, kappa=0.0, fric=np.full(6, 0.3), f_static=np.full(6, 0.4))
    assert np.allclose(sym._raw_static_pos, 0.4) and np.allclose(sym._raw_static_neg, 0.4)

    # (9) damping saturates: full-slope guard near rest, but the felt drag at guiding speed is
    # capped at d_eff*vsat - NOT proportional to velocity (that made the arm feel rigid on hardware)
    bs2 = BalancedDrag(DYN, kappa=0.0, fric_scale=0.0, damp_t=4.0, damp_r=0.6, damp_vsat=0.15)
    q0d = np.array([0.0, 0.9, 1.1, -0.4, 0.0, 0.0])
    slow = bs2.update(q0d, np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0]), 0.01)[1]
    fast = bs2.update(q0d, np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]), 0.01)[1]
    assert abs(slow) > 0.04, f"near rest the guard must act at full slope (got {slow:+.3f})"
    assert abs(fast) < 0.35, f"at 2 rad/s the drag must be saturated, not ~2.6 N.m linear (got {fast:+.3f})"
    assert abs(fast) < 3.0 * abs(slow) / (0.05 / 0.15), "drag must saturate, not grow linearly"

    # (10) hybrid soft intent gate: full paper relief when the hand drives, floor*relief when
    # not - the intent-vs-creep discrimination (creep lives below the observer dead-band).
    bg = BalancedDrag(DYN, kappa=0.0, fric=np.full(6, 0.4), fric_scale=1.0, gate_floor=0.5)
    bg._observe = lambda q, v, dt: setattr(bg, "_g_abs", np.zeros(6))
    vg = np.full(6, 0.5)
    bg.r = np.zeros(6)                                   # un-driven (creep): floor only
    creep_ff = bg.update(Q_WRIST.copy(), vg, 0.011)
    bg.r = np.full(6, 0.5)                               # clearly hand-driven: full relief
    drive_ff = bg.update(Q_WRIST.copy(), vg, 0.011)
    assert np.allclose(creep_ff, 0.5 * drive_ff, atol=1e-9), \
        f"un-driven relief must be floor*driven: {creep_ff[1]:.3f} vs {drive_ff[1]:.3f}"
    assert abs(drive_ff[1] - 0.4 * np.tanh(0.5 / bg.v0)) < 1e-6, "driven relief must be full paper"
    assert BalancedDrag(DYN, kappa=0.0).gate_floor == 1.0, "constructor default must stay pure paper"

    # defaults: no feature changes the output vs a plain build
    base=BalancedDrag(DYN,kappa=1.0,fric_scale=0.85,fric=COULOMB,f_static=COULOMB)
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
               test_ungated_relief_sustains_motion, test_passivity_damping_bounds_over_relief, test_paper_features,
               test_live_mode_toggles, test_kappa_validation]:
        print(f"-- {fn.__name__}")
        fn()
        print("   ok")
    print("all balance tests passed")
