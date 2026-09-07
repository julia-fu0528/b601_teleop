#!/usr/bin/env python3
"""Multi-speed, current-based friction ID (the 'fv' sweep + scripts/fit_friction.py eq 25).

Two levels:
  1. test_fit_recovers_tauc_mu_and_B_synthetic - the authoritative math check: from clean +/-v
     data with per-pose even offsets and outliers, the combined fit recovers tau_c (de-biased),
     the load slope mu (only where the poses give |g| leverage), and viscous B.
  2. test_vfric_sweep_pipeline_sim           - end to end in the simulator: the 'fv' sweep writes
     the CSV and the fit recovers the Coulomb intercept tau_c. (B is NOT asserted in sim: the
     simulator's torque feedback is its *commanded* torque reconstructed from the PD loop, whose
     kd term corrupts the apparent viscous slope on heavy joints - an artifact of the sim having no
     real current sensor. On hardware `torq` is a direct current measurement, so B is meaningful
     there; its recovery is proven by test 1.)
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import fit_friction as ff  # noqa: E402
from b601 import GravityDragController, Phase  # noqa: E402
from b601.sim import SimArm  # noqa: E402
from test_gravity_drag import CFG, make_dyn  # noqa: E402

_TMP = Path(os.environ.get("TMPDIR", "/tmp")) / "b601_vfric_test"


def _fit_dyn():
    """The same dynamics fit_v uses internally (config g_scale/g_bias applied)."""
    from b601 import ArmDynamics, load_config
    cfg = load_config(ROOT / "config" / "b601_rs.toml")
    return ArmDynamics(cfg.urdf, cfg.joint_names, cfg.lock_joints,
                       [j.g_scale for j in cfg.joints], [j.g_bias for j in cfg.joints])


def test_fit_recovers_tauc_mu_and_B_synthetic():
    """The combined eq-25 fit recovers tau_c (de-biased), the load slope mu, and viscous B,
    and does NOT invent mu where the data carries no load signal."""
    _TMP.mkdir(parents=True, exist_ok=True)
    path = _TMP / "synth.csv"
    n = 6
    tau_c = np.array([0.53, 0.18, 0.52, 0.13, 0.23, 0.21])
    mu = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])              # load slope on j2 only
    B = np.array([0.04, 0.00, 0.06, 0.02, 0.03, 0.00])
    eps = 0.02
    speeds = [0.05, 0.10, 0.20, 0.40]
    poses = [np.array([0.0, 0.3, 0.3, 0.0, 0.0, 0.0]),          # j2 |g| spans several N.m
             np.array([0.0, 0.9, 1.0, -0.5, 0.0, 0.0]),
             np.array([0.0, 1.4, 1.6, -1.0, 0.0, 0.0]),
             np.array([0.0, 1.9, 1.8, -0.2, 0.0, 0.0])]
    dyn = _fit_dyn()
    G = np.array([np.abs(dyn.gravity(q)) for q in poses])
    assert G[:, 1].max() - G[:, 1].min() >= 1.0, "test poses must give j2 load leverage"
    rng = np.random.RandomState(0)
    offs = [rng.normal(0, 0.2, n) for _ in poses]               # even per-pose offsets (must cancel)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vset"] + [f"q{i+1}" for i in range(n)]
                   + [f"mv{i+1}" for i in range(n)] + [f"f{i+1}" for i in range(n)])
        for pi, q in enumerate(poses):
            for spd in speeds:
                for sgn in (+1, -1):
                    v = sgn * spd
                    fric = (tau_c + mu * G[pi]) * np.tanh(v / eps) + B * v
                    f = fric + offs[pi] + rng.normal(0, 0.01, n)
                    if pi == 0 and spd == speeds[1] and sgn > 0:
                        f[2] += 3.0                              # outlier for the robust fit
                    w.writerow([f"{v:.5f}"] + [f"{x:.5f}" for x in q]
                               + [f"{v:.5f}"] * n + [f"{x:.4f}" for x in f])

    res = ff.fit_v(str(path), eps=eps)
    assert res["n_pairs"] == len(speeds) * len(poses)
    assert np.allclose(res["tau_c"], tau_c, atol=0.06), f"tau_c: {res['tau_c']}"
    assert res["mu_kept"][1] and abs(res["mu"][1] - 0.03) < 0.012, f"j2 mu: {res['mu'][1]}"
    assert np.allclose(res["B"], B, atol=0.03), f"B: {res['B']}"
    # no invented mu on joints without a load signal
    assert not any(res["mu_kept"][j] and res["mu"][j] > 0.015 for j in (0, 4, 5)), f"mu: {res['mu']}"
    path.unlink()


def test_vfric_sweep_pipeline_sim():
    """End to end: the 'fv' sweep runs, writes the CSV, and the fit recovers the Coulomb tau_c."""
    _TMP.mkdir(parents=True, exist_ok=True)
    path = _TMP / "sim.csv"
    if path.exists():
        path.unlink()
    dyn = make_dyn()
    dt = 1.0 / CFG.loop.rate_hz
    arm = SimArm(dyn, np.array([0.0, 0.7, 1.1, 0.0, 0.0, 0.0]), dt, coulomb=0.3, viscous=0.05)
    ctrl = GravityDragController(arm, dyn, CFG, interactive=False, realtime=False, print_every=0,
                                 duration=80.0, auto_release=True, hold_timeout=1.0,
                                 vfric_path=str(path), vfric_speeds=(0.1, 0.2, 0.3, 0.4),
                                 vfric_period=3.0, vfric_periods=2)
    fired = {"go": False}
    orig_say = ctrl._say
    def say(msg):
        if "-> DRAG" in msg and not fired["go"]:
            fired["go"] = True
            ctrl._cmds.put("fv")
    ctrl._say = say
    phase = ctrl.run()
    assert phase is Phase.DONE
    assert path.exists(), "fv sweep wrote no CSV"

    res = ff.fit_v(str(path), eps=0.02)
    assert res["n_pairs"] == 4, f"expected 4 speed pairs, got {res['n_pairs']}"
    # Coulomb intercept is recovered on the loaded, cleanly-moving joints (this is what sim can prove)
    for j in (1, 2):
        assert abs(res["tau_c"][j] - 0.3) < 0.08, f"joint{j+1} tau_c off: {res['tau_c'][j]}"
    # B must at least come out finite and non-negative everywhere (no garbage / no false negative)
    assert np.all(np.isfinite(res["B"])) and np.all(res["B"] >= 0.0), f"B: {res['B']}"
    path.unlink()


if __name__ == "__main__":
    test_fit_recovers_tauc_mu_and_B_synthetic()
    print("test_fit_recovers_tauc_mu_and_B_synthetic ok")
    test_vfric_sweep_pipeline_sim()
    print("test_vfric_sweep_pipeline_sim ok")
    print("all vfric tests passed")
