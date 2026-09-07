#!/usr/bin/env python3
"""Fit the friction model from the 'f' (kinetic) and 's' (static) sweeps in friction.csv.

Each sweep appends one row per kind with the POSE it was measured at, so besides the
per-joint median this fits the load model that explains the pose-to-pose spread:

    f_j(q) = f0_j + mu_j * |g_j(q)|          (gear-mesh loss grows with transmitted torque)

|g_j(q)| is the calibrated gravity torque from the URDF - the same quantity the runtime
uses - so the fit needs no extra measurements. Per joint and kind the model is kept only
when it beats the median (>= 4 poses, >= 1 N.m of load range, mu > 0, fit rms < 0.8 x
median rms); otherwise it falls back to the median (mu = 0). Joints whose load never
varies (j1's axis is vertical) naturally keep the median.

    python scripts/fit_friction.py friction.csv

prints per joint: n, min/median/max, the fitted f0 / mu with rms before vs after, and a
paste-ready config block (fric_kinetic/-_static = f0, fric_kinetic_mu/-_static_mu = mu).
--balance-fric applies the 85 % at runtime; raw values go in the config.

MULTI-SPEED (current-based) mode: if the CSV has a 'v' column (from the 'fv' sweep, which reads the
motor torque feedback torq = K_t*iq and subtracts gravity, eq 22), this instead fits

    tau_f(v) = tau_c * tanh(v/eps) + B * v            (eq 25)

per joint by pairing +/-v at each pose (the even offset cancels, eq 23-24, so tau_c is de-biased) and
robust (Huber) regression. Prints tau_c and viscous B -> fric_kinetic / fric_viscous.

    python scripts/fit_friction.py friction_v.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from b601 import ArmDynamics, load_config  # noqa: E402


def load_rows(path):
    rows = defaultdict(list)   # kind -> list of (q[6], f[6])
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        n = sum(1 for k in rd.fieldnames if k.startswith("f") and k[1:].isdigit())
        for row in rd:
            q = np.array([float(row[f"q{i+1}"]) for i in range(n)])
            f = np.array([float(row[f"f{i+1}"]) for i in range(n)])
            rows[row["kind"]].append((q, f))
    return rows, n


# ---- velocity-friction (multi-speed, current-based) fit: tau_c + B --------------------------

def has_velocity_column(path) -> bool:
    with open(path, newline="") as fh:
        hdr = csv.reader(fh).__next__()
    return "vset" in hdr and "kind" not in hdr


def load_v_rows(path):
    """Rows from the 'fv' sweep: (vset_signed, q[n], mv[n], f[n]) where mv = measured per-joint
    velocity and f = torq - g (current-based). vset is only a pairing label."""
    out = []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        n = sum(1 for k in rd.fieldnames if k.startswith("f") and k[1:].isdigit())
        for row in rd:
            vset = float(row["vset"])
            q = np.array([float(row[f"q{i+1}"]) for i in range(n)])
            mv = np.array([float(row[f"mv{i+1}"]) for i in range(n)])
            f = np.array([float(row[f"f{i+1}"]) for i in range(n)])
            out.append((vset, q, mv, f))
    return out, n


def _pair_odd(rows, n):
    """Pair each +vset row with the -vset row at the SAME pose and speed setting. Per joint take the
    odd part of the friction, 0.5*(f(+) - f(-)) -- which cancels the even offset per pose (gravity
    residual / current bias), the paper's +/-v subtraction (eq 23-24), so tau_c comes out de-biased --
    against the MEASURED speed 0.5*(mv(+) - mv(-)) (immune to tracking lag).
    Returns v_eff[k, n] (>0 per joint), ODD[k, n], and the pose q[k, n] of each pair."""
    plus, minus = {}, {}
    for vset, q, mv, f in rows:
        key = (tuple(np.round(q, 4)), round(abs(vset), 5))
        (plus if vset > 0 else minus)[key] = (mv, f)
    veff, odd, qpose = [], [], []
    for key in sorted(set(plus) & set(minus), key=lambda k: k[1]):
        mvp, fp = plus[key]
        mvn, fn = minus[key]
        veff.append(0.5 * (mvp - mvn))       # per-joint measured speed magnitude
        odd.append(0.5 * (fp - fn))          # per-joint de-biased friction
        qpose.append(np.array(key[0]))
    z = np.zeros((0, n))
    return ((np.array(veff) if veff else z), (np.array(odd) if odd else z),
            (np.array(qpose) if qpose else z))


def _robust_fit(A, y):
    """Robust (Huber IRLS) linear fit y ~ A @ coef. Returns (coef, rms)."""
    w = np.ones(len(y))
    coef = np.zeros(A.shape[1])
    for _ in range(12):
        W = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
        res = y - A @ coef
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9   # robust sigma (MAD)
        delta = 1.345 * s
        a = np.abs(res)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9))
    rms = float(np.sqrt(np.mean((y - A @ coef) ** 2)))
    return coef, rms


def fit_v(csv_path, eps=0.02, config_path=None):
    """Per joint, from the multi-speed sweep: the COMBINED kinetic model

        odd(v, q) ~ (tau_c + mu * |g_j(q)|) * tanh(v/eps) + B * v

    i.e. de-biased Coulomb tau_c, load slope mu (needs poses spanning >= 1 N.m of |g_j|),
    and viscous B. mu is kept only when the load model clearly beats the flat one
    (>= 6 pairs, span >= 1, mu > 0, rms < 0.9x flat); otherwise mu = 0 (flat + B).
    Returns {'n_pairs','tau_c','mu','B','rms','rms_flat','vspan','gspan','mu_kept','eps'}."""
    cfg = load_config(config_path or ROOT / "config" / "b601_rs.toml")
    dyn = ArmDynamics(cfg.urdf, cfg.joint_names, cfg.lock_joints,
                      [j.g_scale for j in cfg.joints], [j.g_bias for j in cfg.joints])
    rows, n = load_v_rows(csv_path)
    VEFF, ODD, QP = _pair_odd(rows, n)
    res = {"n_pairs": len(VEFF), "tau_c": np.full(n, np.nan), "mu": np.zeros(n), "B": np.zeros(n),
           "rms": np.full(n, np.nan), "rms_flat": np.full(n, np.nan), "vspan": np.full(n, np.nan),
           "gspan": np.full(n, np.nan), "mu_kept": np.zeros(n, bool), "eps": float(eps)}
    if len(VEFF) == 0:
        return res
    GA = np.array([np.abs(dyn.gravity(q)) for q in QP])              # |g_j| at each pair's pose
    for j in range(n):
        vj, yj, gj = VEFF[:, j], ODD[:, j], GA[:, j]
        ok = np.isfinite(yj) & np.isfinite(vj) & (vj > 1e-3)
        if ok.sum() < 3:
            continue
        vj, yj, gj = vj[ok], yj[ok], gj[ok]
        res["vspan"][j] = float(vj.max() - vj.min())
        res["gspan"][j] = float(gj.max() - gj.min())
        t = np.tanh(vj / eps)
        (tc2, B2), rms2 = _robust_fit(np.c_[t, vj], yj)              # flat: tau_c + B*v
        res["tau_c"][j], res["B"][j], res["rms"][j] = float(tc2), max(float(B2), 0.0), rms2
        res["rms_flat"][j] = rms2
        if ok.sum() >= 6 and res["gspan"][j] >= 1.0:                 # + load slope mu*|g|
            (tc3, mu3, B3), rms3 = _robust_fit(np.c_[t, gj * t, vj], yj)
            if mu3 > 0.0 and tc3 >= 0.0 and rms3 < 0.9 * rms2:
                res["tau_c"][j], res["mu"][j] = float(tc3), float(mu3)
                res["B"][j], res["rms"][j] = max(float(B3), 0.0), rms3
                res["mu_kept"][j] = True
    return res


def print_v(res, n):
    print(f"[velocity-friction] {res['n_pairs']} +/-v pair(s), tanh eps = {res['eps']:.3f} rad/s "
          f"- combined model tau_c + mu*|g(q)| + B*qd")
    print("  joint    tau_c    mu       B(visc)  |g|span  rms     rms(flat)  note")
    for j in range(n):
        tc, mu, B = res["tau_c"][j], res["mu"][j], res["B"][j]
        if not np.isfinite(tc):
            print(f"  joint{j+1}     --       --       --        --       --       --       too few speeds")
            continue
        if res["mu_kept"][j]:
            note = "load model kept"
        elif np.isfinite(res["gspan"][j]) and res["gspan"][j] < 1.0:
            note = "mu unfittable (|g| span < 1 - sweep more spread poses)"
        else:
            note = "flat model good enough"
        if B <= 1e-3:
            note += "; B~0"
        print(f"  joint{j+1}   {tc:6.3f}   {mu:6.3f}   {B:7.4f}   {res['gspan'][j]:6.2f}   "
              f"{res['rms'][j]:6.3f}   {res['rms_flat'][j]:6.3f}    {note}")
    print("\npaste into config/b601_rs.toml under each [[joint]] (raw values; --balance-fric applies the 85 %):")
    for j in range(n):
        tc, mu, B = res["tau_c"][j], res["mu"][j], res["B"][j]
        if not np.isfinite(tc):
            continue
        line = f"  joint{j+1}:  fric_kinetic = {tc:.2f}"
        if res["mu_kept"][j]:
            line += f"   fric_kinetic_mu = {mu:.3f}"
        if B > 1e-3:
            line += f"   fric_viscous = {B:.4f}"
        print(line)
    print("  (tau_c is the de-biased Coulomb intercept; mu grows it with transmitted load |g_j(q)|; "
          "B is per-joint viscous. More spread poses -> tighter mu.)")


def fit(csv_path, config_path=None):
    """Returns {kind: {"n", "f0", "mu", "median", "rms_med", "rms_fit", "model"(bool)}} per joint."""
    cfg = load_config(config_path or ROOT / "config" / "b601_rs.toml")
    dyn = ArmDynamics(cfg.urdf, cfg.joint_names, cfg.lock_joints,
                      [j.g_scale for j in cfg.joints], [j.g_bias for j in cfg.joints])
    rows, n = load_rows(csv_path)
    out = {}
    for kind, data in rows.items():
        G = np.array([np.abs(dyn.gravity(q)) for q, _ in data])      # (poses, n)
        F = np.array([f for _, f in data])
        res = {"n": len(data), "f0": np.full(n, np.nan), "mu": np.zeros(n),
               "median": np.nanmedian(F, axis=0), "rms_med": np.full(n, np.nan),
               "rms_fit": np.full(n, np.nan), "gspan": np.full(n, np.nan),
               "model": np.zeros(n, bool)}
        for j in range(n):
            ok = np.isfinite(F[:, j])
            g, f = G[ok, j], F[ok, j]
            if len(f) == 0:
                continue
            med = float(np.median(f))
            res["median"][j] = med
            res["rms_med"][j] = float(np.sqrt(np.mean((f - med) ** 2)))
            res["gspan"][j] = float(g.max() - g.min()) if len(g) else 0.0
            res["f0"][j], res["mu"][j] = med, 0.0
            if len(f) >= 4 and res["gspan"][j] >= 1.0:
                A = np.c_[np.ones_like(g), g]
                (f0, mu), *_ = np.linalg.lstsq(A, f, rcond=None)
                rms_fit = float(np.sqrt(np.mean((A @ [f0, mu] - f) ** 2)))
                if mu > 0.0 and f0 >= 0.02 and rms_fit < 0.8 * res["rms_med"][j]:
                    res["f0"][j], res["mu"][j] = float(f0), float(mu)
                    res["rms_fit"][j] = rms_fit
                    res["model"][j] = True
        out[kind] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="friction.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--eps", type=float, default=0.02, help="tanh knee (rad/s) for the velocity-friction fit")
    args = ap.parse_args()

    # multi-speed current-based sweep (has a 'v' column) -> tau_c + viscous B
    if Path(args.csv).exists() and has_velocity_column(args.csv):
        rows, n = load_v_rows(args.csv)
        if not rows:
            raise SystemExit(f"{args.csv}: no velocity-friction rows (run the 'fv' sweep while dragging)")
        print_v(fit_v(args.csv, eps=args.eps, config_path=args.config), n)
        return

    out = fit(args.csv, args.config)
    if not out:
        raise SystemExit(f"{args.csv}: no sweeps recorded (run the 'f' and 's' keys while dragging)")

    for kind in ("kinetic", "static"):
        if kind not in out:
            print(f"[{kind}] no sweeps yet (key '{'f' if kind == 'kinetic' else 's'}')")
            continue
        r = out[kind]
        print(f"[{kind}] {r['n']} pose(s)")
        print("  joint    median   f0      mu      |g| span   rms(med)  rms(fit)")
        for j in range(len(r["median"])):
            if r["model"][j]:
                print(f"  joint{j+1}   {r['median'][j]:6.3f}  {r['f0'][j]:6.3f}  {r['mu'][j]:6.3f}  "
                      f"{r['gspan'][j]:7.2f}   {r['rms_med'][j]:7.3f}  {r['rms_fit'][j]:7.3f}   <- load model")
            else:
                why = "load never varies" if (np.isfinite(r["gspan"][j]) and r["gspan"][j] < 1.0) \
                    else "median good enough / too few poses"
                print(f"  joint{j+1}   {r['median'][j]:6.3f}  {r['f0'][j]:6.3f}  {r['mu'][j]:6.3f}  "
                      f"{r['gspan'][j]:7.2f}   {r['rms_med'][j]:7.3f}       --    ({why})")

    print("\npaste into config/b601_rs.toml under each [[joint]] (raw values; --balance-fric applies the 85 %):")
    kinds = [k for k in ("static", "kinetic") if k in out]
    for j in range(len(next(iter(out.values()))["median"])):
        parts = []
        for kind in kinds:
            r = out[kind]
            if np.isfinite(r["f0"][j]):
                parts.append(f"fric_{kind} = {r['f0'][j]:.2f}")
                if r["mu"][j] > 0:
                    parts.append(f"fric_{kind}_mu = {r['mu'][j]:.3f}")
        print(f"  joint{j+1}:  " + "   ".join(parts))
    if "static" in out and "kinetic" in out:
        s, k = out["static"], out["kinetic"]
        bad = [j + 1 for j in range(len(s["f0"]))
               if np.isfinite(s["f0"][j]) and np.isfinite(k["f0"][j]) and s["f0"][j] < k["f0"][j] - 0.02]
        if bad:
            print(f"  note: f0_static < f0_kinetic on joints {bad} (backlash under-read / viscous in the "
                  f"f-sweep) - fine for the Stribeck blend, it just ramps up at onset instead of down")


if __name__ == "__main__":
    main()
