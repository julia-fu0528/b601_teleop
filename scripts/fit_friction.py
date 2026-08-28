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
    args = ap.parse_args()

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
