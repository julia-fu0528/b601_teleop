#!/usr/bin/env python3
"""Fit per-joint gravity scales from 'c'-key captures (calib.csv written by gravity_drag.py).

Model per joint i:  resid_i = (k_i - 1) * g_raw_i  (+ c_i with --bias), where resid is the torque the
PD hold had to add on top of the currently configured feed-forward ff_i = g_scale_i * g_raw_i.
Since ff was already scaled, the fit is on:  ff_i + resid_i  ~=  k_i * g_raw_i  (+ c_i).
Captures are static, so each residual carries a +/- Coulomb friction uncertainty (~0.5 N.m on j2/j3,
~0.2 on the wrist); use several well-spread poses and prefer poses where |g_raw_i| is large.

Usage:  python scripts/fit_gravity.py calib.csv [--bias]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b601 import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--bias", action="store_true", help="also fit a constant torque offset per joint")
    ap.add_argument("--rows", help="fit only these captures, python slice syntax (1-based), e.g. '32:' = from "
                                   "capture 32 on, '1:10' = the first batch; global fits average incompatible "
                                   "pose regions - prefer fitting the region you actually drag in")
    ap.add_argument("--config", default=str(ROOT / "config" / "b601_rs.toml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    rows = list(csv.DictReader(open(args.csv)))
    if args.rows:
        a, _, b = args.rows.partition(":")
        lo = int(a) - 1 if a else 0
        hi = int(b) if b else len(rows)
        rows = rows[lo:hi]
        print(f"using captures {lo + 1}..{lo + len(rows)}")
    n = len(cfg.joints)
    if not rows:
        raise SystemExit("no captures in file")
    q = np.array([[float(r[f"q{i+1}"]) for i in range(n)] for r in rows])
    graw = np.array([[float(r[f"graw{i+1}"]) for i in range(n)] for r in rows])
    ff = np.array([[float(r[f"ff{i+1}"]) for i in range(n)] for r in rows])
    resid = np.array([[float(r[f"resid{i+1}"]) for i in range(n)] for r in rows])
    need = ff + resid          # torque the motor actually had to supply
    print(f"{len(rows)} captures")
    print(f"{'joint':8s} {'cfg':>6s} {'k_fit':>6s} {'bias':>6s} {'rms':>6s} {'|g|max':>7s}  note")
    for i, j in enumerate(cfg.joints):
        x, y = graw[:, i], need[:, i]
        if args.bias:
            A = np.column_stack([x, np.ones_like(x)])
        else:
            A = x[:, None]
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        k = sol[0]; c = sol[1] if args.bias else 0.0
        rms = float(np.sqrt(np.mean((A @ sol - y) ** 2)))
        gmax = float(np.abs(x).max())
        note = ""
        if gmax < 0.4:
            note = "gravity torque below friction: fit not meaningful, keep cfg"
        elif abs(k - j.g_scale) < 0.03:
            note = "matches config"
        print(f"{j.name:8s} {j.g_scale:6.2f} {k:6.2f} {c:+6.2f} {rms:6.2f} {gmax:7.2f}  {note}")
    print("\nSuggested config (only for joints with |g|max well above friction):")
    for i, j in enumerate(cfg.joints):
        x, y = graw[:, i], need[:, i]
        if np.abs(x).max() >= 0.4:
            k = float(np.linalg.lstsq(x[:, None], y, rcond=None)[0][0])
            rms_k = float(np.sqrt(np.mean((k * x - y) ** 2)))
            A = np.column_stack([x, np.ones_like(x)])
            (k2, c2), *_ = np.linalg.lstsq(A, y, rcond=None)
            rms_kc = float(np.sqrt(np.mean((A @ [k2, c2] - y) ** 2)))
            if abs(c2) > 0.05 and rms_kc < 0.5 * rms_k and len(rows) >= 5:
                print(f"  {j.name}: g_scale = {k2:.2f}, g_bias = {c2:+.2f}   (offset halves the rms: {rms_k:.2f} -> {rms_kc:.2f})")
            else:
                print(f"  {j.name}: g_scale = {k:.2f}, g_bias = 0.0")


if __name__ == "__main__":
    main()
