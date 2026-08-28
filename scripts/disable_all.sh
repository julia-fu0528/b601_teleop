#!/usr/bin/env bash
# Disable (de-energize) ALL B601-RS motors: joints + gripper.
#
# Use this as a software kill, or after a crashed gravity_drag/teleop process leaves
# motors energized. A disabled motor produces NO torque: the arm goes limp and will
# FALL under gravity — rest it on the table or support it before running this.
#
# Run with the arm otherwise idle (no gravity_drag/teleop process on the bus).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/Users/juliafu/miniforge3/envs/rebot/bin/python}"
# libPCBUSB lives in /usr/local/lib; normally added by the conda env's activate hook
export DYLD_LIBRARY_PATH="/usr/local/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
cd "$REPO"

exec "$PY" - <<'EOF'
import sys
import time

sys.path.insert(0, ".")
from motorbridge import CallError

from b601.arm import RobstrideArm
from b601.config import load_config

cfg = load_config("config/b601_rs.toml")
arm = RobstrideArm(cfg)
try:
    names = list(cfg.joint_names)
    ids = [j.id for j in cfg.joints]
    handles = list(arm.motors)
    if arm.gripper is not None:
        names.append("gripper")
        ids.append(cfg.gripper.id)
        handles.append(arm.gripper)

    failed = []
    for name, mid, m in zip(names, ids, handles):
        try:
            m.disable()
            time.sleep(0.02)
            print(f"{name:8s} id {mid}: disabled")
        except CallError as e:
            failed.append(name)
            print(f"{name:8s} id {mid}: DISABLE FAILED ({e})")

    if failed:
        print(f"\nNOT disabled: {', '.join(failed)} — motor did not answer on the bus "
              "(check CAN connection/power, or power-cycle the supply to be safe).")
        sys.exit(1)
    print("\nall motors disabled — no torque on any joint")
finally:
    arm.close()
EOF
