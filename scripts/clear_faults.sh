#!/usr/bin/env bash
# Clear latched faults on all B601-RS motors (joints + gripper).
#
# A latched fault (e.g. undervoltage after a supply brownout) leaves a RobStride motor
# answering feedback frames but ignoring enable/MIT commands: gravity_drag shows fb:6
# with torq_fb all 0.00 and the arm never stiffens. This clears the fault; it does NOT
# enable the motors or command any torque.
#
# Run with the arm idle (no gravity_drag/teleop process on the bus).
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

    still_faulted = []
    for name, mid, m in zip(names, ids, handles):
        try:
            f0, w0 = m.robstride_get_fault_report()
        except CallError as e:
            print(f"{name:8s} id {mid}: no answer to fault report ({e})")
            still_faulted.append(name)
            continue
        m.clear_error()
        time.sleep(0.05)
        f1, w1 = m.robstride_get_fault_report()
        if f1 or w1:
            status = "STILL FAULTED"
            still_faulted.append(name)
        elif f0 or w0:
            status = "cleared"
        else:
            status = "no fault"
        print(f"{name:8s} id {mid}: fault 0x{f0:08X} warn 0x{w0:08X} -> fault 0x{f1:08X} warn 0x{w1:08X}  [{status}]")

    if still_faulted:
        print(f"\nNOT cleared: {', '.join(still_faulted)} — the fault condition is still present "
              "(check supply voltage / temperature), or the motor needs a power cycle.")
        sys.exit(1)
    print("\nall motors fault-free")
finally:
    arm.close()
EOF
