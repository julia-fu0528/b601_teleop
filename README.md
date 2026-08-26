# b601_teleop

Control experiments for the Seeed **reBot Arm B601-RS** (6 RobStride joints + gripper, CAN @ 1 Mbps).

## Setup (macOS)

```bash
conda activate rebot        # motorbridge 0.5.1 + pinocchio 4.1 + numpy, DYLD path for libPCBUSB
motorbridge-cli scan --vendor robstride --channel can0 --start-id 1 --end-id 7 --timeout-ms 300
```

`urdf/00-arm-rs_asm-v3/` is the vendor URDF (Seeed reBotArm_control_py). Motor mechPos == URDF q,
motor zero == URDF zero (rest pose = extended arm), identity sign. Verified by Seeed's contact-free
calibration (2026-07-17); the per-joint scales in `config/b601_rs.toml` come from it.

## Gravity-compensated drag (MIT mode, feed-forward only)

```bash
python scripts/gravity_drag.py read              # motors off: sign probe, shows q and the ff it would send
python scripts/gravity_drag.py drag --sim        # simulator: exercise ramp / hold / release / runaway guard
python scripts/gravity_drag.py drag --joints joint3      # first real test: one joint
python scripts/gravity_drag.py drag                       # all joints
python scripts/gravity_drag.py drag --kd 0                # pure feed-forward, no damping
```

Control law while dragging: `tau_i = clamp(scale * g_scale_i * g_i(q))`, `kp = 0`, `kd = kd_drag_i`
(light damping; `--kd 0` for none). `g(q)` is Pinocchio's generalized gravity from the URDF,
evaluated at the measured pose every cycle (100 Hz, mechPos param reads).

### Safety procedure (read before energizing)

1. Arm resting on the table (rest pose), workspace clear, power-switch within reach.
2. `read` first: lift joint2 / joint3 slightly by hand — values must go **positive**.
3. `drag` (all joints): during **RAMP (3 s, stiff hold while the feed-forward ramps in) and FADE
   (2 s, hold gains fade out) do not touch the arm**. If any joint drifts more than 0.2 rad from the
   start pose on its own, the controller freezes into HOLD (PD hold + feed-forward). That means the sign
   or the model is wrong — do not continue.
   A single-joint test (`--joints joint3`) only makes sense with the arm propped so that joint can swing
   freely: from the rest pose the gripper sits on the table and unpowered joints are stiff, so the powered
   joint just pushes into contact and feels "blocked".
4. When it prints `DRAG`, lift the whole arm into the elbow-up "L" pose (j2 ≈ 40°, j3 ≈ 60°, nothing
   touching) and let go. It should float and stay where you leave it (friction 0.2–0.5 N.m per joint
   covers the ~5 % model error). HOLD is stiff on purpose — `d` + Enter resumes dragging.
5. `Ctrl+C` -> HOLD (arm stays). Put the arm back on its rest, then `r` + Enter (or Ctrl+C again) -> RELEASE:
   torques fade over 3 s with strong damping, then the motors are disabled. `q!` disables immediately (arm falls).
6. Repeat with all joints.

Guards: velocity > 3 rad/s -> HOLD; 3 consecutive CAN read failures -> HOLD; motor temperature > 80 C -> HOLD;
per-joint torque clamps (config `tau_max`); start pose outside URDF limits -> refuses to start.
A loaded arm free-falls when torque is cut, so every abort *holds* instead of disabling.

## Calibrating the gravity scales on your unit

While dragging, rest the arm in a pose, let go, then type `c` + Enter: the controller holds the pose,
sweeps every joint ±3° as a slow triangle wave for 4 s (so friction averages out), measures the mean
torque the PD has to add per joint (= model error at that pose), prints it, appends it to `calib.csv`,
and resumes drag. Do this in 6–8 well-spread poses (large |g| per joint:
shoulder far forward/back, elbow high/low, wrist pitched ±90°), then

```bash
python scripts/fit_gravity.py calib.csv        # per-joint k = torque needed / URDF torque
```

and copy the suggested `g_scale` values into `config/b601_rs.toml`. Joints whose gravity torque never
exceeds friction (j1, j6, often j5) cannot be fitted this way — leave them.

## Teleoperation (reBot Arm 102 leader)

The leader is Seeed's reBot Arm 102: 7 Fashion Star UART servos on a CH340 port (`/dev/cu.usbserial-130`,
1 Mbps), read with `motorbridge-smart-servo`. Mapping (from Seeed's LeRobot integration):
`follower_deg = direction × (leader_deg − offset)`, directions `[+1, +1, −1, −1, −1, +1]`, gripper ×6.
The leader's zero coincides with the follower's rest pose, so offsets are normally 0.

```bash
python scripts/teleop.py compare          # motors off: leader (mapped) vs follower, live table
python scripts/teleop.py calibrate        # only if compare shows a constant offset: both arms in the same pose
python scripts/teleop.py run --log teleop.csv
python scripts/teleop.py run --sim        # simulator follower + scripted leader
```

`run`: ENGAGE first — the follower target starts at its own pose and glides to the leader pose at 0.4 rad/s
(never a jump), then TRACK: MIT position tracking (kp 50/50/50/30/50/50, kd 3/5/5/3/4/4, Seeed's values)
plus the calibrated gravity feed-forward, target rate-limited to 2.5 rad/s. Gripper: host-side impedance
with a 3.5 N·m force limit (1.0 N·m when holding). Guards: leader glitch (> 30° per sample, ignored; 3 in a
row → HOLD), stale leader (> 0.2 s), joint velocity, temperature, CAN failures, refusal to start if a joint
is > 90° from the leader. Keys: `h` hold, `t` re-engage, `r` release, `q!` disable now. Ctrl+C = hold, again
= release. Start with `--kp-scale 0.5` for a softer follower.

## Tests

```bash
python tests/test_gravity_drag.py    # or: pytest tests/
```
