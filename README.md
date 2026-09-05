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

## Balanced drag (`--balance`, separate from `--assist`)

`--balance KAPPA` runs the momentum-observer + Cartesian-inertia-shaping controller from the
balanced-drag derivation (2026-08-27 artifact): the hand torque is estimated from motion (the
RobStride torque echo is just the command), mapped to an isotropic virtual mass at the gripper
(default 1.8 kg / 0.06 kg.m^2), and joints heavier than that target assist while joints that
would run away (the folding wrist) resist. `KAPPA` in [0, 2] is the max lightening ratio minus
one; 2 is the stability ceiling of the ~90 Hz loop.

### Modes

Three independent toggles — `--balance KAPPA` (inertia shaping, 0 = off), `--balance-fric`
(friction comp, 0.85 default, 0 = off), `--balance-sustain` (j1 sustained relief, on by
default). The startup banner names the active mode.

| mode | run | meaning |
|---|---|---|
| plain drag | `drag` | gravity ff + config `fric_comp`; no observer |
| observer only | `drag --observe` | zero output, `r` logged — safe first run (`r` ≈ 0 at rest, follows your hand) |
| full balanced drag | `drag --balance 1` (or `2`) | shaping + 85 % friction + j1 sustain — the working mode |
| friction comp only | `drag --balance 0` | natural inertia, relieved friction |
| shaping only | `drag --balance 1 --balance-fric 0` | rebalanced inertia, full friction |
| no sustain | `drag --balance 1 --balance-sustain 0` | all joints drive-gated (relief only while out-pushing friction) |
| torque rebalance (legacy) | `drag --assist 1.2` | old wrist→shoulder pipeline; exclusive with `--balance` |

Live keys switch modes mid-run (state printed, shown on the status line): `b` = shaping,
`bf` = friction comp, `bs` = j1 sustain. Tuning: `--balance-md/-irot` or live keys `m <kg>` /
`i <kg·m²>` / `+` / `-` (virtual inertia, slewed ~0.5 s), `--balance-fo`, `--balance-resist`.

### Calibrating friction (static + kinetic)

From DRAG, arm in a mid-range pose, hands off: `f` + Enter measures **kinetic** friction
(±0.15 rad triangle sweep, PD residual split by direction); `s` + Enter measures **static**
breakaway (per joint, both directions, ~10 s each; the gravity residual cancels and is printed
as a cross-check). Repeat at 4–6 spread poses — friction is load-dependent — each run appends
to `friction.csv`, then:

```bash
python scripts/fit_friction.py friction.csv
```

fits `f_j(q) = f0_j + mu_j·|g_j(q)|` per joint (median where the fit isn't better) and prints
paste-ready `fric_*` / `fric_*_mu` lines for `config/b601_rs.toml`.

At runtime `--balance-fric` compensates 85 % of that level while a joint moves (Stribeck:
static level at onset → kinetic at speed), gated on the estimated drive so it cannot creep;
every joint keeps its relief while clearly moving (> 0.1 rad/s), capped at (real level − margin)
with per-joint margins sized above the measured model residuals — so steady sliding is relieved
without over-pushing, yet an un-driven joint always decelerates.
Breakaway itself stays yours (invisible to a motion-based estimator). Notes: the sweeps always
measure raw friction (all friction ffs are DRAG-only, off during measurement); `fric_comp` is
auto-disabled under `--balance` (double compensation is refused); module defaults apply until
you calibrate. Guards in every mode: eigen-clipped gains, 2 s ramp, singularity fade, per-joint
caps, runaway detector. Offline tests: `tests/test_balance.py`.

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
