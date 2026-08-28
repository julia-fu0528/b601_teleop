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

The three behaviours are independent toggles — `--balance KAPPA` (inertia shaping, 0 = off),
`--balance-fric` (friction compensation, default 0.85, 0/off = none), `--balance-sustain`
(sustained relief on joint1, default on). Every combination is a usable mode; the startup
banner names which one you are in.

| mode | run | what it does / when to use it |
|---|---|---|
| **Plain gravity drag** | `drag` | Baseline: gravity feed-forward + config `fric_comp` only. No observer, no shaping. The reference to compare everything against. |
| **Observer only** | `drag --observe` | Zero output — identical feel to plain drag, but the momentum observer runs and `r` (estimated hand torque per joint) shows on the status line / `--log` CSV. The safe first run after any model or calibration change: at rest `r` must sit near zero; push a joint and `r` must follow your hand. |
| **Full balanced drag** | `drag --balance 1` (or `2`) | Everything on: inertia rebalanced toward the 1.8 kg / 0.06 kg·m² virtual body (assist where heavier, resist where lighter), 85 % load-tracking friction compensation, sustained relief on joint1. The normal working mode; `2` is the lightest allowed. |
| **Friction comp only** | `drag --balance 0` | No inertia shaping — the arm keeps its natural (gravity-compensated) inertia, but sliding friction is 85 %-relieved (drive-gated; j1 sustained). Useful to A/B how much of the feel comes from friction vs shaping. |
| **Shaping only** | `drag --balance 1 --balance-fric 0` | Inertia rebalancing without any friction feed-forward. Isolates the shaping: directions should feel mass-balanced but sliding drag stays at full friction. |
| **No j1 sustain** | `drag --balance 1 --balance-sustain 0` | Full mode but joint1 falls back to the drive-only gate like every other joint: relief only while you out-push full friction. Use if close-in lateral behaviour ever feels suspect — this removes the only sustained-relief path. |
| **Torque rebalance (legacy)** | `drag --assist 1.2` | The older wrist→shoulder transfer pipeline (`b601/assist.py`, see `torque_rebalance.md`). Separate experiment; mutually exclusive with `--balance`. |

Fine-tuning on top of any mode: `--balance-md` / `--balance-irot` or the live keys (below) for
the virtual inertia, `--balance-fo` for observer bandwidth, `--balance-resist` for how much the
light directions may be stiffened, `--balance-fric 0.5` etc. for a gentler friction fraction.

### Calibrating friction (static + kinetic)

While dragging (`--balance` or plain `drag`), park the arm in a mid-range pose (the L pose) and:

* type `f` + Enter — **kinetic** (sliding Coulomb) friction: slow ±0.15 rad triangle sweep under
  stiff PD, residual split by direction of motion;
* type `s` + Enter — **static** (breakaway) friction: joints one at a time, all others held stiff;
  the free joint's feed-forward ramps slowly (0.25 N·m/s RS-06, 0.12 RS-00) until it moves 8 mrad,
  in both directions. `f_static = (τ⁺+|τ⁻|)/2`, so the gravity residual cancels (and is printed as
  a cross-check). ~10 s per joint, hands off.

Friction is load-dependent (gear mesh / bearing load change with pose), so repeat both sweeps at
4–6 well-spread poses — each run appends a row to `friction.csv` (`--fric-csv`) — then

```bash
python scripts/fit_friction.py friction.csv
```

fits the load model `f_j(q) = f0_j + mu_j*|g_j(q)|` per joint (gear-mesh loss grows with the
transmitted gravity torque — this is what made the per-pose sweeps spread), keeps it only where
it beats the median, and emits paste-ready
`fric_* = f0` / `fric_*_mu = mu` values for each `[[joint]]` in `config/b601_rs.toml`
(the runtime then tracks the pose: compensation follows `f0 + mu*|g(q)|` instead of one all-pose median)
(a single sweep also prints its own paste lines if one pose is all you need). With `--balance`, 85 % of the measured friction is then compensated while
moving (`--balance-fric`, default 0.85): a Stribeck curve pays ~0.85·f_static right after breakaway,
decaying to 0.85·f_kinetic as speed builds — gated on the estimated drive, so it cannot creep. Exception: **joint1** keeps
its relief while clearly moving (> 0.1 rad/s), capped at (real kinetic friction − 0.20 N·m) — its
axis is vertical and its friction constant, so nothing can self-drive, and close-in lateral drags
(short lever to the j1 axis) stop paying full base friction. `--balance-sustain none` disables it,
or name other joints at your own judgement.
Breakaway itself is still paid by your hand (invisible to a motion-based estimator); what changes is
that the joint stops feeling sticky the moment it moves. The sweeps themselves always measure the
raw friction — every friction feed-forward is DRAG-only and off during measurement, so no need for
`--fric 0` while calibrating. When the balance friction ff is active, the old ungated `fric_comp`
is switched off automatically — combining both is refused (they would compensate the same friction
twice). `--balance-fric 0` falls back to the old `fric_comp` path if you want to A/B them. Until you calibrate, module defaults are
used (`FRIC` in `b601/balance.py`, static = kinetic).

While dragging, retune by feel without restarting: type `m 2.0` + Enter (virtual mass, kg),
`i 0.08` (rotational, kg.m^2), or `+` / `-` (25 % heavier / lighter); the change slews in over
~0.5 s. Startup knobs: `--balance-md/-irot` (target mass/inertia), `--balance-fo` (observer Hz, default 3),
`--balance-resist` (how much heavier the light directions may be made). Guards on top of the
usual ones: eigen-clipped gains, 2 s output ramp, singularity fade, per-joint caps
(0.4 x tau_max), and a runaway detector (kinetic energy rising with no estimated hand power)
that halves the gain per trip. The URDF has no rotor inertia yet — that under-estimate is the
safe direction, but identify it before trusting kappa 2. Offline tests: `tests/test_balance.py`.

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
