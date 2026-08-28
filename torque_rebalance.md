# Torque Rebalance — math and algorithm

`b601_teleop / b601/assist.py :: TorqueRebalance` — designed 2026-08-26

**Goal.** Isotropic drag feel: a hand force at the gripper should translate the EE the same way in every direction, instead of escaping as cheap wrist rotation.

## Notation

| symbol | meaning |
|---|---|
| $q,\ \dot q \in \mathbb{R}^6$ | joint positions and velocities (joint1…joint6) |
| $J \in \mathbb{R}^{3\times 6}$ | position Jacobian of `gripper_end`, world frame: $v_{ee} = J\dot q$ |
| $J_i$ | column $i$: direction & lever with which joint $i$ moves the EE |
| $B = J_{[:,\,1:3]}$ | **upstream** columns (base/shoulder, joints 1–3) |
| $W = J_{[:,\,4:6]}$ | **downstream** columns (wrist, joints 4–6) |
| $\odot$ | elementwise product |
| $\mathrm{lp}_T(x)$ | first-order low-pass: $x \leftarrow x + \frac{dt}{T+dt}(x_{raw} - x)$ |

## Physical background — why the wrist "does everything"

A hand force $F$ at the EE loads **every** joint at once through the transpose map (virtual work):

$$\tau_i = J_i^{\top} F$$

Joint $i$ yields only when $|\tau_i|$ exceeds its own breakaway friction $b_i$. The chain yields at its weakest link: the wrist crosses its threshold at a lower push force (lower friction; for radial pushes also a bigger lever), the gripper then follows the hand, and the contact force plateaus **below** the base joints' threshold. Additionally, while sliding, each joint's speed per newton is

$$\dot q_i = \frac{J_i^{\top} F - b_i}{k_{d,i}}$$

and the wrist's slope $(\text{lever}/k_d)$ is ~20× the shoulder's, so the wrist supplies the hand's motion first and the force never climbs.

## Step 1 — measure the external torque on the downstream joints

External torque is observable only through **backdrive**: a stuck joint hides it inside its static friction band, and in free-float drag ($k_p=0$) the motor does not resist a push, so torque feedback cannot see it either. While a joint moves, the hand torque on it is exactly what sustains the motion:

$$\tau_{dn} = f_{dn} \odot \tanh\!\left(\frac{\dot q_{dn}}{v_0}\right) + k_{dn} \odot \dot q_{dn}, \qquad \tau_{dn} \leftarrow \mathrm{lp}_{\tau_f}(\tau_{dn})$$

Zero at rest by construction. $f_{dn} = [0.30,\ 0.21,\ 0.21]$ N·m (Seeed friction sweep, j4/j5/j6); $k_{dn}$ = the `kd_drag` actually commanded on the wrist (default 0.05).

## Step 2 — infer the external torque on the upstream joints (Jacobian)

The upstream joints are the stuck ones, so their external torque cannot be measured; it is reconstructed. The hand force explaining the downstream torques is the damped least-squares solution of $W^{\top} F = \tau_{dn}$:

$$A = W^{\top}, \qquad \hat F = \left(A^{\top}A + \lambda^2 I\right)^{-1} A^{\top} \tau_{dn}, \qquad \tau_{up} = B^{\top}\hat F$$

**Observability caveat:** only the force component in the span of the *backdriving* joints' lever rows is recoverable (minimum-norm elsewhere). One yielding wrist joint ⇒ 1-D information — e.g. a pure-outward push and an out-and-up push give identical j4 torque and are indistinguishable. This is why step 4 does **not** take its direction from $\hat F$.

## Step 3 — compare the halves → transfer gate $w$

Two rejected comparisons: *raw torque vs raw torque* (a joint yields against its **own** breakaway; laterally joint1 receives more raw torque than joint5 and still loses) and *threshold-normalized loading vs loading* (the step-2 ambiguity inflates the inferred upstream loading, and a loaded-but-stuck joint is not helping anyway — loading is not carriage). The robust comparison is what each half is actually **doing**:

$$r_{down} = \max_i \frac{|\tau_{dn,i}|}{b_{dn,i}}, \qquad c_{up} = \frac{\lVert B\dot q_{up}\rVert}{\lVert B\dot q_{up}\rVert + \lVert W\dot q_{dn}\rVert + \varepsilon}$$

$$w = \mathrm{clip}(r_{down},\,0,\,1)\,\cdot\,(1 - c_{up}) \ \in [0,1]$$

$w\!\approx\!1$: wrist carries the push alone → transfer fully. $w\!\to\!0$: upstream has taken over the motion → stop pushing it. $w=0$: no hand → no output.

## Step 4 — transfer ($J^{\top}$ size, $J^{+}$ direction)

The two Jacobian maps answer different questions and can **disagree in sign** in skewed geometry (folded-wrist radial push: a force along the motion loads j2 negative, but j2 must rotate positive to carry that motion):

- $J^{\top}F$ — torque the force **exerts** on a joint (statics)
- $J^{+}u$ — rotation the joint needs to **carry** motion $u$ (kinematics)

So: per-joint direction pattern from the motion-carry map, magnitude from the sensed force.

$$u_m = \frac{W\dot q_{dn}}{\lVert W\dot q_{dn}\rVert} \qquad \text{(EE direction the wrist is carrying)}$$

$$p = B^{\top}\left(BB^{\top} + \lambda^2 I\right)^{-1} u_m \qquad \text{(upstream rates that carry } u_m\text{)}$$

$$B^{\top} F_{eq} = p \ \text{(DLS)}, \qquad s = \gamma\,\frac{\lVert \hat F\rVert}{\lVert F_{eq}\rVert} \qquad \text{(pattern's EE force} = \gamma \times \text{sensed force)}$$

$$\tau_{tx} = \mathrm{clip}\!\left(\mathrm{lp}_{\tau_f}(w \cdot s \cdot p),\ \pm\tau_{max}\right) \qquad \text{per joint 1–3}$$

Injected into the drag torque: $\tau_{cmd} = g(q) + f_{comp} \odot \tanh(\dot q/0.08) + \tau_{tx}$.

## Safety properties (all tested in `tests/test_assist.py`)

- **Zero at rest**: no backdrive ⇒ $\tau_{dn}=0$, $w=0$, $u_m$ undefined ⇒ no output.
- **Bounded**: EE-equivalent force ≤ $\gamma\times$ sensed hand force; per-joint $\tau_{max}$ cap; controller clamp and `vel_abort` freeze as backstops.
- **Self-fading**: gate closes as the upstream half takes over; the $u_m$ source vanishes when the wrist stops slipping.
- **No wrong-way pressing**: every driven joint aids the observed motion ($J^{+}$ sign).
- No rate targets, no integrator, no added damping.

## Pseudocode (per control cycle, 100 Hz, DRAG phase only)

```text
state: tau_dn = 0 (R^3), tau_tx = 0 (R^3)        # low-pass filter states

update(q, qd, dt):
    Jv = jacobian(q)[:3]                         # 3x6
    B, W = Jv[:, :3], Jv[:, 3:]

    # step 1: measure downstream external torque from backdrive
    raw    = fric_dn * tanh(qd[3:] / v0) + kd_dn * qd[3:]
    tau_dn = lowpass(tau_dn, raw, tau_f, dt)

    # step 2: infer hand force and upstream received torque
    A      = W.T
    F_hat  = solve(A.T A + lam2*I3, A.T tau_dn)
    tau_up = B.T @ F_hat                         # displayed; sizing input

    # step 3: compare halves -> gate
    r_down   = max(|tau_dn| / brk_dn)
    carry_up = |B qd[:3]| / (|B qd[:3]| + |W qd[3:]| + eps)
    w        = clip(r_down, 0, 1) * (1 - carry_up)

    # step 4: transfer (J^T size, J+ direction)
    v_dn = W @ qd[3:]
    if |v_dn| < 1e-4:  raw_tx = 0
    else:
        u_m     = v_dn / |v_dn|
        p       = B.T @ solve(B B.T + lam2*I3, u_m)
        F_eq    = solve(B B.T-style DLS of  B.T F = p)
        s       = gain * |F_hat| / |F_eq|
        raw_tx  = w * s * p
    tau_tx = clip(lowpass(tau_tx, raw_tx, tau_f, dt), -tau_max, tau_max)

    return [tau_tx, 0, 0, 0]                     # torque on joints 1..3 only
```

## Parameters (defaults)

| param | default | meaning |
|---|---|---|
| $\gamma$ (`gain`) | 2.0 | force multiplication wrist → base (`--assist-gain`) |
| $\tau_{max}$ | 1.2 N·m | cap per upstream joint (the `--assist TAU` argument) |
| $f_{dn}$, $b_{dn}$ | [0.30, 0.21, 0.21] N·m | wrist Coulomb friction / breakaway (Seeed sweep) |
| $k_{dn}$ | [0.05, 0.05, 0.05] | wrist `kd_drag` actually commanded |
| $v_0$ | 0.08 rad/s | tanh knee (above finite-difference velocity noise) |
| $\tau_f$ | 0.1 s | low-pass time constant (measurement and output) |
| $\lambda$ | 0.05 m | damping of every least-squares solve |

## Results (sim, identical 2 N pushes at the folded-wrist pose $q = [0, 1.08, 1.22, -1, 0, 0]$)

| direction | plain | rebalanced |
|---|---|---|
| lateral | 0.263 m | 0.269 m |
| radial | 0.085 m | **0.148 m** (+74%, the joint4 direction) |
| vertical | 0.233 m | 0.233 m |

Direction spread 3.1× → 1.8×. High-stiction j1 (0.8 N·m) recruited with peak 0.49 N·m transfer, zero torque sign flips, stops with the hand.

## Run

```
python scripts/gravity_drag.py drag --assist 1.2 [--assist-gain 2]
```

Status line: `tau_dn <j4 j5 j6> -> tau_up <j1 j2 j3>  w <gate>  tx <j1 j2 j3>`
