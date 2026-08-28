"""Drag-teach rebalancing, introduced one step at a time.

Step 2 - LateralAssist as a takeover servo. Step 1 capped the assist below joint1's
breakaway friction so it could only tip the balance - and joint1 never moved. The
mechanics: a sideways push at the gripper breaks the wrist loose at ~1 N of contact
force (0.21 N.m breakaway / 0.20 m lever) while joint1 needs ~1.8 N (0.53 / 0.29);
force ramps from zero, joint5 slips first, the gripper then follows the hand, and the
contact force plateaus near joint5's kinetic level. The chain yields at its weakest
link, so joint1 never sees breakaway torque no matter how hard the intent of the push.

Step 2 therefore drives joint1 to TAKE OVER the motion the wrist is absorbing. The
lateral EE velocity produced by every joint EXCEPT joint1,

    v_lat = u1 . (Jv[:, 1:] @ v[1:]),        u1 = Jv[:, 0] / |Jv[:, 0]|

is exactly the sideways motion the wrist is yielding to the hand. Joint1 is servoed
to carry it instead, with integral action, because the wrist yields slowly (a few
cm/s): a proportional term alone tops out well below breakaway, while a persistent
yield that joint1 is not carrying is precisely the signal to keep winding torque up:

    v1*  = clip(sign(v_lat) * (|v_lat| - deadzone) / |Jv[:, 0]|, +-v1_max)
    err  = v1* - v1,   engage = tanh(|v_lat| / v0)
    I   += engage * ki * err * dt          (leaks to 0 in ~t_leak when disengaged)
    tau1 = clip(engage * kv * err + I, +-tau_max)

tau_max now deliberately EXCEEDS joint1's breakaway - the servo, not the cap, keeps
it safe: with no wrist yield, engage = 0, the integrator drains, and the assist
outputs nothing (it cannot start motion on its own, and dragging joint1 directly by
the forearm feels unchanged); the velocity target is bounded by the hand's own speed
and v1_max; err reverses to unwind and brake as soon as joint1 outruns the hand.
v_lat is low-passed so the handover is smooth instead of chattering when the wrist
stops yielding. Vertical or radial pushes produce no lateral component and are
ignored. The controller's vel_abort freeze and per-joint tau_max clamp remain as
backstops.
"""
from __future__ import annotations

import numpy as np


class LateralAssist:
    def __init__(self, dyn, *, tau_max: float = 1.2, kv: float = 3.0, ki: float = 15.0,
                 v1_max: float = 1.2, deadzone: float = 0.01, v0: float = 0.02,
                 tau_f: float = 0.12, t_leak: float = 0.15) -> None:
        self.dyn = dyn
        self.tau_max = float(tau_max)   # N.m cap on the joint1 torque (above breakaway on purpose)
        self.kv = float(kv)             # N.m per rad/s of joint1 velocity error
        self.ki = float(ki)             # N.m/s per rad/s: winds up while a yield persists unserved
        self.v1_max = float(v1_max)     # rad/s cap on the joint1 velocity target (vel_abort is 4)
        self.deadzone = float(deadzone) # m/s of lateral EE velocity ignored (estimator noise)
        self.v0 = float(v0)             # m/s engagement knee: no yield -> no torque at all
        self.tau_f = float(tau_f)       # s low-pass on the wrist-yield velocity (smooth handover)
        self.t_leak = float(t_leak)     # s integrator drain time constant when disengaged
        self.v_lat = 0.0                # filtered lateral EE velocity from the non-j1 joints
        self.integ = 0.0                # integral torque state
        self.tau1 = 0.0                 # last output, for the status line

    def update(self, q: np.ndarray, v: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Assist torque (only joint1 is ever nonzero)."""
        Jv = self.dyn.ee_jacobian(q)[:3]
        u1 = Jv[:, 0]
        n1 = np.linalg.norm(u1)
        tau = np.zeros(len(v))
        if n1 < 1e-6:              # gripper on the joint1 axis: no lateral direction exists
            self.v_lat = 0.0
            self.integ = 0.0
            self.tau1 = 0.0
            return tau
        u1 = u1 / n1
        raw = float(u1 @ (Jv[:, 1:] @ v[1:]))          # lateral EE velocity the wrist is yielding
        self.v_lat += (dt / (self.tau_f + dt)) * (raw - self.v_lat)
        mag = max(0.0, abs(self.v_lat) - self.deadzone)
        engage = float(np.tanh(mag / self.v0))
        v1_tgt = float(np.clip(np.sign(self.v_lat) * mag / n1, -self.v1_max, self.v1_max))
        err = v1_tgt - v[0]
        self.integ *= 1.0 - min(1.0, (1.0 - engage) * dt / self.t_leak)
        self.integ = float(np.clip(self.integ + engage * self.ki * err * dt, -self.tau_max, self.tau_max))
        self.tau1 = float(np.clip(engage * self.kv * err + self.integ, -self.tau_max, self.tau_max))
        tau[0] = self.tau1
        return tau
