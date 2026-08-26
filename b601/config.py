"""TOML configuration for the B601-RS."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JointCfg:
    name: str
    id: int
    model: str
    g_scale: float = 1.0
    g_bias: float = 0.0      # constant torque added to the feed-forward (cable harness etc.), N.m
    tau_max: float = 1.0
    kd_drag: float = 0.0
    kp_drag: float = 0.0     # optional stiffness while dragging (e.g. to hide wrist backlash)
    fric_comp: float = 0.0   # Coulomb friction feed-forward while dragging, N.m (keep below the joint's real friction!)
    hold_kp: float = 10.0
    hold_kd: float = 0.5


@dataclass(frozen=True)
class GripperCfg:
    id: int
    model: str
    tau_max: float = 1.5
    hold_kp: float = 8.0
    hold_kd: float = 0.5


@dataclass(frozen=True)
class LoopCfg:
    rate_hz: float = 100.0
    ramp_s: float = 3.0
    fade_s: float = 2.0
    ramp_window: float = 0.2
    vel_abort: float = 4.0
    read_timeout_ms: int = 50
    max_read_failures: int = 3
    temp_abort_c: float = 80.0
    release_s: float = 3.0


@dataclass(frozen=True)
class LeaderCfg:
    """Seeed reBot Arm 102 leader (Fashion Star UART servos)."""
    port: str
    baudrate: int = 1_000_000
    ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)                       # servo id per follower joint (j1..j6, gripper)
    ranges: tuple[tuple[float, float], ...] = ((-150, 150), (-1, 170), (-200, 1), (-80, 90), (-90, 90), (-130, 130), (0, 270))
    directions: tuple[float, ...] = (1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 6.0)
    offsets: tuple[float, ...] = (0.0,) * 7                             # leader deg at which the follower joint is at 0 (calib file)
    stale_s: float = 0.2


@dataclass(frozen=True)
class TeleopCfg:
    kp: tuple[float, ...] = (50.0, 50.0, 50.0, 30.0, 50.0, 50.0)
    kd: tuple[float, ...] = (3.0, 5.0, 5.0, 3.0, 4.0, 4.0)
    limits_deg: tuple[tuple[float, float], ...] = ((-145, 145), (0, 170), (0, 200), (-80, 90), (-90, 90), (-130, 130))
    max_vel: float = 4.0           # rad/s rate limit on the follower target while tracking
    vel_ff: bool = True            # command the target velocity in the MIT frame (kd then damps the velocity *error*)
    vel_tau: float = 0.04          # s, low-pass on the target velocity estimate
    lead_s: float = 0.03           # s, predict the target ahead by ~sensing + loop + feedback latency
    engage_vel: float = 0.4        # rad/s approach speed when engaging
    max_engage_deg: float = 90.0   # refuse to start if any joint is further than this from the leader
    jump_deg: float = 30.0         # leader change per sample treated as a glitch
    gripper_limits_deg: tuple[float, float] = (0.0, 270.0)
    gripper_kp: float = 12.0
    gripper_kd: float = 0.05
    gripper_tau_max: float = 3.5
    gripper_tau_hold: float = 1.0


@dataclass(frozen=True)
class Config:
    root: Path
    channel: str
    host_id: int
    urdf: Path
    lock_joints: tuple[str, ...]
    joints: tuple[JointCfg, ...]
    gripper: GripperCfg | None
    loop: LoopCfg
    leader: LeaderCfg | None = None
    teleop: TeleopCfg = TeleopCfg()

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    root = path.parent.parent if path.parent.name == "config" else path.parent

    bus = raw.get("bus", {})
    model = raw.get("model", {})
    urdf = Path(model["urdf"])
    if not urdf.is_absolute():
        urdf = root / urdf
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")

    joints = tuple(JointCfg(**j) for j in raw.get("joint", []))
    if not joints:
        raise ValueError("config has no [[joint]] entries")
    ids = [j.id for j in joints]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate motor ids in config: {ids}")

    grip = raw.get("gripper")
    gripper = GripperCfg(**grip) if grip else None

    def _tup(v):
        return tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v

    leader = None
    if raw.get("leader"):
        ld = {k: _tup(v) for k, v in raw["leader"].items()}
        calib = root / "config" / "leader_calib.json"
        if calib.exists():
            import json
            ld["offsets"] = tuple(json.loads(calib.read_text())["offsets"])
        leader = LeaderCfg(**ld)
        if not (len(leader.ids) == len(leader.ranges) == len(leader.directions) == len(leader.offsets) == 7):
            raise ValueError("[leader] ids/ranges/directions/offsets must all have 7 entries (j1..j6, gripper)")
    teleop = TeleopCfg(**{k: _tup(v) for k, v in raw.get("teleop", {}).items()})
    if len(teleop.kp) != len(joints) or len(teleop.kd) != len(joints) or len(teleop.limits_deg) != len(joints):
        raise ValueError("[teleop] kp/kd/limits_deg must have one entry per arm joint")
    if gripper is not None and gripper.id in ids:
        raise ValueError("gripper id collides with an arm joint id")

    return Config(
        root=root,
        channel=str(bus.get("channel", "can0")),
        host_id=int(bus.get("host_id", 0xFD)),
        urdf=urdf,
        lock_joints=tuple(model.get("lock_joints", ())),
        joints=joints,
        gripper=gripper,
        loop=LoopCfg(**raw.get("loop", {})),
        leader=leader,
        teleop=teleop,
    )
