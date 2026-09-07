"""Force/torque sensor reader for an ATI Nano25 mounted at the wrist (between joint6 and
the gripper). Standalone helper - imported by scripts/ft_drag.py; nothing else depends on it.

Two backends, same interface:
  NetFTSensor   - ATI Net F/T box (Ethernet). Streams the RDT record over UDP (no libraries).
  StubFTSensor  - returns zeros; lets the driver script run with no sensor (or in --sim).

Interface:
  s.start()            begin streaming (opens the socket / spawns the recv thread)
  s.tare(n=200)        average n samples at rest and store as bias (subtracted from reads)
  w = s.read()         latest wrench as np.array([Fx,Fy,Fz,Tx,Ty,Tz]) - N, N.m - bias applied
  s.stop()             stop streaming / close

RDT record (36 bytes, big-endian): rdt_seq u32, ft_seq u32, status u32, then Fx Fy Fz Tx Ty Tz
as int32 *counts*. Divide forces by counts_per_force and torques by counts_per_torque (both are
configured in the Net box - read them off its web page / Telnet and pass them in; 1e6 is the
common default but VERIFY yours, or the numbers are just scaled wrong).

NOTE (frame / calibration): the wrench is in the SENSOR's own frame, and includes the weight of
whatever hangs below the sensor (the gripper + anything grasped). Bias/tare removes it only for
the pose held at tare time; a pose-dependent tool-gravity compensation and a transform to the
base/EE frame are deliberately left for later (per your note). Raw + tared is what this returns.
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np

RDT_PORT = 49152
_REQ = struct.Struct(">HHI")        # header 0x1234, command, sample count
_REC = struct.Struct(">IIIiiiiii")  # rdt_seq, ft_seq, status, Fx,Fy,Fz,Tx,Ty,Tz
_CMD_START_STREAM = 0x0002
_CMD_STOP = 0x0000


class StubFTSensor:
    """No hardware: always reads zero. Keeps the driver script identical with/without a sensor."""
    def __init__(self, *a, **k):
        self.bias = np.zeros(6)
        self.status = 0
        self.connected = True   # stub is 'connected' (deliberate zeros, not missing data)

    def start(self): pass
    def stop(self): pass
    def tare(self, n=0): return self.bias.copy()
    def read(self): return np.zeros(6)


class NetFTSensor:
    def __init__(self, ip: str, port: int = RDT_PORT,
                 counts_per_force: float = 1_000_000.0,
                 counts_per_torque: float = 1_000_000.0,
                 timeout_s: float = 0.5):
        self.ip = ip
        self.port = int(port)
        self.cpf = float(counts_per_force)
        self.cpt = float(counts_per_torque)
        self.timeout_s = float(timeout_s)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = np.zeros(6)        # raw (pre-bias) wrench
        self.bias = np.zeros(6)
        self.status = 0
        self.connected = False
        self.n_recv = 0

    # ---- lifecycle -----------------------------------------------------------------------
    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self.timeout_s)
        self._send(_CMD_START_STREAM, 0)          # 0 samples = continuous
        self._thread = threading.Thread(target=self._recv_loop, name="ft-recv", daemon=True)
        self._thread.start()
        # wait briefly for the first record so read() is meaningful right away
        t0 = time.time()
        while not self.connected and time.time() - t0 < 1.5:
            time.sleep(0.02)
        if not self.connected:
            raise ConnectionError(
                f"no RDT data from Net F/T at {self.ip}:{self.port} within 1.5 s - "
                "check the IP, that the box is powered/linked, and firewall/UDP {self.port}.")

    def stop(self):
        self._stop.set()
        try:
            if self._sock is not None:
                self._send(_CMD_STOP, 0)
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    # ---- reads ---------------------------------------------------------------------------
    def read(self) -> np.ndarray:
        with self._lock:
            return self._latest - self.bias

    def tare(self, n: int = 200) -> np.ndarray:
        acc = np.zeros(6); got = 0; t0 = time.time()
        while got < n and time.time() - t0 < 5.0:
            with self._lock:
                acc += self._latest; got += 1
            time.sleep(0.002)
        self.bias = acc / max(got, 1)
        return self.bias.copy()

    # ---- internals -----------------------------------------------------------------------
    def _send(self, command: int, count: int):
        self._sock.sendto(_REQ.pack(0x1234, command, count), (self.ip, self.port))

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) < _REC.size:
                continue
            rec = _REC.unpack(data[:_REC.size])
            w = np.array([rec[3] / self.cpf, rec[4] / self.cpf, rec[5] / self.cpf,
                          rec[6] / self.cpt, rec[7] / self.cpt, rec[8] / self.cpt])
            with self._lock:
                self._latest = w
                self.status = rec[2]
                self.n_recv += 1
                self.connected = True


def make_sensor(kind: str, **kw):
    """kind: 'net' -> NetFTSensor(**kw), anything else -> StubFTSensor()."""
    return NetFTSensor(**kw) if kind == "net" else StubFTSensor()
