"""Local control server for the balanced-drag session (drag --serve).

A sandboxed claude.ai artifact cannot reach this process, so live control needs the page
served from here over localhost. This starts a tiny stdlib HTTP server in a daemon thread:

  GET  /                serves balance_panel.html (the Balance Console); when it detects it is
                        being served from localhost it wires its sliders/checkboxes to /set
  GET  /set?p=bal&v=1.5           -> push "bal 1.5"  into the controller's command queue
  GET  /set?p=fric&v=0.5          -> push "fric 0.5"
  GET  /set?p=sus&j=2&v=on        -> push "sus 2 on"
  GET  /state                     -> JSON of the live values (so the page reflects reality)

Every /set turns into the same queued command a typed key would, so the control loop applies
it and prints it on the CLI. Nothing here touches the arm directly - it only enqueues.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs


def start(ctrl, html_path: str | Path, host: str = "127.0.0.1", port: int = 8730):
    """Start the control server for `ctrl` (a GravityDragController with a BalancedDrag assist).
    Returns the HTTPServer (already serving in a daemon thread)."""
    html_path = Path(html_path)
    a = ctrl.assist

    def _arr(x):
        return None if x is None else [float(v) for v in x]

    def pose():
        q = getattr(ctrl, "q_now", None)
        tel = getattr(ctrl, "telem", None) or {}
        temp = tel.get("temp")
        return {
            "q": _arr(q),
            "phase": getattr(getattr(ctrl, "phase", None), "value", None),
            "v": _arr(tel.get("v")),            # velocity (finite-difference, filtered) - rad/s
            "tau": _arr(tel.get("tau")),        # commanded joint torque (what we send) - N.m
            "r": _arr(tel.get("r")),            # observer estimate of external/hand torque - N.m
            "temp": (None if temp is None or temp != temp else float(temp)),  # NaN -> None
        }

    def state():
        s = {
            "kappa": float(getattr(a, "kappa", 0.0)),
            "shaping_on": bool(getattr(a, "shaping_on", False)),
            "fric_scale": float(getattr(a, "fric_scale", 0.0)),
            "fric_on": bool(getattr(a, "fric_on", False)),
            "sustain": [bool(x) for x in getattr(a, "sustain", [])],
            "lam": _arr(getattr(a, "lam_d", None)),          # live Cartesian inertia (actual, slewed)
            "lam_goal": _arr(getattr(a, "_lam_goal", None)),  # its target (what the sliders set)
            "n": int(getattr(a, "n", 6)),
            # paper-feature ablations (all opt-in; 0 = off)
            "detent_kp": float(getattr(a, "detent_kp", 0.0)),
            "damp_t": float(getattr(a, "damp_t", 0.0)),
            "damp_r": float(getattr(a, "damp_r", 0.0)),
            "break_beta": float(getattr(a, "break_beta", 0.0)),
            "alpha_sigma_v": float(getattr(a, "alpha_sigma_v", 0.0)),
            "alpha_kappa0": float(getattr(a, "alpha_kappa0", 0.0)),
            "fric_model": str(getattr(a, "fric_model", "")),
            "fric_model_names": sorted(getattr(a, "fric_models", {}) or []),
            "mu_on": bool(getattr(a, "mu_on", True)),
            "visc_on": bool(getattr(a, "visc_on", True)),
        }
        s.update(pose())
        return s

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):        # keep the drag CLI clean
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                try:
                    self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                except OSError:
                    self._send(404, b"balance_panel.html not found", "text/plain")
                return
            if u.path == "/state":
                self._send(200, json.dumps(state()).encode(), "application/json")
                return
            if u.path == "/pose":                 # lightweight, polled fast for live rendering
                self._send(200, json.dumps(pose()).encode(), "application/json")
                return
            if u.path == "/set":
                q = parse_qs(u.query)
                p = (q.get("p", [""])[0]).strip()
                v = (q.get("v", [""])[0]).strip()
                idx = int((q.get("i", q.get("j", ["0"])) )[0])   # shared index param (i, or legacy j)
                cmd = None
                if p == "bal":
                    cmd = f"bal {float(v)}"
                elif p == "fric":
                    cmd = f"fric {float(v)}"
                elif p == "sus":
                    cmd = f"sus {idx} {'on' if v in ('on', '1', 'true') else 'off'}"
                elif p == "lam":
                    cmd = f"lam {idx} {float(v)}"
                elif p in ("detent", "damp_t", "damp_r", "break", "alphav", "alphas"):
                    cmd = f"{p} {float(v)}"       # single-scalar ablation knobs
                elif p == "fmodel" and v.replace("_", "").isalnum():
                    cmd = f"fmodel {v}"           # kinetic friction model A/B toggle
                elif p in ("mu", "visc"):
                    cmd = f"{p} {'on' if v in ('on', '1', 'true') else 'off'}"   # per-term ablation
                if cmd is None:
                    self._send(400, b'{"ok":false}', "application/json")
                    return
                ctrl._cmds.put(cmd)              # same path as a typed key -> loop applies + prints
                self._send(200, json.dumps({"ok": True, "cmd": cmd}).encode(), "application/json")
                return
            self._send(404, b"not found", "text/plain")

    srv = ThreadingHTTPServer((host, port), H)
    threading.Thread(target=srv.serve_forever, name="balance-web", daemon=True).start()
    return srv
