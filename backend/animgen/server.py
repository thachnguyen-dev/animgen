"""Loopback HTTP server between the WPF app and the backend (§9.1, §14 "Backend process").

- Bound to 127.0.0.1 on a free port; prints ``READY <port>`` on stdout once listening.
- Every request must carry ``X-AnimGen-Token`` equal to the per-run token (passed in ANIMGEN_TOKEN or --token).
- Long work (generate, regenerate, export, model loading) runs as a task; the app polls ``GET /tasks/<id>``.
- Replies are JSON built only from plain Python types (numpy values are converted first: a NumPy boolean once
  broke a reply in chargen).
- The backend exits when the app's process disappears (--parent-pid), in addition to the app's job object.

Routes (all JSON):
  GET  /health                         -> {ok, engine, demo, version}
  GET  /animations                     -> [{id, name_vi, ready, grip, weapon_kinds, status}]
  GET  /state                          -> session state (frames, previews, candidates, ...)
  POST /character {path, facing?, name?}
  POST /character/angle {angle}       -> {character, state}  ("side" | "34": which pose set the character uses)
  POST /character/skeleton {points}
  POST /rig {seed?, part?}             -> {task}   (the part map from the owner's image and the AI's open hand;
                                        part 'parts': the parts found again, 'hand': the hand drawn again)
  POST /rig/rebuild                    -> {task}   (the AI paints what each part of the part map hides)
  POST /rig/map                        -> state    (back to the part map, the painted parts dropped)
  POST /rig/edit                       -> {file, state}  (the Photoshop file of the stage, to correct)
  POST /rig/reload                     -> state    (the rig rebuilt from the saved Photoshop file)
  POST /rig/hanging {name, on: bool}   -> state    (a cape, skirt or hair swings or stays as drawn)
  POST /generate {animation, grip?: bool, seed?, finish?: bool} -> {task}
                                       (grip: hands closed, where the animation leaves it open;
                                        finish: the AI finishing pass, off by default)
  POST /regenerate {frame?: 1..8}      -> {task}
  POST /accept {frame}   POST /keep {frame}   POST /accept_all
  POST /export {folder, height}        -> {task}
  GET  /tasks/<id>       POST /tasks/<id>/cancel
  POST /settings {auto_retries}
  POST /shutdown
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__, config
from .errors import AppError
from .session import Session
from .tasks import TaskRunner
from .templates import available


def plain(value: Any) -> Any:
    """Convert to JSON-safe plain types (numpy scalars and arrays, paths, tuples, NaN)."""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class App:
    def __init__(self, session: Session, token: str) -> None:
        self.session = session
        self.token = token
        self.runner = TaskRunner()
        self.shutdown_requested = threading.Event()

    # Each handler returns (status, body)
    def route(self, method: str, path: str, body: dict) -> tuple[int, Any]:
        s = self.session
        parts = [p for p in path.split("?")[0].split("/") if p]
        if method == "GET":
            if parts == ["health"]:
                return 200, {"ok": True, "engine": s.engine.name, "demo": s.engine.is_demo, "version": __version__}
            if parts == ["animations"]:
                return 200, available(s.character.angle if s.character else "side")
            if parts == ["state"]:
                return 200, s.state()
            if len(parts) == 2 and parts[0] == "tasks":
                t = self.runner.get(parts[1])
                return (404, {"error": {"code": "not_found", "message": "task"}}) if t is None else (200, t.to_json())
        if method == "POST":
            busy_guard = parts[:1] in (["character"], ["accept"], ["keep"], ["accept_all"], ["settings"], ["rig"])
            if busy_guard and self.runner.busy():
                raise AppError("busy")
            if parts == ["character"]:
                ch = s.set_character(_req(body, "path", str), body.get("facing"), body.get("name"))
                return 200, {"character": ch.to_json(), "state": s.state()}
            if parts == ["character", "skeleton"]:
                ch = s.confirm_skeleton(_req(body, "points", list))
                return 200, {"character": ch.to_json(), "state": s.state()}
            if parts == ["character", "angle"]:
                ch = s.set_angle(_req(body, "angle", str))
                return 200, {"character": ch.to_json(), "state": s.state()}
            if parts == ["rig"]:
                if s.character is None:
                    raise AppError("no_character")
                if s.require_confirmed_skeleton and not s.character.skeleton_confirmed:
                    raise AppError("skeleton_unconfirmed")
                seed = body.get("seed")
                part = body.get("part")
                if part not in (None, "parts", "hand"):
                    raise AppError("unknown_part")
                if part is not None and s.rig is None:
                    raise AppError("rig_missing")
                t = self.runner.start("rig", lambda task: s.make_rig(task, int(seed) if seed is not None else None,
                                                                     part))
                return 200, {"task": t.to_json()}
            if parts == ["rig", "rebuild"]:
                if s.rig is None:
                    raise AppError("rig_missing")
                if s.rig.stage != "map":
                    raise AppError("rig_map_missing")
                t = self.runner.start("rig", s.rebuild_hidden)
                return 200, {"task": t.to_json()}
            if parts == ["rig", "map"]:
                return 200, s.back_to_map()
            if parts == ["rig", "edit"]:
                return 200, s.rig_edit_file()
            if parts == ["rig", "reload"]:
                return 200, s.reload_rig_edits()
            if parts == ["rig", "hanging"]:
                on = body.get("on")
                if not isinstance(on, bool):
                    raise AppError("bad_request", detail="on must be true or false")
                return 200, s.set_hanging(_req(body, "name", str), on)
            if parts == ["generate"]:
                anim = _req(body, "animation", str)
                grip = body.get("grip")
                if grip is not None and not isinstance(grip, bool):
                    raise AppError("bad_request", detail="grip must be true or false")
                finish = body.get("finish", False)
                if not isinstance(finish, bool):
                    raise AppError("bad_request", detail="finish must be true or false")
                seed = body.get("seed")
                s.prepare(anim)  # fail fast on a bad choice, before a task starts
                if s.require_rig and s.rig is None:
                    raise AppError("rig_missing")
                t = self.runner.start("generate", lambda task: s.generate(
                    task, anim, grip, int(seed) if seed is not None else None, finish=finish))
                return 200, {"task": t.to_json()}
            if parts == ["regenerate"]:
                frame = body.get("frame")
                if frame is not None:
                    s._index(int(frame))
                if s.template is None:
                    raise AppError("no_frames")
                if s.rig is not None and not s.finished:
                    raise AppError("nothing_to_redraw")
                t = self.runner.start("regenerate",
                                      lambda task: s.regenerate(task, int(frame) if frame is not None else None))
                return 200, {"task": t.to_json()}
            if parts == ["accept"]:
                return 200, s.accept(int(_req(body, "frame", int)))
            if parts == ["keep"]:
                return 200, s.keep(int(_req(body, "frame", int)))
            if parts == ["accept_all"]:
                return 200, s.accept_all()
            if parts == ["export"]:
                folder = _req(body, "folder", str)
                height = int(_req(body, "height", int))
                if height not in config.EXPORT_HEIGHTS:
                    raise AppError("export_bad_size")
                if s.template is None or any(x.status != "ok" for x in s.slots):
                    raise AppError("export_incomplete")
                t = self.runner.start("export", lambda task: s.export(task, folder, height))
                return 200, {"task": t.to_json()}
            if len(parts) == 3 and parts[0] == "tasks" and parts[2] == "cancel":
                return 200, {"ok": self.runner.cancel(parts[1])}
            if parts == ["settings"]:
                if "auto_retries" in body:
                    s.auto_retries = max(0, min(int(body["auto_retries"]), config.MAX_AUTO_RETRIES))
                return 200, s.state()
            if parts == ["shutdown"]:
                self.shutdown_requested.set()
                return 200, {"ok": True}
        return 404, {"error": {"code": "not_found", "message": path}}


def _req(body: dict, key: str, typ: type) -> Any:
    if key not in body or body[key] is None:
        raise AppError("bad_request", detail=f"missing {key}")
    v = body[key]
    if typ is int:
        try:
            return int(v)
        except (TypeError, ValueError) as exc:
            raise AppError("bad_request", detail=f"{key} must be a number") from exc
    if not isinstance(v, typ):
        raise AppError("bad_request", detail=f"{key} has the wrong type")
    return v


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "AnimGen"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet; errors still go to stderr
            pass

        def _send(self, status: int, body: Any) -> None:
            data = json.dumps(plain(body), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, method: str) -> None:
            token = self.headers.get("X-AnimGen-Token", "")
            if not hmac.compare_digest(token.encode(), app.token.encode()):
                self._send(401, {"error": {"code": "unauthorized", "message": "token"}})
                return
            body: dict = {}
            if method == "POST":
                if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                    # Only sized bodies are read; a chunked one would be ignored and break the connection.
                    self.close_connection = True
                    self._send(411, {"error": AppError("bad_request", detail="length required").to_json()})
                    return
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    try:
                        body = json.loads(self.rfile.read(n).decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        self._send(400, {"error": AppError("bad_request", detail="json").to_json()})
                        return
                    if not isinstance(body, dict):
                        self._send(400, {"error": AppError("bad_request", detail="json").to_json()})
                        return
            try:
                status, out = app.route(method, self.path, body)
            except AppError as exc:
                status, out = 400, {"error": exc.to_json()}
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                status, out = 500, {"error": AppError("inference_failed", detail=f"{type(exc).__name__}: {exc}")
                                    .to_json()}
            self._send(status, out)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

    return Handler


def parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if sys.platform == "win32":
        import ctypes
        SYNCHRONIZE, WAIT_TIMEOUT = 0x00100000, 0x102
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not h:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def make_engine(name: str, models_dir: Path):
    if name == "demo":
        from .engines.demo import DemoEngine
        return DemoEngine()
    if name == "diffusers":
        from .engines.diffusers_engine import DiffusersEngine
        return DiffusersEngine(models_dir)
    raise SystemExit(f"unknown engine {name}")


def serve(engine_name: str, data_dir: Path, models_dir: Path, token: str, port: int = 0, parent_pid: int = 0,
          ready_out=None) -> None:
    # Offline only (§2.4): no model hub access, ever.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    session = Session(make_engine(engine_name, models_dir), data_dir, require_confirmed_skeleton=True,
                      require_rig=True)
    app = App(session, token)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    httpd.daemon_threads = True
    actual = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    t.start()
    out = ready_out or sys.stdout
    print(f"READY {actual}", file=out, flush=True)
    try:
        while not app.shutdown_requested.is_set():
            if not parent_alive(parent_pid):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        session.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="animgen-backend")
    ap.add_argument("--engine", default=os.environ.get("ANIMGEN_ENGINE", "diffusers"), choices=["diffusers", "demo"])
    ap.add_argument("--data", default=os.environ.get("ANIMGEN_DATA", str(Path.cwd() / "data")))
    ap.add_argument("--models", default=None)
    ap.add_argument("--token", default=os.environ.get("ANIMGEN_TOKEN", ""))
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--parent-pid", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.token:
        raise SystemExit("a token is required (ANIMGEN_TOKEN or --token)")
    data = Path(a.data)
    models = Path(a.models) if a.models else data / "models"
    # Stdout carries the READY line; make it UTF-8 regardless of the Windows console code page.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except AttributeError:
        pass
    serve(a.engine, data, models, a.token, a.port, a.parent_pid)


if __name__ == "__main__":
    main()
