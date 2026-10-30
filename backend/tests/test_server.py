"""End-to-end through the HTTP server, as the WPF app uses it (§14: test every backend result over HTTP)."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import synth

TOKEN = "test-token-123"


@pytest.fixture
def server(tmp_path):
    env = dict(os.environ, ANIMGEN_TOKEN=TOKEN, PYTHONPATH=str(Path(__file__).parent.parent))
    p = subprocess.Popen([sys.executable, "-m", "animgen", "--engine", "demo", "--data", str(tmp_path / "data"),
                          "--parent-pid", str(os.getpid())],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, encoding="utf-8")
    line = p.stdout.readline().strip()
    assert line.startswith("READY "), line + p.stderr.read()
    port = int(line.split()[1])
    yield f"http://127.0.0.1:{port}"
    try:
        call(f"http://127.0.0.1:{port}", "POST", "/shutdown", {})
    except Exception:
        pass
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()


def call(base, method, path, body=None, token=TOKEN):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token is not None:
        req.add_header("X-AnimGen-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def wait_task(base, tid, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        _, t = call(base, "GET", f"/tasks/{tid}")
        if t["state"] not in ("queued", "running"):
            return t
        time.sleep(0.05)
    raise TimeoutError


def test_token_required(server):
    assert call(server, "GET", "/health", token=None)[0] == 401
    assert call(server, "GET", "/health", token="wrong")[0] == 401
    st, h = call(server, "GET", "/health")
    assert st == 200 and h["demo"] is True


def test_full_flow(server, tmp_path):
    hero = synth.character(tmp_path / "hero.png")
    st, anims = call(server, "GET", "/animations")
    assert st == 200 and {a["id"] for a in anims if a["ready"]} == {"idle", "walk", "run", "a1a", "a1b", "a2",
                                                                      "a5", "a6", "a7"}
    assert {a["id"]: a["grip"] for a in anims if a["ready"]}["walk"] == "optional"
    st, r = call(server, "POST", "/character", {"path": str(hero), "name": "Hiệp sĩ"})
    assert st == 200 and r["character"]["facing_source"] == "assumed"
    skel = r["character"]["skeleton"]
    assert not skel["confirmed"] and Path(skel["image"]).is_file() and len(skel["points"]) == 18

    # the app asks for the skeleton to be confirmed before anything is generated
    st, err = call(server, "POST", "/generate", {"animation": "idle"})
    assert st == 400 and err["error"]["code"] == "skeleton_unconfirmed"
    st, r = call(server, "POST", "/character/skeleton", {"points": skel["points"]})
    assert st == 200 and r["character"]["skeleton"]["confirmed"]

    # a bad request fails at once
    st, err = call(server, "POST", "/generate", {"animation": "walk", "grip": "yes"})
    assert st == 400 and err["error"]["code"] == "bad_request"

    # then the character's rig: the AI draws it once, and every frame is made from it
    st, err = call(server, "POST", "/generate", {"animation": "a2"})
    assert st == 400 and err["error"]["code"] == "rig_missing"
    st, r = call(server, "POST", "/rig", {"seed": 5})
    task = wait_task(server, r["task"]["id"])
    assert task["state"] == "completed", task
    rig = task["result"]["rig"]
    assert rig["seed"] == 5 and Path(rig["view"]).is_file() and Path(rig["hand_view"]).is_file()
    for part in rig["hanging"]:                         # the owner can stop a hanging part from swinging
        st, state = call(server, "POST", "/rig/hanging", {"name": part["name"], "on": False})
        assert st == 200 and not [h for h in state["rig"]["hanging"] if h["name"] == part["name"]][0]["on"]
    st, err = call(server, "POST", "/rig/hanging", {"name": "tail", "on": False})
    assert st == 400 and err["error"]["code"] == "unknown_part"

    st, r = call(server, "POST", "/generate", {"animation": "a2"})
    assert st == 200
    # one task at a time
    st2, busy = call(server, "POST", "/generate", {"animation": "walk"})
    task = wait_task(server, r["task"]["id"])
    assert task["state"] == "completed", task
    assert st2 == 200 or busy["error"]["code"] == "busy"
    st, state = call(server, "GET", "/state")
    assert state["complete"] and all(Path(f["preview"]).is_file() for f in state["frames"])
    assert state["grip"] is True and state["animation"]["grip"] == "always" and state["finished"] is False
    # rig frames come out the same every time: only finished frames can be redrawn
    st, err = call(server, "POST", "/regenerate", {"frame": 3})
    assert st == 400 and err["error"]["code"] == "nothing_to_redraw"
    st, err = call(server, "POST", "/generate", {"animation": "a2", "finish": "yes"})
    assert st == 400 and err["error"]["code"] == "bad_request"
    st, r = call(server, "POST", "/generate", {"animation": "a2", "finish": True})
    assert wait_task(server, r["task"]["id"])["state"] == "completed"
    _, state = call(server, "GET", "/state")
    assert state["complete"] and state["finished"] is True

    st, r = call(server, "POST", "/regenerate", {"frame": 3})
    wait_task(server, r["task"]["id"])
    _, state = call(server, "GET", "/state")
    assert state["frames"][2]["candidate_status"] == "ok"
    st, state = call(server, "POST", "/accept", {"frame": 3})
    assert st == 200 and state["frames"][2]["candidate_status"] == "none"

    st, err = call(server, "POST", "/export", {"folder": str(tmp_path / "out"), "height": 300})
    assert st == 400 and err["error"]["code"] == "export_bad_size"
    st, r = call(server, "POST", "/export", {"folder": str(tmp_path / "out"), "height": 256})
    task = wait_task(server, r["task"]["id"])
    files = task["result"]["files"]     # 8 frames, the sheet and the weapon hand positions
    assert task["state"] == "completed" and len(files) == 10 and files[-1].endswith("_weapon.json")


def test_bad_json_and_unknown_route(server):
    req = urllib.request.Request(server + "/character", data=b"{nope", method="POST")
    req.add_header("X-AnimGen-Token", TOKEN)
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400
    assert call(server, "GET", "/nothing")[0] == 404


def test_chunked_body_refused(server):
    # The WPF client once sent chunked JSON, which was read as an empty body ("missing path").
    req = urllib.request.Request(server + "/character", data=iter([b'{"path": "x.png"}']), method="POST")
    req.add_header("X-AnimGen-Token", TOKEN)
    req.add_header("Transfer-Encoding", "chunked")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 411


def test_input_error_over_http(server, tmp_path):
    p = synth.checkerboard_character(tmp_path / "cb.png")
    st, err = call(server, "POST", "/character", {"path": str(p)})
    assert st == 400 and err["error"]["code"] == "input_checkerboard"
