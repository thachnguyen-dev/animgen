"""A/B variants of the real engine on one animation, one model load.
usage: python ab_run.py <character.png> <animation> <out_dir> <variant>...
variant = name:ip=<scale>:cn=<scale>:str=<strength>:steps=<n>:style=<extra positive tags, '+' for spaces/commas>
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\Game\ainm\animgen\backend")
import animgen.engines.diffusers_engine as de  # noqa: E402
from animgen.server import make_engine  # noqa: E402
from animgen.session import Session  # noqa: E402
from animgen.tasks import TaskRunner  # noqa: E402

char, anim, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
base_positive = de.POSITIVE
DEFAULTS = {"steps": de.STEPS, "cn": de.CONTROL_SCALE, "ip": de.IP_ADAPTER_SCALE, "str": de.STRENGTH}
engine = make_engine("diffusers", Path(r"C:\AnimGen\models"))
RAW = {"dir": out, "n": 0}


class SavingPipe:
    """Wraps the pipeline: saves each raw two-panel output and its pose guide before cropping."""
    def __init__(self, pipe):
        self._pipe = pipe

    def __getattr__(self, k):
        return getattr(self._pipe, k)

    def __call__(self, **kw):
        res = self._pipe(**kw)
        RAW["n"] += 1
        d = RAW["dir"] / "raw"
        d.mkdir(exist_ok=True)
        res.images[0].save(d / f"{RAW['tag']}_{RAW['n']:02d}_out.png")
        kw["control_image"][0].save(d / f"{RAW['tag']}_{RAW['n']:02d}_pose.png")
        return res


_orig_load = engine.drawer.load


def _load(progress):
    _orig_load(progress)
    engine.drawer.pipe = SavingPipe(engine.drawer.pipe)


engine.drawer.load = _load
runner = TaskRunner()
results = []
for spec in sys.argv[4:]:
    name, *opts = spec.split(":")
    o = dict(x.split("=", 1) for x in opts)
    de.STEPS = int(o.get("steps", DEFAULTS["steps"]))
    de.CONTROL_SCALE = float(o.get("cn", DEFAULTS["cn"]))
    de.IP_ADAPTER_SCALE = float(o.get("ip", DEFAULTS["ip"]))
    de.STRENGTH = float(o.get("str", DEFAULTS["str"]))
    style = o.get("style", "").replace("+", " ").replace("|", ", ")
    de.POSITIVE = base_positive.replace("{motion}, ", "{motion}, " + (style + ", " if style else ""))
    RAW["tag"], RAW["n"] = name, 0
    s = Session(engine, out / f"work_{name}")
    s.set_character(char, None)
    if engine.drawer.pipe is not None:
        engine.drawer.pipe.set_ip_adapter_scale(de.IP_ADAPTER_SCALE)
    t0 = time.time()
    weapon = None
    if anim == "a2":  # the temporary sword (draw_weapons.py)
        from animgen.session import WeaponChoice
        w = s.weapons.add("ab-sword", "one_handed", "a", [str(Path(char).parent / "test_sword.png")])
        s.weapons.set_points(w.id, "a", 1, (70, 35), (548, 35))
        weapon = WeaponChoice(w.id, "a")
    task = runner.wait(runner.start(anim, lambda t: s.generate(t, anim, weapon, 1234)).id, timeout=3 * 3600)
    if engine.drawer.pipe is not None:
        engine.drawer.pipe.set_ip_adapter_scale(de.IP_ADAPTER_SCALE)
    st = s.state()
    r = {"variant": spec, "state": task.state, "error": task.error, "seconds": round(time.time() - t0),
         "frames": [f["preview"] for f in st["frames"]]}
    results.append(r)
    print(json.dumps({k: r[k] for k in ("variant", "state", "seconds")}), flush=True)
json.dump(results, open(out / "ab.json", "w"), indent=1)
