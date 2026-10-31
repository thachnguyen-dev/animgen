"""Rough reposes of the test character for some template frames (no GPU). usage: python puppet_test.py <char> <out.png> anim:frame..."""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, r"C:\AnimGen\project\animgen\backend")
sys.path.insert(0, str(Path(__file__).parent))
from animgen.server import make_engine  # noqa: E402
from animgen.session import Session  # noqa: E402
from animgen.skeleton import draw_openpose, fit_frame, measure  # noqa: E402
from animgen.templates import load_template  # noqa: E402
from animgen.puppet import repose  # noqa: E402

char, out = sys.argv[1], sys.argv[2]
engine = make_engine("diffusers", Path(r"C:\AnimGen\models"))
s = Session(engine, Path(out).parent / "puppet_work")
ch = s.set_character(char, None)
print("facing:", ch.facing, ch.facing_source if hasattr(ch, "facing_source") else "")
print("ref skeleton:", ch.skeleton.to_json()[:14])
cells = []
for spec in sys.argv[3:]:
    anim, f = spec.split(":")
    tpl = load_template(anim)
    props = measure(ch.skeleton, tpl.rest, ch.placement.figure_height, ch.placement.anchor[1])
    tf = tpl.frames[int(f) - 1]
    tgt = fit_frame(tf.skeleton, tpl.rest, props, ch.placement.anchor, tf.grounded)
    rough = repose(ch.placement.image, ch.skeleton, tgt)
    pose = Image.fromarray(draw_openpose(tgt, rough.size)).convert("RGBA")
    cell = Image.new("RGBA", rough.size, (255, 255, 255, 255))
    cell.alpha_composite(Image.blend(Image.new("RGBA", rough.size, (255, 255, 255, 255)), pose, 0.35))
    cell.alpha_composite(rough)
    cells.append(cell.crop((250, 250, 800, 1000)))
sheet = Image.new("RGB", (550 * len(cells), 750), "white")
for i, c in enumerate(cells):
    sheet.paste(c.convert("RGB"), (i * 550, 0))
sheet.save(out)
