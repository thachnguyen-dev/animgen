import numpy as np

from animgen.finish import paste_head
from animgen.templates import load_template


def test_head_pasted_over_the_repainted_frame():
    out = np.zeros((64, 64, 4), np.uint8)
    out[..., 1] = 200
    out[..., 3] = 255
    head = np.zeros((64, 64, 4), np.uint8)
    head[16:48, 16:48] = (250, 10, 10, 255)
    res = paste_head(out, head)
    assert tuple(res[32, 32, :3]) == (250, 10, 10)               # the owner's head inside
    assert tuple(res[2, 2, :3]) == (0, 200, 0)                   # the repainted frame outside
    assert 10 < res[32, 17, 0] < 250                              # a soft edge


def test_cast_opens_the_far_hand_in_the_reach_frames():
    tpl = load_template("a7")
    assert [f.open_hands for f in tpl.frames] == [(), (), (), ("l",), ("l",), ("l",), (), ()]
    assert all(f.open_hands == () for f in load_template("a1a").frames)
