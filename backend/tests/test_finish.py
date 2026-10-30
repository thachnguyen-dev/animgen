import numpy as np

from animgen.finish import frame_prompt, outline_limit, paste_head
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


def test_frames_without_an_open_hand_do_not_ask_for_one():
    a7 = "casting spell, arm outstretched forward, open palm, other hand clenched"
    assert frame_prompt(a7, ("l",)) == a7
    assert frame_prompt(a7, ()) == "casting spell, arm outstretched forward, other hand clenched"
    assert frame_prompt("walking", ()) == "walking, clenched fists"


def test_repaint_keeps_the_frame_outline_tightly_around_the_hands():
    alpha = np.zeros((100, 100), np.uint8)
    alpha[20:80, 20:80] = 255
    keep = np.zeros((100, 100), bool)
    keep[20:30, 70:80] = True                                     # a hand at the corner
    lim = outline_limit(alpha, keep)
    assert lim[50, 50] == 255 and lim[50, 81] == 255              # inside, and a little past the outline elsewhere
    assert lim[50, 95] == 0                                       # not far past it
    assert lim[25, 81] == 0                                       # nothing past it beside the hand
