import numpy as np
from PIL import Image, ImageDraw

from animgen.puppet import repose
from animgen.skeleton import LELB, LWRI, RELB, RWRI, Skeleton

# A standing side-view stick figure, canvas 1024 (OpenPose order, the face points right).
STAND = [(520, 300), (500, 360), (500, 365), (500, 470), (500, 560), (500, 365), (500, 470), (500, 560),
         (495, 580), (495, 730), (495, 880), (505, 580), (505, 730), (505, 880), (515, 290), (525, 290),
         (505, 295), (530, 295)]
FOREARM = (0, 200, 0, 255)


def figure() -> Image.Image:
    img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((480, 260, 550, 340), fill=(230, 180, 140, 255))              # head
    d.line([STAND[1], (500, 580)], fill=(120, 60, 30, 255), width=40)       # torso
    d.line([STAND[9 - 1], STAND[9], STAND[10]], fill=(60, 60, 60, 255), width=22)   # legs
    d.line([STAND[2], STAND[3]], fill=(200, 0, 0, 255), width=16)           # upper arm
    d.line([STAND[3], STAND[4]], fill=FOREARM, width=14)                    # forearm (green)
    return img


def test_same_pose_keeps_the_figure():
    ref, sk = figure(), Skeleton(list(STAND))
    out = repose(ref, sk, Skeleton(list(STAND)))
    a, b = np.array(ref)[..., 3] > 0, np.array(out)[..., 3] > 0
    assert (a & b).sum() / a.sum() > 0.95


def test_forearm_follows_the_target():
    ref, sk = figure(), Skeleton(list(STAND))
    tgt = list(STAND)
    tgt[RWRI] = tgt[LWRI] = (590, 470)                  # forearm raised forward, horizontal
    out = np.array(repose(ref, sk, Skeleton(tgt)))
    green = (out[..., 1] > 150) & (out[..., 0] < 80) & (out[..., 3] > 200)
    ys, xs = np.nonzero(green)
    assert xs.max() > 570 and abs(ys.mean() - 470) < 20          # moved to the new place
    assert not green[540:560, 490:510].any()                      # and gone from the old one


def test_far_arm_is_darker_copy_behind():
    ref, sk = figure(), Skeleton(list(STAND))
    tgt = list(STAND)
    tgt[LELB], tgt[LWRI] = (440, 450), (400, 520)       # far arm swung back, clear of the body
    out = np.array(repose(ref, sk, Skeleton(tgt)))
    px = out[505, 415]
    assert px[3] > 200 and 100 < px[1] < 170 and px[0] < 60       # green forearm at FAR_DARKEN
    assert repose(ref, sk, Skeleton([None] * 18)).size == ref.size  # no target bones: unchanged copy
