import cv2
import numpy as np
from PIL import Image, ImageDraw

from animgen.rig import build_rig
from animgen.skeleton import Skeleton

from test_puppet import STAND, figure

# The rig image: the same figure in the spread pose (near arm forward, far arm back, wide stride), drawn with its own
# colours so a test can tell which image a pixel came from.
SPREAD = list(STAND)
SPREAD[3], SPREAD[4] = (560, 450), (610, 520)           # near arm forward
SPREAD[6], SPREAD[7] = (440, 450), (400, 520)           # far arm back
SPREAD[9], SPREAD[10] = (560, 730), (600, 880)          # near leg forward
SPREAD[12], SPREAD[13] = (440, 730), (400, 880)         # far leg back
RIG_HEAD = (0, 0, 255, 255)
FAR_LEG = (250, 220, 0, 255)
FOREARM = (0, 200, 0, 255)


def rig_image() -> Image.Image:
    img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line([SPREAD[11], SPREAD[12], SPREAD[13]], fill=FAR_LEG, width=22)
    d.line([SPREAD[5], SPREAD[6], SPREAD[7]], fill=(160, 0, 160, 255), width=16)    # far arm: a cape-like stand-in
    d.line([SPREAD[8], SPREAD[9], SPREAD[10]], fill=(60, 60, 60, 255), width=22)
    d.line([SPREAD[1], (500, 580)], fill=(120, 60, 30, 255), width=40)
    d.ellipse((480, 260, 550, 340), fill=RIG_HEAD)
    d.line([SPREAD[2], SPREAD[3]], fill=(200, 0, 0, 255), width=16)
    d.line([SPREAD[3], SPREAD[4]], fill=FOREARM, width=14)
    return img


def rig():
    return build_rig(figure(), Skeleton(list(STAND)), rig_image(), Skeleton(list(SPREAD)))


def test_head_comes_from_the_original():
    out = np.array(rig().render(Skeleton(list(SPREAD))))
    px = out[300, 515]
    assert px[3] > 200 and px[2] < 200 and px[0] > 200            # skin, not the rig image's blue head


def test_far_leg_comes_from_the_rig_image():
    out = np.array(rig().render(Skeleton(list(SPREAD))))
    px = out[805, 420]
    assert px[3] > 200 and px[0] > 200 and px[1] > 180 and px[2] < 60


def test_far_arm_is_a_darker_copy_of_the_near_arm():
    out = np.array(rig().render(Skeleton(list(SPREAD))))
    px = out[485, 420]
    assert px[3] > 200 and 100 < px[1] < 200 and px[0] < 60 and px[2] < 60   # darkened green, not the purple stand-in


def test_forearm_follows_the_target():
    tgt = list(SPREAD)
    tgt[4] = (650, 450)                                 # near forearm raised level
    out = np.array(rig().render(Skeleton(tgt)))
    green = (out[..., 1] > 150) & (out[..., 0] < 80) & (out[..., 3] > 200)
    ys, xs = np.nonzero(green)
    assert xs.max() > 630 and abs(ys.mean() - 450) < 20


def test_head_turns_with_its_bone_and_keeps_its_size():
    r = rig()
    still = np.array(r.render(Skeleton(list(SPREAD))))
    tgt = list(SPREAD)
    neck, nose = np.array(SPREAD[1], float), np.array(SPREAD[0], float)
    ang = np.radians(20)                                # chin down 20 degrees
    v = nose - neck
    tgt[0] = tuple(neck + [v[0] * np.cos(ang) - v[1] * np.sin(ang), v[0] * np.sin(ang) + v[1] * np.cos(ang)])
    turned = np.array(r.render(Skeleton(tgt)))
    skin = lambda a: (a[..., 0] > 200) & (a[..., 1] > 150) & (a[..., 2] > 100) & (a[..., 3] > 200)   # noqa: E731
    assert (skin(still) != skin(turned)).sum() > 200              # it moved
    assert abs(int(skin(turned).sum()) - int(skin(still).sum())) < 0.05 * skin(still).sum()   # not stretched


def test_bent_knee_stays_in_one_piece():
    tgt = list(SPREAD)
    tgt[10] = (470, 800)                                # near lower leg folded back
    out = np.array(rig().render(Skeleton(tgt)))
    grey = ((np.abs(out[..., :3].astype(int) - 60).max(axis=-1) < 15) & (out[..., 3] > 200)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(grey, connectivity=8)
    assert sorted(stats[1:, cv2.CC_STAT_AREA])[-1] > 0.95 * stats[1:, cv2.CC_STAT_AREA].sum()


# The hand image: the near arm straight forward, with a magenta open hand past the wrist.
HAND_SK = list(SPREAD)
HAND_SK[3], HAND_SK[4] = (590, 365), (680, 365)
PALM = (255, 0, 255, 255)


def hand_image() -> Image.Image:
    img = rig_image()
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 1024, 1024), fill=(0, 0, 0, 0))
    d.line([(500, 365), HAND_SK[3]], fill=(200, 0, 0, 255), width=16)
    d.line([HAND_SK[3], HAND_SK[4]], fill=FOREARM, width=14)
    d.ellipse((672, 345, 712, 385), fill=PALM)
    return img


def open_rig():
    return build_rig(figure(), Skeleton(list(STAND)), rig_image(), Skeleton(list(SPREAD)), hand_image(),
                     Skeleton(list(HAND_SK)))


def magenta(a: np.ndarray, level: int = 200) -> np.ndarray:
    return (a[..., 0] > level) & (a[..., 2] > level) & (a[..., 1] < 80) & (a[..., 3] > 200)


def test_open_hand_only_where_the_frame_asks():
    r = open_rig()
    fist = np.array(r.render(Skeleton(list(SPREAD))))
    near = np.array(r.render(Skeleton(list(SPREAD)), open_hands=("r",)))
    far = np.array(r.render(Skeleton(list(SPREAD)), open_hands=("l",)))
    assert not magenta(fist).any()
    ys, xs = np.nonzero(magenta(near))
    assert len(xs) > 300 and np.hypot(xs.mean() - SPREAD[4][0], ys.mean() - SPREAD[4][1]) < 40   # on the near wrist
    ys, xs = np.nonzero(magenta(far, level=150))
    assert len(xs) > 300 and np.hypot(xs.mean() - SPREAD[7][0], ys.mean() - SPREAD[7][1]) < 40   # on the far wrist
    assert far[ys, xs, 0].max() < 230                             # darkened as far


def test_keep_mask_is_the_head_and_the_visible_hands():
    _, keep = open_rig().render_keep(Skeleton(list(SPREAD)), open_hands=("r",))
    assert keep[300, 515]                                         # the head
    assert keep[SPREAD[4][1] + 5, SPREAD[4][0] + 15]              # past the near wrist: the hand
    assert not keep[470, 500] and not keep[805, 420]              # the body, a leg
    assert not keep[SPREAD[3][1], SPREAD[3][0]]                   # the elbow
