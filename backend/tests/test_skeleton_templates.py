import json
import math

import pytest

from animgen import authoring, config
from animgen.skeleton import EDGES, LANK, RANK, Skeleton, bone_length, draw_openpose, fit_frame, measure
from animgen.templates import ANIMATIONS, TemplateError, available, load_template, parse_template

READY = [a[0] for a in ANIMATIONS if a[4]]


def scaled_rest(h=600.0, foot=900.0, cx=500.0, stretch_legs=1.0):
    pts, _ = authoring.solve(authoring.Pose())
    out = []
    for i, p in enumerate(pts):
        if p is None:
            out.append(None)
            continue
        y = p[1] * h
        if stretch_legs != 1.0 and i in (9, 10, 12, 13):
            y = y * stretch_legs
        out.append((cx + p[0] * h, foot + y))
    return Skeleton(out)


@pytest.mark.parametrize("tid", READY)
def test_ready_templates_parse(tid):
    t = load_template(tid)
    assert len(t.frames) == config.FRAME_COUNT == 8
    if t.weapon != "never":
        assert all(f.weapon is not None for f in t.frames)


def test_bow_uses_three_draw_stages():
    t = load_template("a5")
    assert {f.weapon.stage for f in t.frames} == {1, 2, 3}


def test_unarmed_attacks_never_take_a_weapon():
    assert load_template("a1a").weapon == "never"
    assert load_template("a1b").weapon == "never"


def test_non_attack_templates_store_weapon_data():
    for tid in ("idle", "walk", "run"):
        assert load_template(tid).weapon == "optional"


def test_stored_templates_match_authoring_source():
    for tid in authoring.TEMPLATES:
        with open(f"animgen/template_data/{tid}.json", encoding="utf-8") as f:
            assert json.load(f) == json.loads(json.dumps(authoring.build(tid))), tid


def test_template_must_have_eight_frames():
    data = authoring.build("idle")
    data["frames"] = data["frames"][:7]
    with pytest.raises(TemplateError):
        parse_template(data)


def test_available_marks_cut_animations_not_ready():
    av = {a["id"]: a for a in available()}
    assert av["walk"]["ready"] and not av["jump"]["ready"] and not av["a3"]["ready"]


@pytest.mark.parametrize("tid", READY)
def test_fit_keeps_character_lengths_and_feet(tid):
    t = load_template(tid)
    ref = scaled_rest(stretch_legs=1.1)  # a character with longer legs than the template
    props = measure(ref, t.rest, 600.0, 900.0)
    anchor = (500.0, 900.0)
    pelvis_x = []
    for tf in t.frames:
        f = fit_frame(tf.skeleton, t.rest, props, anchor, tf.grounded)
        for a, b in EDGES:
            if f.points[b] is None:
                continue
            L = bone_length(f, a, b)
            assert L <= props.lengths[(a, b)] * 1.26 + 1e-6
        low = f.lowest_ankle_y()
        if tf.grounded:
            assert math.isclose(low, anchor[1] - props.ankle_height, abs_tol=1e-6)
        else:
            assert low < anchor[1] - props.ankle_height
        pelvis_x.append(f.pelvis()[0])
    # No unnecessary drift: the pelvis stays near the anchor (§6.3).
    assert max(abs(x - anchor[0]) for x in pelvis_x) < 0.05 * 600


def test_measure_uses_counterpart_for_unseen_far_limb():
    ref = scaled_rest()
    pts = list(ref.points)
    pts[12] = None  # far knee not seen
    pts[13] = None
    props = measure(Skeleton(pts), load_template("idle").rest, 600, 900)
    assert math.isclose(props.lengths[(11, 12)], props.lengths[(8, 9)], rel_tol=1e-6)


def test_draw_openpose_draws_something():
    img = draw_openpose(scaled_rest(), (1024, 1024))
    assert img.shape == (1024, 1024, 3) and img.sum() > 0


def test_facing():
    s = scaled_rest()
    assert s.facing() == "right"
    assert s.mirrored(1000).facing() == "left"
