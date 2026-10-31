"""The three-quarter view pose set (owner, 2026-10-31): authoring34.py, the character's angle, per-frame order."""

import json
import math
from types import SimpleNamespace

import pytest

from animgen import authoring, authoring34, config
from animgen.rig import Rig
from animgen.session import guess_angle
from animgen.skeleton import (EDGES, LEYE, LSHO, RSHO, Skeleton, bone_length, fit_frame, measure)
from animgen.tasks import TaskRunner
from animgen.templates import ANIMATIONS, load_template

READY = [a[0] for a in ANIMATIONS if a[4]]


def run(fn, timeout=120):
    r = TaskRunner()
    return r.wait(r.start("t", fn).id, timeout)


def placed(points, h=600.0, foot=900.0, cx=500.0):
    return Skeleton([None if p is None else (cx + p[0] * h, foot + p[1] * h) for p in points])


@pytest.mark.parametrize("tid", READY)
def test_every_animation_has_a_three_quarter_template(tid):
    t = load_template(tid, angle="34")
    assert len(t.frames) == config.FRAME_COUNT
    assert t.rest.points[LEYE] is not None                           # both eyes show
    assert abs(t.rest.points[RSHO][0] - t.rest.points[LSHO][0]) > 0.08   # the shoulders apart across the picture
    side = load_template(tid)
    assert (t.grip, t.weapon_kinds, t.name_vi) == (side.grip, side.weapon_kinds, side.name_vi)


def test_stored_three_quarter_templates_match_authoring_source():
    for tid in authoring.TEMPLATES:
        with open(f"animgen/template_data/view34/{tid}.json", encoding="utf-8") as f:
            assert json.load(f) == json.loads(json.dumps(authoring34.build(tid))), tid


@pytest.mark.parametrize("tid", READY)
def test_three_quarter_fit_keeps_lengths_and_does_not_drift(tid):
    t = load_template(tid, angle="34")
    ref = placed(authoring34.rest_points())
    props = measure(ref, t.rest, 600.0, 900.0)
    anchor = (500.0, 900.0)
    xs = []
    for tf in t.frames:
        f = fit_frame(tf.skeleton, t.rest, props, anchor, tf.grounded)
        for a, b in EDGES:
            if f.points[b] is not None and (a, b) in props.lengths:
                assert bone_length(f, a, b) <= props.lengths[(a, b)] * 1.26 + 1e-6
        xs.append(f.pelvis()[0])
    assert max(abs(x - anchor[0]) for x in xs) < 0.05 * 600              # §6.3


def test_walk_turns_hips_and_shoulders_against_each_other_and_steps_the_far_leg_in_front():
    poses = authoring34.poses34("walk")
    turns = [authoring34.turns(p) for p in poses]
    assert any(h > 3 for _, h in turns) and any(h < -3 for _, h in turns)         # the hips turn with the legs
    assert all(c * h <= 0 for c, h in turns if abs(h) > 3)                       # the shoulders against them
    fronts = ["leg_far" in fr.front for fr in load_template("walk", angle="34").frames]
    assert any(fronts) and not all(fronts)


def test_a_slash_winds_the_chest_back_and_swings_it_through():
    chest = [authoring34.turns(p)[0] for p in authoring34.poses34("a2")]
    assert min(chest) < -10 and max(chest) > 10


def test_the_far_arm_always_stays_behind_the_body():
    # Owner, 2026-10-31: even reaching forward (A7 casts with the far hand) it is drawn behind the body; only on
    # the two-handed grip (A3) is it drawn over the head and body, in every frame.
    for tid in READY:
        fronts = ["arm_far" in fr.front for fr in load_template(tid, angle="34").frames]
        assert all(fronts) if tid == "a3" else not any(fronts), tid


def test_the_two_handed_near_elbow_leads():
    from animgen.skeleton import RELB, RSHO, RWRI
    for fr in load_template("a3", angle="34").frames:
        p = fr.skeleton.points
        assert p[RELB][0] > max(p[RSHO][0], p[RWRI][0])             # forward of both the shoulder and the hand
        assert "arm" not in fr.back


def test_the_kick_keeps_its_length_and_the_bow_draws_level():
    from animgen.skeleton import RANK, RHIP, RWRI
    t = load_template("a1b", angle="34")
    rest_leg = bone_length(t.rest, RHIP, 9) + bone_length(t.rest, 9, RANK)
    kick = t.frames[4].skeleton
    assert bone_length(kick, RHIP, 9) + bone_length(kick, 9, RANK) > 0.9 * rest_leg     # not shortened
    assert kick.points[RANK][1] < kick.points[RHIP][1]                                     # at hip height or above
    a5 = load_template("a5", angle="34").frames
    xs = [f.skeleton.points[RWRI] for f in a5[2:5]]                # the draw: set-up to full draw
    assert abs(xs[-1][1] - xs[0][1]) < 0.5 * abs(xs[-1][0] - xs[0][0])                    # more across than down


def test_drawing_order_moves_the_meshes_of_one_frame():
    names = ["cape", "arm_far", "leg_far", "leg", "body", "skirt", "hair", "arm"]
    rig = Rig(meshes=[SimpleNamespace(name=n) for n in names], bind={}, size=(1, 1))
    order = lambda front=(), back=(): [m.name for m in rig.drawing_order(front, back)]   # noqa: E731
    assert order() == names
    assert order(front=("arm_far",)) == ["cape", "leg_far", "leg", "body", "skirt", "hair", "arm_far", "arm"]
    assert order(front=("leg_far",)) == ["cape", "arm_far", "leg", "leg_far", "body", "skirt", "hair", "arm"]
    assert order(back=("arm",)) == ["cape", "arm_far", "leg_far", "leg", "arm", "body", "skirt", "hair"]


def test_the_angle_is_guessed_from_the_skeleton():
    assert guess_angle(placed(authoring34.rest_points())) == "34"
    assert guess_angle(placed(authoring.solve(authoring.Pose())[0])) == "side"


def test_the_owners_angle_is_kept_and_picks_the_pose_set(session, hero):
    ch = session.set_character(str(hero))
    assert ch.angle == "side"
    session.confirm_skeleton(ch.view_points())
    ch = session.set_angle("34")
    assert ch.to_json()["angle"] == "34" and ch.angle_source == "owner"
    ch = session.set_character(str(hero))                     # opened again: the choice is remembered
    assert ch.angle == "34" and ch.angle_source == "saved"
    t = run(lambda task: session.generate(task, "walk", None, seed=1))
    assert t.state == "completed", t.error
    assert session.template.rest.points[LEYE] is not None
    assert any("leg_far" in s.front for s in session.slots)
    with pytest.raises(Exception):
        session.set_angle("top")


def test_projection_is_a_side_view_at_zero_turn(monkeypatch):
    monkeypatch.setattr(authoring34, "YAW", 0.0)
    monkeypatch.setattr(authoring34, "PITCH", 0.0)
    x, y, d = authoring34.project((0.1, 0.2, 0.05))
    assert math.isclose(x, 0.1) and math.isclose(y, 0.2) and math.isclose(d, 0.05)


def test_flipping_the_facing_keeps_the_owners_angle(session, hero):
    session.set_character(str(hero))
    session.set_angle("34")                                   # chosen before the skeleton is confirmed
    ch = session.set_character(str(hero), "left")
    assert ch.angle == "34" and ch.angle_source == "owner"


def test_two_hands_on_one_weapon_stay_together():
    from animgen.skeleton import LWRI, RWRI
    for tid in ("a3", "a4"):
        for fr in load_template(tid, angle="34").frames:
            r, l = fr.skeleton.points[RWRI], fr.skeleton.points[LWRI]
            assert math.dist(r, l) < 0.2, tid                       # on one handle (the side poses' gap)


def test_the_bow_keeps_fists_in_three_quarter_view():
    assert not any(fr.open_hands for fr in load_template("a5", angle="34").frames)


def test_a_foreshortened_limb_keeps_its_width():
    import numpy as np
    from animgen.rig import _limb_matrix
    m = _limb_matrix((0, 0), (0, 100), (0, 0), (0, 60))           # the bone drawn at 60 % of its length
    assert np.allclose(m[:, :2] @ np.array([10.0, 0.0]), [10.0, 0.0])   # across: unchanged
    assert np.allclose(m[:, :2] @ np.array([0.0, 100.0]), [0.0, 60.0])  # along: shortened
