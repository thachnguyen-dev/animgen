import numpy as np
import pytest

from animgen import authoring
from animgen.shape import Shape, apply_reach, map_point, parse, shaped_hand, squash_matrix
from animgen.skeleton import RELB, RSHO, RWRI, Skeleton, hand_point
from animgen.templates import TemplateError, available, load_template, parse_template

from test_puppet import STAND
from test_rig import FOREARM, SPREAD, rig

GROUND = max(p[1] for p in STAND if p is not None) + 20


def test_parse_checks_ranges_and_names():
    assert parse(None) is None
    assert parse({"squash": 0.0, "size": {"head": 1.0}}) is None          # neutral is nothing
    s = parse({"squash": 0.1, "size": {"hand_r": 1.2}, "reach": {"arm_r": 1.1}, "bend": {"torso": -8}})
    assert s.squash == 0.1 and s.size == {"hand_r": 1.2} and s.reach == {"arm_r": 1.1} and s.bend == {"torso": -8}
    for bad in ({"squash": 0.5}, {"size": {"hand_r": 3}}, {"size": {"tail": 1.1}}, {"bend": {"torso": 90}},
                {"reach": {"arm_r": "long"}}, [1]):
        with pytest.raises(ValueError):
            parse(bad)


def test_a_bad_shape_is_refused_with_the_template():
    data = authoring.build("idle")
    data["frames"][3]["shape"] = {"squash": 0.9}
    with pytest.raises(TemplateError):
        parse_template(data)


def test_reach_lengthens_the_arm_from_the_shoulder():
    sk = Skeleton(list(SPREAD))
    out = apply_reach(sk, Shape(reach={"arm_r": 1.2}))
    sho = np.array(sk.points[RSHO])
    for j in (RELB, RWRI):
        before = np.linalg.norm(np.array(sk.points[j]) - sho)
        after = np.linalg.norm(np.array(out.points[j]) - sho)
        assert after == pytest.approx(before * 1.2, rel=1e-6)
    assert out.points[1] == sk.points[1]                                      # the rest is untouched


def test_squash_keeps_the_feet_on_the_ground_and_the_area():
    sk = Skeleton(list(STAND))
    low = max(p[1] for p in STAND if p is not None)
    m = squash_matrix(Shape(squash=0.1), sk, low + 5)
    assert abs(np.linalg.det(m[:, :2]) - 1.0) < 1e-9                         # area kept
    assert map_point(m, (300.0, low + 5))[1] == pytest.approx(low + 5)       # the ground line stays
    assert map_point(m, sk.points[0])[1] > sk.points[0][1]                    # the head comes down


def test_a_crouched_landing_still_squashes_on_the_ground():
    # A crouch is short, so "near the ground" must not be measured against the figure's height (it floated).
    pts, grounded = authoring.solve(authoring.jump()[6])
    assert grounded
    sk = Skeleton([None if p is None else (p[0] * 400 + 500, p[1] * 400 + 900) for p in pts])
    m = squash_matrix(Shape(squash=0.18), sk, 900.0)
    assert map_point(m, (500.0, 900.0))[1] == pytest.approx(900.0)


def test_squash_in_the_air_is_about_the_middle():
    sk = Skeleton(list(STAND)).translated(0, -300)
    m = squash_matrix(Shape(squash=-0.1), sk, GROUND)
    x0, y0, x1, y1 = sk.bounds()
    assert map_point(m, ((x0 + x1) / 2, (y0 + y1) / 2))[1] == pytest.approx((y0 + y1) / 2)


def _colour_count(img, colour) -> int:
    a = np.array(img)
    return int((np.abs(a[..., :4].astype(int) - colour).sum(axis=2) < 30).sum())


def test_a_bigger_hand_draws_more_hand_and_moves_the_anchor():
    r, sk = rig(), Skeleton(list(SPREAD))
    plain = r.render_keep(sk, floor=GROUND)[0]
    big = r.render_keep(sk, floor=GROUND, shape=Shape(size={"hand_r": 1.3}))[0]
    assert _colour_count(big, FOREARM) > _colour_count(plain, FOREARM)
    p, q = hand_point(sk, "r"), shaped_hand(sk, "r", Shape(size={"hand_r": 1.3}), GROUND)
    assert np.linalg.norm(np.subtract(q, sk.points[RELB])) > np.linalg.norm(np.subtract(p, sk.points[RELB]))


def test_a_bend_moves_the_middle_and_keeps_the_ends():
    r, sk = rig(), Skeleton(list(SPREAD))
    plain = np.array(r.render_keep(sk, floor=GROUND)[0])[..., 3] > 0
    bent = np.array(r.render_keep(sk, floor=GROUND, shape=Shape(bend={"torso": 20}))[0])[..., 3] > 0
    assert (plain != bent).sum() > 100


def test_no_shape_draws_exactly_as_before():
    r, sk = rig(), Skeleton(list(SPREAD))
    a = np.array(r.render_keep(sk, floor=GROUND)[0])
    b = np.array(r.render_keep(sk, floor=GROUND, shape=Shape())[0])
    assert (a == b).all()


@pytest.mark.parametrize("angle", ["side", "34"])
def test_every_built_in_pose_loads_with_its_shapes(angle):
    shaped = 0
    for e in available(angle):
        if e["ready"]:
            shaped += any(f.shape is not None for f in load_template(e["id"], angle=angle).frames)
    assert shaped >= 19
