"""The pose editor (owner, 2026-10-31): a sample character posed in 8 frames and saved as an animation."""

import math

import pytest

from animgen.errors import AppError
from animgen.poses import PoseEditor
from animgen.skeleton import NECK, RELB, RWRI
from animgen.tasks import TaskRunner
from animgen.templates import available, load_template


@pytest.fixture
def editor(tmp_path):
    e = PoseEditor(tmp_path / "data")
    yield e
    e.close()


def close(a, b, tol=1.5):
    return all((p is None) == (q is None) and (p is None or math.dist(p, q) <= tol) for p, q in zip(a, b))


@pytest.mark.parametrize("angle", ["side", "34"])
def test_a_frame_comes_back_as_it_was_posed(editor, angle):
    # The sample is fitted to its own skeleton: an existing pose shows unchanged, so editing is what you get.
    start = editor.start(angle, "a2")
    assert len(start["frames"]) == 8 and start["sample"]["angle"] == angle
    for fr in start["frames"]:
        shown = editor.preview(angle, fr)
        assert close(shown["points"], fr["points"])


def test_a_new_pose_starts_standing_in_every_frame(editor):
    frames = editor.start("side", None)["frames"]
    assert all(f["points"] == frames[0]["points"] for f in frames)


def test_moved_joints_keep_the_bone_lengths(editor):
    fr = editor.start("side", None)["frames"][0]
    pts = [list(p) if p else None for p in fr["points"]]
    pts[RWRI] = [pts[RWRI][0] + 400, pts[RWRI][1]]                   # dragged far off
    shown = editor.preview("side", dict(fr, points=pts))["points"]
    assert math.dist(shown[RELB], shown[RWRI]) <= 1.25 * math.dist(fr["points"][RELB], fr["points"][RWRI]) + 1


def test_a_saved_pose_is_an_animation_for_its_view_only(editor, session, hero):
    frames = editor.start("side", "a2")["frames"]
    frames[4]["contact"] = "air"
    saved = editor.save({"angle": "side", "name": "Chém xoay", "grip": "optional", "frames": frames})
    tid = saved["id"]
    side = {a["id"]: a for a in available("side")}
    assert side[tid]["ready"] and side[tid]["name_vi"] == "Chém xoay" and side[tid]["grip"] == "optional"
    assert tid not in {a["id"] for a in available("34")}
    t = load_template(tid, angle="side")
    assert len(t.frames) == 8 and not t.frames[4].grounded and t.frames[0].weapon is not None
    with pytest.raises(FileNotFoundError):
        load_template(tid, angle="34")
    # It generates like a built-in animation.
    session.set_character(str(hero))
    tpl = session.prepare(tid)
    assert tpl.id == tid
    # Opened again it is the same; deleted it is gone.
    again = editor.open(tid)
    assert again["name"] == "Chém xoay" and close(again["frames"][2]["points"], frames[2]["points"], 2.0)
    editor.delete(tid)
    assert tid not in {a["id"] for a in available("side")}


def test_saving_needs_a_name_and_eight_frames(editor):
    frames = editor.start("side", None)["frames"]
    with pytest.raises(AppError) as e:
        editor.save({"angle": "side", "name": "  ", "frames": frames})
    assert e.value.code == "pose_name_missing"
    with pytest.raises(AppError):
        editor.save({"angle": "side", "name": "x", "frames": frames[:7]})
    with pytest.raises(AppError) as e:
        editor.open("u_00000000")
    assert e.value.code == "pose_unknown"
    with pytest.raises(AppError):
        editor.delete("../idle")


def test_a_frame_shape_is_shown_previewed_and_saved(editor):
    # The built-in poses carry shapes (jump lands squashed); the editor shows them, draws them and keeps them.
    frames = editor.start("side", "jump")["frames"]
    assert frames[6]["shape"] == {"squash": 0.1}
    fr = dict(frames[3], shape={"size": {"head": 1.3}, "reach": {"arm_r": 1.2}, "bend": {"torso": 10}})
    from PIL import Image
    plain = editor.preview("side", frames[3])
    with Image.open(plain["image"]) as a:                    # only the latest preview is kept
        before = a.tobytes()
    shaped = editor.preview("side", fr)
    assert close(shaped["points"], plain["points"])          # the joints shown are the pose's own
    with Image.open(shaped["image"]) as b:
        assert b.tobytes() != before
    frames[3] = fr
    tid = editor.save({"angle": "side", "name": "Nhảy to đầu", "frames": frames})["id"]
    t = load_template(tid, angle="side")
    assert t.frames[3].shape.size == {"head": 1.3} and t.frames[6].shape.squash == 0.1
    assert editor.open(tid)["frames"][3]["shape"]["reach"] == {"arm_r": 1.2}
    with pytest.raises(AppError):
        editor.preview("side", dict(fr, shape={"squash": 3}))


def test_the_preview_draws_the_sample(editor):
    from PIL import Image
    fr = editor.start("34", "walk")["frames"][3]
    out = editor.preview("34", fr)
    with Image.open(out["image"]) as im:
        assert im.getchannel("A").getbbox() is not None
    assert out["points"][NECK] is not None
