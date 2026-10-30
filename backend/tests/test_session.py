import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import synth
from animgen.errors import AppError
from animgen.tasks import TaskRunner


def run(fn, timeout=120):
    r = TaskRunner()
    t = r.start("t", fn)
    return r.wait(t.id, timeout)


def test_unarmed_generation_all_frames_ok(session, hero):
    session.set_character(str(hero))
    t = run(lambda task: session.generate(task, "walk", None, seed=1))
    assert t.state == "completed", t.error
    st = session.state()
    assert [f["status"] for f in st["frames"]] == ["ok"] * 8 and st["complete"]
    assert st["demo"] is True


def test_retry_then_success(session, hero):
    session.set_character(str(hero))
    session.engine.drawer.fail_plan = {2: 1}
    t = run(lambda task: session.generate(task, "idle", None))
    f3 = session.state()["frames"][2]
    assert t.state == "completed" and f3["status"] == "ok" and f3["attempts"] == 2


def test_retries_are_bounded_then_owner_is_told(session, hero):
    session.set_character(str(hero))
    session.engine.drawer.fail_plan = {4: 99}
    t = run(lambda task: session.generate(task, "idle", None))
    f5 = session.state()["frames"][4]
    assert f5["status"] == "failed" and f5["attempts"] == 3  # 1 + 2 automatic retries
    assert f5["error"]["code"] == "frame_failed_after_retries"
    assert t.warnings and not session.state()["complete"]
    # export refuses an incomplete animation
    with pytest.raises(AppError):
        session.export(t, str(hero.parent / "out"), 256)


def test_regenerate_keeps_good_frame_until_accepted(session, hero):
    session.set_character(str(hero))
    run(lambda task: session.generate(task, "idle", None))
    before = session.slots[3].base
    run(lambda task: session.regenerate(task, 4))
    s = session.slots[3]
    assert s.base == before and s.candidate is not None and s.candidate_status == "ok"
    session.keep(4)
    assert session.slots[3].base == before and session.slots[3].candidate is None
    run(lambda task: session.regenerate(task, 4))
    cand = session.slots[3].candidate
    session.accept(4)
    assert session.slots[3].base == cand


def test_failed_regeneration_never_loses_good_frame(session, hero):
    session.set_character(str(hero))
    run(lambda task: session.generate(task, "idle", None))
    before = session.slots[0].base
    session.engine.drawer.fail_plan = {0: 99}
    session.engine.drawer._attempts = {}
    run(lambda task: session.regenerate(task, 1))
    s = session.slots[0]
    assert s.base == before and s.status == "ok" and s.candidate_status == "failed"


def test_regenerating_failed_frame_fills_it_directly(session, hero):
    session.set_character(str(hero))
    session.engine.drawer.fail_plan = {1: 3}
    run(lambda task: session.generate(task, "idle", None))
    assert session.slots[1].status == "failed"
    run(lambda task: session.regenerate(task, 2))
    assert session.slots[1].status == "ok" and session.slots[1].candidate is None


def test_regenerate_whole_animation_and_accept_all(session, hero):
    session.set_character(str(hero))
    run(lambda task: session.generate(task, "walk", None))
    old = [s.base for s in session.slots]
    run(lambda task: session.regenerate(task, None))
    assert all(s.candidate is not None for s in session.slots)
    session.accept_all()
    assert all(s.base != o for s, o in zip(session.slots, old))


def test_cancel_between_frames(session, hero):
    session.set_character(str(hero))
    r = TaskRunner()
    orig = session.engine.drawer.draw

    def slow(req):
        time.sleep(0.05)
        return orig(req)
    session.engine.drawer.draw = slow
    t = r.start("g", lambda task: session.generate(task, "walk", None))
    time.sleep(0.12)
    r.cancel(t.id)
    t = r.wait(t.id)
    assert t.state == "cancelled"


def test_grip_rules(session, hero):
    session.set_character(str(hero))
    seen = []
    draw = session.engine.drawer.draw
    session.engine.drawer.draw = lambda req: seen.append(req.hands_gripping) or draw(req)
    with pytest.raises(AppError) as e:
        session.prepare("jump")
    assert e.value.code == "animation_not_ready"
    for anim, asked, gripping in (("a1a", True, False),   # unarmed: never
                                  ("a2", False, True),    # made for a weapon: always
                                  ("a7", None, True),
                                  ("walk", None, False),  # the owner's choice
                                  ("walk", True, True)):
        seen.clear()
        run(lambda task: session.generate(task, anim, asked, only=[0]))
        assert session.grip is gripping and seen == [gripping], anim


def test_four_frame_test_run_draws_every_other_frame(session, hero):
    session.set_character(str(hero))
    run(lambda task: session.generate(task, "idle", None, only=[0, 2, 4, 6]))
    st = session.state()
    assert [f["status"] for f in st["frames"]] == ["ok", "empty"] * 4 and not st["complete"]


def test_left_facing_character_keeps_its_facing(session, tmp_path):
    p = synth.character(tmp_path / "left.png", facing="left")
    ch = session.set_character(str(p), facing="left")
    assert ch.placement.mirrored and ch.facing_source == "owner"
    run(lambda task: session.generate(task, "a1a", None))
    a = np.asarray(Image.open(session.slots[4].preview))[..., 3]
    xs = np.nonzero(a.max(axis=0) > 128)[0]
    # punching to the left: the figure reaches further left of centre than right
    assert (512 - xs.min()) > (xs.max() - 512)


def test_export_names(session, hero, tmp_path):
    session.set_character(str(hero), name="Hiệp sĩ")
    run(lambda task: session.generate(task, "a2", None))
    t = run(lambda task: session.export(task, str(tmp_path / "out"), 512))
    assert t.state == "completed", t.error
    names = sorted(Path(p).name for p in t.result["files"])
    assert names[0] == "hiep-si_attack-a2_01.png"
    assert "hiep-si_attack-a2_sheet.png" in names and "hiep-si_attack-a2_weapon.json" in names


def test_weapon_attack_grips_and_exports_hand_anchors(session, tmp_path):
    import json
    from animgen.skeleton import hand_point
    left = synth.character(tmp_path / "left.png", facing="left")
    session.set_character(str(left), "left", name="Cung thủ")
    seen = []
    draw = session.engine.drawer.draw
    session.engine.drawer.draw = lambda req: seen.append(req.hands_gripping) or draw(req)
    run(lambda task: session.generate(task, "a5", None))
    assert seen and all(seen)                             # hands closed as if holding the bow
    t = run(lambda task: session.export(task, str(tmp_path / "out"), 256))
    assert t.state == "completed", t.error
    js = [p for p in t.result["files"] if p.endswith("_weapon.json")]
    assert len(js) == 1
    data = json.load(open(js[0], encoding="utf-8"))
    assert data["frame_size"] == [256, 256] and len(data["frames"]) == 8 and data["animation"] == "a5"
    canvas = session.character.placement.image.width
    k = 256 / canvas
    tpl = session.template
    for f, tf, slot in zip(data["frames"], tpl.frames, session.slots):
        assert f["weapon"] and f["hand"] == "far"          # the bow is held by the far arm
        hx, hy = hand_point(slot.skeleton, tf.weapon.hand)
        assert f["x"] == pytest.approx((canvas - 1 - hx) * k, abs=0.02)   # mirrored back: the character faces left
        assert f["y"] == pytest.approx(hy * k, abs=0.02)
        assert f["angle"] == pytest.approx((180.0 - tf.weapon.angle + 180.0) % 360.0 - 180.0, abs=0.02)


@pytest.mark.parametrize("anim", ["a1a", "idle"])
def test_no_hand_anchors_without_gripping_hands(session, hero, tmp_path, anim):
    session.set_character(str(hero))
    run(lambda task: session.generate(task, anim, None))
    t = run(lambda task: session.export(task, str(tmp_path / "out"), 128))
    assert not any(p.endswith(".json") for p in t.result["files"])


def test_cast_keeps_the_weapon_in_the_near_hand(session, hero, tmp_path):
    import json
    session.set_character(str(hero))
    run(lambda task: session.generate(task, "a7", None))
    t = run(lambda task: session.export(task, str(tmp_path / "out"), 128))
    js = [p for p in t.result["files"] if p.endswith("_weapon.json")]
    data = json.load(open(js[0], encoding="utf-8"))
    assert {f["hand"] for f in data["frames"]} == {"near"}


def test_temp_folder_cleanup(tmp_path):
    from animgen.engines.demo import DemoEngine
    from animgen.session import Session
    s1 = Session(DemoEngine(), tmp_path / "d")
    old = s1.temp
    s2 = Session(DemoEngine(), tmp_path / "d")  # next start removes the previous session's files
    assert not old.exists() and s2.temp.exists()
    s2.close()


# ---------------------------------------------------------------------------------------------- skeleton confirm

def test_skeleton_must_be_confirmed_when_required(tmp_path, hero):
    from animgen.engines.demo import DemoEngine
    from animgen.session import Session
    s = Session(DemoEngine(), tmp_path / "d", require_confirmed_skeleton=True)
    ch = s.set_character(str(hero))
    assert ch.to_json()["skeleton"]["confirmed"] is False
    with pytest.raises(AppError) as e:
        s.prepare("idle")
    assert e.value.code == "skeleton_unconfirmed"
    s.confirm_skeleton(ch.view_points())
    assert s.prepare("idle").id == "idle"
    s.close()


def test_confirmed_skeleton_is_saved_and_reused(session, tmp_path):
    left = synth.character(tmp_path / "left.png", facing="left")
    ch = session.set_character(str(left), "left")
    pts = ch.view_points()
    pts[4] = [pts[4][0] + 30, pts[4][1] - 40]          # the owner moves the right wrist on the view image
    session.confirm_skeleton(pts)
    again = session.set_character(str(left))             # same image, facing not given: both come from the save
    j = again.to_json()["skeleton"]
    assert again.facing == "left" and again.facing_source == "saved"
    assert j["confirmed"] and j["source"] == "saved"
    assert j["points"][4] == pytest.approx(pts[4], abs=0.01)
    # internally the character faces right: the view points come back mirrored
    w = again.placement.image.width
    assert again.skeleton.points[4][0] == pytest.approx(w - 1 - pts[4][0], abs=0.01)


def test_bad_skeleton_points_refused(session, hero):
    ch = session.set_character(str(hero))
    pts = ch.view_points()
    with pytest.raises(AppError) as e:
        session.confirm_skeleton(pts[:5])
    assert e.value.code == "skeleton_invalid"
    bad = list(pts)
    bad[1] = [5000, 10]
    with pytest.raises(AppError):
        session.confirm_skeleton(bad)
    bad = list(pts)
    bad[1] = None
    with pytest.raises(AppError) as e:
        session.confirm_skeleton(bad)
    assert e.value.code == "skeleton_incomplete"


# ---------------------------------------------------------------------------------------------- rig


def test_rig_is_saved_per_image_and_dropped_with_a_new_skeleton(tmp_path, hero):
    from animgen.engines.demo import DemoEngine
    from animgen.session import Session
    s = Session(DemoEngine(), tmp_path / "d", require_confirmed_skeleton=True, require_rig=True)
    ch = s.set_character(str(hero))
    s.confirm_skeleton(ch.view_points())
    t = run(lambda task: s.generate(task, "idle"))
    assert t.state == "failed" and t.error["code"] == "rig_missing"
    t = run(lambda task: s.make_rig(task, seed=3))
    assert t.state == "completed", t.error
    assert s.rig is not None and s.state()["rig"]["seed"] == 3
    t = run(lambda task: s.generate(task, "walk", seed=1))
    assert t.state == "completed", t.error and s.state()["complete"]
    s.close()

    s2 = Session(DemoEngine(), tmp_path / "d", require_confirmed_skeleton=True, require_rig=True)
    ch = s2.set_character(str(hero))                     # same image: the saved skeleton and rig come back
    assert s2.rig is not None and s2.rig.seed == 3
    pts = ch.view_points()
    pts[4] = [pts[4][0] + 20, pts[4][1]]
    s2.confirm_skeleton(pts)                              # a different skeleton: the rig was drawn for the old one
    assert s2.rig is None and s2.state()["rig"] is None
    s2.close()


def test_rig_frames_are_the_same_every_time_unless_finished(session, hero):
    session.set_character(str(hero))
    assert run(lambda task: session.make_rig(task, seed=2)).state == "completed"
    run(lambda task: session.generate(task, "a7", seed=1))
    first = [np.array(Image.open(f["preview"])) for f in session.state()["frames"]]
    run(lambda task: session.generate(task, "a7", seed=99))
    again = [np.array(Image.open(f["preview"])) for f in session.state()["frames"]]
    assert all((a == b).all() for a, b in zip(first, again))
    with pytest.raises(AppError) as e:
        session.regenerate(None, 2)
    assert e.value.code == "nothing_to_redraw"
    t = run(lambda task: session.generate(task, "a7", seed=1, finish=True))
    assert t.state == "completed", t.error
    assert session.state()["finished"] is True
