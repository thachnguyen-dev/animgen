import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import synth
from animgen.errors import AppError
from animgen.session import WeaponChoice
from animgen.tasks import TaskRunner


def run(fn, timeout=120):
    r = TaskRunner()
    t = r.start("t", fn)
    return r.wait(t.id, timeout)


def add_sword(session, tmp_path, name="Kiếm", skin="bạc", colour=(200, 200, 215)):
    p, g, t = synth.sword(tmp_path / f"{skin}.png", colour)
    w = session.weapons.add(name, "one_handed", skin, [str(p)])
    session.weapons.set_points(w.id, skin, 1, g, t)
    return w


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


def test_weapon_rules(session, hero, tmp_path):
    session.set_character(str(hero))
    w = add_sword(session, tmp_path)
    with pytest.raises(AppError) as e:
        session.prepare("a2", None)
    assert e.value.code == "weapon_required"
    with pytest.raises(AppError) as e:
        session.prepare("a1a", WeaponChoice(w.id, None))
    assert e.value.code == "weapon_not_allowed"
    with pytest.raises(AppError) as e:
        session.prepare("a5", WeaponChoice(w.id, None))
    assert e.value.code == "weapon_wrong_kind"
    with pytest.raises(AppError) as e:
        session.prepare("jump", None)
    assert e.value.code == "animation_not_ready"
    session.prepare("walk", WeaponChoice(w.id, None))  # optional weapon on a non-attack animation


def test_skin_change_is_recombine_only(session, hero, tmp_path):
    session.set_character(str(hero))
    w = add_sword(session, tmp_path, skin="bạc")
    add_sword(session, tmp_path, skin="đỏ", colour=(200, 30, 30))
    run(lambda task: session.generate(task, "a2", WeaponChoice(w.id, "bạc")))
    bases = [s.base for s in session.slots]
    calls = len(session.engine.drawer._attempts)
    t0 = time.time()
    session.set_weapon(WeaponChoice(w.id, "đỏ"))
    assert time.time() - t0 < 10
    assert [s.base for s in session.slots] == bases           # the same character frames
    assert len(session.engine.drawer._attempts) == calls       # no drawing
    a = np.asarray(Image.open(session.slots[4].preview))
    assert ((a[..., 0] > 180) & (a[..., 1] < 60) & (a[..., 3] > 200)).sum() > 50  # red blade present


def test_bow_stages_change_image(session, hero, tmp_path):
    session.set_character(str(hero))
    imgs = []
    for st in (1, 2, 3):
        p, g, t = synth.bow_stage(tmp_path / f"b{st}.png", st)
        imgs.append((p, g, t))
    w = session.weapons.add("Cung", "bow", "gỗ", [str(p) for p, _, _ in imgs])
    for st, (_, g, t) in enumerate(imgs, start=1):
        session.weapons.set_points(w.id, "gỗ", st, g, t)
    tk = run(lambda task: session.generate(task, "a5", WeaponChoice(w.id, None)))
    assert tk.state == "completed", tk.error
    assert all(f["status"] == "ok" for f in session.state()["frames"])


def test_left_facing_character_keeps_its_facing(session, tmp_path):
    p = synth.character(tmp_path / "left.png", facing="left")
    ch = session.set_character(str(p), facing="left")
    assert ch.placement.mirrored and ch.facing_source == "owner"
    run(lambda task: session.generate(task, "a1a", None))
    a = np.asarray(Image.open(session.slots[4].preview))[..., 3]
    xs = np.nonzero(a.max(axis=0) > 128)[0]
    # punching to the left: the figure reaches further left of centre than right
    assert (512 - xs.min()) > (xs.max() - 512)


def test_export_names_with_weapon(session, hero, tmp_path):
    session.set_character(str(hero), name="Hiệp sĩ")
    w = add_sword(session, tmp_path)
    run(lambda task: session.generate(task, "a2", WeaponChoice(w.id, None)))
    t = run(lambda task: session.export(task, str(tmp_path / "out"), 512))
    assert t.state == "completed", t.error
    names = sorted(Path(p).name for p in t.result["files"])
    assert names[0] == "hiep-si_attack-a2_kiem_bac_01.png"
    assert "hiep-si_attack-a2_kiem_bac_sheet.png" in names


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
        s.prepare("idle", None)
    assert e.value.code == "skeleton_unconfirmed"
    s.confirm_skeleton(ch.view_points())
    assert s.prepare("idle", None).id == "idle"
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
