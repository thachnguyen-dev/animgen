import json
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


def test_dual_wield_exports_both_hands(session, hero, tmp_path):
    import json
    session.set_character(str(hero))
    run(lambda task: session.generate(task, "a8a", None))
    t = run(lambda task: session.export(task, str(tmp_path / "out"), 256))
    js = [p for p in t.result["files"] if p.endswith("_weapon.json")]
    data = json.load(open(js[0], encoding="utf-8"))
    for f in data["frames"]:
        assert f["weapon"] and f["hand"] == "near" and f["second"]["hand"] == "far"
        assert f["second"]["layer"] == "behind" and (f["x"], f["y"]) != (f["second"]["x"], f["second"]["y"])


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


def test_one_rig_drawing_is_redrawn_and_the_other_kept(session, hero):
    session.set_character(str(hero))
    assert run(lambda task: session.make_rig(task, seed=2)).state == "completed"
    folder = session.rig_dir / session.character.key
    body = (folder / "rig.png").read_bytes()
    t = run(lambda task: session.make_rig(task, seed=7, part="hand"))
    assert t.state == "completed", t.error
    assert (folder / "rig.png").read_bytes() == body and session.rig.seed == 2
    info = json.loads((folder / "rig.json").read_text(encoding="utf-8"))
    assert info["seed"] == 2 and info["hand_seed"] == 7


def test_the_rig_is_made_of_body_part_layers(session, hero):
    from animgen import parts
    session.set_character(str(hero))
    assert run(lambda task: session.make_rig(task, seed=2)).state == "completed"
    folder = session.rig_dir / session.character.key
    names = {p.stem for p in (folder / "layers").glob("*.png")}
    assert {"torso", "head", "arm_upper", "arm_lower", "leg_upper", "leg_lower", "open_hand"} <= names
    assert names <= set(parts.NAMES)
    st = session.state()["rig"]
    assert Path(st["psd"]).is_file()
    from psd_tools import PSDImage
    psd = PSDImage.open(st["psd"])
    labels = [layer.name for layer in psd.descendants()]
    assert labels[0] == parts.GUIDE and parts.GROUP in labels and parts.LABELS["torso"] in labels
    hand = [layer for layer in psd.descendants() if layer.name == parts.LABELS["open_hand"]][0]
    assert not hand.visible                              # shown only in the frames that open the hand


def test_the_owner_corrects_the_parts_in_photoshop(session, hero):
    from animgen import parts
    session.set_character(str(hero))
    assert run(lambda task: session.make_rig(task, seed=2)).state == "completed"
    folder = session.rig_dir / session.character.key
    size, mirrored = session.character.placement.image.size, session.character.placement.mirrored
    r = session.rig_edit_file()
    assert r["file"] == session.state()["rig"]["psd"] and r["state"]["rig"]["editing"] == ["parts"]
    guide = Image.open(folder / "rig.png")

    layers = parts.read_psd(Path(r["file"]), size, mirrored)
    layers["torso"][..., 0] = np.where(layers["torso"][..., 3] > 0, 255, layers["torso"][..., 0])
    del layers["head"]                                   # the owner may delete a part
    parts.write_psd(Path(r["file"]), guide, layers, mirrored)
    st = session.reload_rig_edits()
    assert st["rig"] is not None and st["rig"]["editing"] == ["parts"]
    torso = np.array(Image.open(folder / "layers" / "torso.png"))
    assert (torso[torso[..., 3] > 0, 0] == 255).all() and not (folder / "layers" / "head.png").exists()

    small = {k: v[::2, ::2].copy() for k, v in layers.items()}      # resized: refused, the layers kept
    parts.write_psd(Path(r["file"]), guide.resize((size[0] // 2, size[1] // 2)), small, mirrored)
    with pytest.raises(AppError) as e:
        session.reload_rig_edits()
    assert e.value.code == "rig_edit_size" and session.rig is not None

    nothing = {"arm_upper": layers["arm_upper"]}         # the body layer gone: refused
    parts.write_psd(Path(r["file"]), guide, nothing, mirrored)
    with pytest.raises(AppError) as e:
        session.reload_rig_edits()
    assert e.value.code == "rig_edit_failed" and (folder / "layers" / "torso.png").exists()

    Path(r["file"]).write_bytes(b"not a psd")
    with pytest.raises(AppError) as e:
        session.reload_rig_edits()
    assert e.value.code == "rig_edit_unreadable"


def test_a_photoshop_file_saved_while_the_app_was_closed_is_taken_on_start(tmp_path, hero):
    from animgen import parts
    from animgen.engines.demo import DemoEngine
    from animgen.session import Session
    s = Session(DemoEngine(), tmp_path / "d")
    s.set_character(str(hero))
    assert run(lambda task: s.make_rig(task, seed=2)).state == "completed"
    folder = s.rig_dir / s.character.key
    size, mirrored = s.character.placement.image.size, s.character.placement.mirrored
    psd = Path(s.state()["rig"]["psd"])
    s.close()
    layers = parts.read_psd(psd, size, mirrored)
    layers["torso"][..., 1] = np.where(layers["torso"][..., 3] > 0, 7, layers["torso"][..., 1])
    time.sleep(0.05)
    parts.write_psd(psd, Image.open(folder / "rig.png"), layers, mirrored)
    s2 = Session(DemoEngine(), tmp_path / "d")
    s2.set_character(str(hero))
    torso = np.array(Image.open(folder / "layers" / "torso.png"))
    assert s2.rig is not None and (torso[torso[..., 3] > 0, 1] == 7).all()
    s2.close()


def test_a_rig_is_kept_when_its_saved_joints_differ_by_rounding_only(session, hero):
    session.set_character(str(hero))
    assert run(lambda task: session.make_rig(task, seed=2)).state == "completed"
    path = session.rig_dir / session.character.key / "rig.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["skeleton"] = [None if p is None else [p[0] + 0.4, p[1] - 0.4] for p in info["skeleton"]]
    path.write_text(json.dumps(info), encoding="utf-8")
    session.set_character(str(hero))
    assert session.rig is not None


def test_psd_round_trip_keeps_names_places_and_facing(tmp_path):
    from animgen import parts
    a = np.zeros((64, 48, 4), np.uint8)
    a[10:20, 5:15] = (200, 10, 10, 255)
    b = np.zeros((64, 48, 4), np.uint8)
    b[30:50, 20:30] = (10, 200, 10, 255)
    guide = Image.new("RGBA", (48, 64), (0, 0, 255, 255))
    for mirrored in (False, True):
        path = tmp_path / f"p{mirrored}.psd"
        parts.write_psd(path, guide, {"torso": a, "arm_upper": b}, mirrored)
        back = parts.read_psd(path, (48, 64), mirrored)
        assert set(back) == {"torso", "arm_upper"}
        assert (back["torso"] == a).all() and (back["arm_upper"] == b).all()


def test_key_background_only_clears_the_background_touching_the_edges():
    from animgen.imaging import key_background
    a = np.full((60, 60, 4), 255, np.uint8)
    a[10:50, 10:50, :3] = 0
    a[25:35, 25:35, :3] = 255                              # white inside the figure stays
    out = np.array(key_background(Image.fromarray(a, "RGBA")))
    assert out[0, 0, 3] == 0 and out[30, 30, 3] == 255 and out[15, 15, 3] == 255
