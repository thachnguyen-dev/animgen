import math

import numpy as np
import pytest
from PIL import Image

import synth
from animgen import config
from animgen.errors import AppError
from animgen.export import base_name, export_animation, sprite_sheet
from animgen.frames import clean_alpha, fit_to_canvas, validate_file, validate_image
from animgen.weapons import Points, WeaponStore, combine, slug, transform_weapon, weapon_clipped

# ---------------------------------------------------------------------------------------------------- weapons


def test_slug_vietnamese():
    assert slug("Kiếm Đồng") == "kiem-dong"
    assert slug("  ") == "x"


def test_grip_lands_on_hand_and_points_at_angle(sword):
    path, grip, tip = sword
    img = Image.open(path).convert("RGBA")
    hand = (400.0, 500.0)
    for angle in (0, -45, 90, 180):
        layer = transform_weapon(img, Points(grip, tip), hand, angle, 1.0, (1024, 1024))
        a = np.asarray(layer)[..., 3]
        # the tip ends up at hand + length * direction
        L = math.hypot(tip[0] - grip[0], tip[1] - grip[1])
        tx = hand[0] + math.cos(math.radians(angle)) * L
        ty = hand[1] + math.sin(math.radians(angle)) * L
        win = a[int(ty) - 6:int(ty) + 7, int(tx) - 6:int(tx) + 7]
        assert win.max() > 100, angle
        # the handle sits on the hand
        assert a[int(hand[1]) - 3:int(hand[1]) + 4, int(hand[0]) - 3:int(hand[0]) + 4].max() > 200


def test_weapon_scale(sword):
    path, grip, tip = sword
    img = Image.open(path).convert("RGBA")
    layer = transform_weapon(img, Points(grip, tip), (100, 500), 0, 0.5, (1024, 1024))
    xs = np.nonzero(np.asarray(layer)[..., 3].max(axis=0) > 128)[0]
    assert abs((xs.max() - xs.min()) - 340 * 0.5) < 8


def test_layer_order():
    ch = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
    wp = Image.new("RGBA", (10, 10), (0, 0, 255, 255))
    assert combine(ch, wp, "front").getpixel((5, 5))[:3] == (0, 0, 255)
    assert combine(ch, wp, "behind").getpixel((5, 5))[:3] == (255, 0, 0)


def test_weapon_clipped(sword):
    path, grip, tip = sword
    img = Image.open(path).convert("RGBA")
    assert not weapon_clipped(img, Points(grip, tip), (500, 500), 0, 1.0, (1024, 1024))
    assert weapon_clipped(img, Points(grip, tip), (900, 500), 0, 1.0, (1024, 1024))


def test_store_skins_reuse_points(tmp_path, sword):
    path, grip, tip = sword
    store = WeaponStore(tmp_path / "w")
    w = store.add("Kiếm", "one_handed", "bạc", [str(path)])
    assert not w.ready(w.skins[0])
    store.set_points(w.id, "bạc", 1, grip, tip)
    red, _, _ = synth.sword(tmp_path / "red.png", colour=(200, 40, 40))
    w = store.add("Kiếm", "one_handed", "đỏ", [str(red)])
    assert [s.name for s in w.skins] == ["bạc", "đỏ"]
    assert all(w.ready(s) for s in w.skins)
    assert store.list()[0].name == "Kiếm"


def test_store_replacing_skin_removes_old_image(tmp_path, sword):
    path, _, _ = sword
    store = WeaponStore(tmp_path / "w")
    old = store.add("Kiếm", "one_handed", "bạc", [str(path)]).skins[0].images[0]
    w = store.add("Kiếm", "one_handed", "bạc", [str(path)])
    assert len(w.skins) == 1
    assert not (w.folder / old).exists()
    assert sorted(p.name for p in w.folder.glob("*.png")) == w.skins[0].images


def test_bow_needs_three_images(tmp_path):
    store = WeaponStore(tmp_path / "w")
    p, _, _ = synth.bow_stage(tmp_path / "b1.png", 1)
    with pytest.raises(AppError) as e:
        store.add("Cung", "bow", "gỗ", [str(p)])
    assert e.value.code == "weapon_bow_needs_three"


def test_weapon_image_needs_alpha(tmp_path):
    store = WeaponStore(tmp_path / "w")
    p = synth.opaque(tmp_path / "o.png")
    with pytest.raises(AppError) as e:
        store.add("X", "one_handed", "", [str(p)])
    assert e.value.code == "weapon_no_alpha"


def test_bad_points(tmp_path, sword):
    path, grip, tip = sword
    store = WeaponStore(tmp_path / "w")
    w = store.add("Kiếm", "one_handed", "", [str(path)])
    with pytest.raises(AppError):
        store.set_points(w.id, None, 1, (5000, 1), tip)
    with pytest.raises(AppError):
        store.set_points(w.id, None, 1, grip, grip)

# ---------------------------------------------------------------------------------------------------- frames


def figure(size=1024):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    arr = np.array(img)
    arr[400:900, 450:560] = (200, 50, 50, 255)
    return Image.fromarray(arr, "RGBA")


def test_validation():
    good = figure()
    assert validate_image(good, (1024, 1024)).ok
    assert not validate_image(good.convert("RGB"), (1024, 1024)).ok
    assert not validate_image(good, (512, 512)).ok
    assert not validate_image(Image.new("RGBA", (1024, 1024), (0, 0, 0, 0)), (1024, 1024)).ok
    assert not validate_image(Image.new("RGBA", (1024, 1024), (9, 9, 9, 255)), (1024, 1024)).ok


def test_validate_file(tmp_path):
    figure().save(tmp_path / "f.png")
    assert validate_file(tmp_path / "f.png", (1024, 1024)).ok
    (tmp_path / "bad.png").write_bytes(b"xx")
    assert not validate_file(tmp_path / "bad.png", (1024, 1024)).ok


def test_clean_alpha_removes_specks():
    arr = np.array(figure())
    arr[10:12, 10:12] = (0, 255, 0, 255)
    arr[100, 100] = (0, 0, 0, 5)
    out = np.array(clean_alpha(Image.fromarray(arr, "RGBA")))
    assert out[10, 10, 3] == 0 and out[100, 100, 3] == 0 and out[500, 500, 3] == 255


def test_clean_alpha_removes_second_figure_away_from_skeleton():
    arr = np.array(figure())
    box = clean_alpha(Image.fromarray(arr, "RGBA")).getchannel("A").getbbox()
    arr[box[1]:box[3], 900:1000] = (200, 120, 80, 255)   # a large second "figure" at the right edge
    near = (box[0], box[1], box[2] - 1, box[3] - 1)
    out = np.array(clean_alpha(Image.fromarray(arr, "RGBA"), near))
    assert out[500, 950, 3] == 0 and out[500, 500, 3] == 255
    kept = np.array(clean_alpha(Image.fromarray(arr, "RGBA"), (2000, 2000, 2100, 2100)))  # skeleton nowhere near
    assert kept[500, 950, 3] == 255 and kept[500, 500, 3] == 255


def test_fit_snaps_small_drift_only():
    img = figure()  # feet at row 899
    out, shift = fit_to_canvas(img, 905.0, 500.0, grounded=True)
    assert shift == 6
    assert np.asarray(out)[905, 500, 3] == 255 and np.asarray(out)[906, 500, 3] == 0
    _, shift = fit_to_canvas(img, 990.0, 500.0, grounded=True)
    assert shift == 0  # too far: left alone
    _, shift = fit_to_canvas(img, 905.0, 500.0, grounded=False)
    assert shift == 0

# ---------------------------------------------------------------------------------------------------- export


def test_names():
    assert base_name("Hiệp sĩ", "attack-a2", "Kiếm", "Đỏ") == "hiep-si_attack-a2_kiem_do"
    assert base_name("hero", "walk") == "hero_walk"


@pytest.mark.parametrize("h", config.EXPORT_HEIGHTS)
def test_export(tmp_path, h):
    frames = [figure() for _ in range(8)]
    files = export_animation(frames, tmp_path / "out", h, "hero_walk")
    assert [f.name for f in files] == [f"hero_walk_{i:02d}.png" for i in range(1, 9)] + ["hero_walk_sheet.png"]
    sheet = Image.open(files[-1])
    assert sheet.size == (8 * h, h) and sheet.mode == "RGBA"
    for f in files[:-1]:
        im = Image.open(f)
        assert im.size == (h, h) and im.mode == "RGBA"
    assert not any(p.name.startswith(".animgen") for p in (tmp_path / "out").iterdir())


def test_export_refuses_bad_input(tmp_path):
    with pytest.raises(AppError) as e:
        export_animation([figure()] * 7, tmp_path, 256, "x")
    assert e.value.code == "export_incomplete"
    with pytest.raises(AppError) as e:
        export_animation([figure()] * 8, tmp_path, 300, "x")
    assert e.value.code == "export_bad_size"


def test_sheet_play_order():
    frames = []
    for i in range(8):
        im = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        im.putpixel((5, 5), (i * 30, 0, 0, 255))
        frames.append(im)
    sheet = sprite_sheet(frames)
    assert [sheet.getpixel((i * 10 + 5, 5))[0] for i in range(8)] == [i * 30 for i in range(8)]
