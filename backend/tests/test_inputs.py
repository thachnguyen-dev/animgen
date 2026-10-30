import numpy as np
import pytest
from PIL import Image

import synth
from animgen.errors import AppError
from animgen.inputs import check_character, looks_like_checkerboard


def code_of(fn):
    with pytest.raises(AppError) as e:
        fn()
    return e.value.code


def test_valid_character(hero):
    ch = check_character(hero)
    assert ch.height > 600
    assert ch.foot_line == ch.bounds.bottom - 1


def test_checkerboard_is_not_transparency(tmp_path):
    p = synth.checkerboard_character(tmp_path / "cb.png")
    assert code_of(lambda: check_character(p)) == "input_checkerboard"


@pytest.mark.parametrize("block", [8, 10, 25, 32])
def test_checkerboard_block_sizes(tmp_path, block):
    p = synth.checkerboard_character(tmp_path / "cb.png", block=block)
    assert code_of(lambda: check_character(p)) == "input_checkerboard"


def test_checkerboard_saved_as_rgba_with_full_alpha(tmp_path):
    p = synth.checkerboard_character(tmp_path / "cb.png")
    Image.open(p).convert("RGBA").save(tmp_path / "cb2.png")
    assert code_of(lambda: check_character(tmp_path / "cb2.png")) == "input_checkerboard"


def test_opaque_picture(tmp_path):
    p = synth.opaque(tmp_path / "o.png")
    assert code_of(lambda: check_character(p)) == "input_no_alpha"


def test_alpha_but_nothing_transparent(tmp_path):
    Image.new("RGBA", (200, 300), (10, 200, 30, 255)).save(tmp_path / "f.png")
    assert code_of(lambda: check_character(tmp_path / "f.png")) == "input_no_transparency"


def test_flat_background_not_mistaken_for_checkerboard():
    flat = np.full((300, 300, 3), 240, np.uint8)
    assert not looks_like_checkerboard(flat)


def test_empty(tmp_path):
    Image.new("RGBA", (300, 300), (0, 0, 0, 0)).save(tmp_path / "e.png")
    assert code_of(lambda: check_character(tmp_path / "e.png")) == "input_empty"


def test_two_figures(tmp_path):
    p = synth.two_figures(tmp_path / "two.png")
    assert code_of(lambda: check_character(p)) == "input_many_figures"


def test_small_detached_bits_are_allowed(tmp_path, hero):
    img = Image.open(hero).convert("RGBA")
    arr = np.array(img)
    arr[20:26, 20:26] = (255, 0, 0, 255)  # a stray hair strand far from the body
    Image.fromarray(arr).save(tmp_path / "h.png")
    check_character(tmp_path / "h.png")


def test_unreadable_and_missing(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    assert code_of(lambda: check_character(tmp_path / "x.png")) == "input_unreadable"
    assert code_of(lambda: check_character(tmp_path / "nope.png")) == "input_not_found"


def test_too_small(tmp_path):
    Image.new("RGBA", (30, 30), (0, 0, 0, 0)).save(tmp_path / "s.png")
    assert code_of(lambda: check_character(tmp_path / "s.png")) == "input_too_small"


def test_messages_are_vietnamese(tmp_path):
    p = synth.opaque(tmp_path / "o.png")
    with pytest.raises(AppError) as e:
        check_character(p)
    assert "trong suốt" in e.value.message
