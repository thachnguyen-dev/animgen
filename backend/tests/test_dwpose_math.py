"""DWPose pre/post-processing (no model files needed). The ONNX models themselves are tested on the owner's PC."""

import numpy as np
import pytest
from PIL import Image

from animgen.engines import dwpose
from animgen.engines.base import EngineError


def test_letterbox_keeps_aspect():
    img = np.zeros((1000, 500, 3), np.uint8)
    out, r = dwpose.letterbox(img)
    assert out.shape == (640, 640, 3) and r == pytest.approx(0.64)
    assert out[0, 400, 0] == 114  # padding on the right


def test_nms_keeps_separate_people_close_together():
    boxes = np.array([[0, 0, 100, 200], [5, 0, 105, 200], [60, 0, 160, 200]], np.float32)
    scores = np.array([0.9, 0.8, 0.85], np.float32)
    keep = dwpose.nms(boxes, scores, 0.6)
    assert sorted(keep) == [0, 2]   # the near-duplicate is dropped, the neighbour (IoU ~0.25) is kept


def test_yolox_decode_grid():
    n = (640 // 8) ** 2 + (640 // 16) ** 2 + (640 // 32) ** 2
    raw = np.zeros((n, 6), np.float32)
    raw[1, :4] = [0.5, 0.5, 0, 0]  # grid cell (1, 0) at stride 8
    out = dwpose.yolox_decode(raw)
    assert out[1, 0] == pytest.approx(12.0) and out[1, 1] == pytest.approx(4.0) and out[1, 2] == pytest.approx(8.0)


def test_simcc_decode_and_affine_roundtrip():
    k = 133
    sx = np.zeros((k, 576), np.float32)
    sy = np.zeros((k, 768), np.float32)
    sx[:, 288] = 1.0
    sy[:, 384] = 1.0   # every point at the crop centre
    kpts, scores = dwpose.simcc_decode(sx, sy)
    assert np.allclose(kpts, [144, 192]) and np.all(scores == 1.0)
    c, s = dwpose.box_to_center_scale(np.array([100, 50, 300, 450], np.float32))
    m = dwpose.crop_affine(c, s)
    import cv2
    inv = cv2.invertAffineTransform(m)
    back = np.array([144, 192, 1.0]) @ inv.T
    assert np.allclose(back, c, atol=1e-3)


def test_wholebody_to_openpose():
    kpts = np.zeros((133, 2), np.float32)
    kpts[:17] = np.arange(17)[:, None] * [10, 1]
    scores = np.ones(133, np.float32)
    scores[16] = 0.1  # right ankle not seen
    sk = dwpose.to_openpose(kpts, scores)
    assert sk.points[0] == (0.0, 0.0)
    assert sk.points[1] == pytest.approx((55.0, 5.5))   # neck = middle of shoulders 5 and 6
    assert sk.points[2] == (60.0, 6.0)                  # R shoulder from COCO 6
    assert sk.points[10] is None                        # R ankle below threshold


def test_missing_models_reported(tmp_path):
    with pytest.raises(EngineError) as e:
        dwpose.DWPoseReader(tmp_path).read(Image.new("RGBA", (64, 64)))
    assert e.value.kind == "models_missing"


def test_diffusers_engine_reports_missing_models(tmp_path):
    from animgen.engines.diffusers_engine import DiffusersEngine
    eng = DiffusersEngine(tmp_path)
    assert len(eng.missing_models()) == 11
    with pytest.raises(EngineError) as e:
        eng.load()
    assert e.value.kind == "models_missing"


def test_session_with_real_engine_but_no_models(tmp_path):
    """The owner gets a clear error, not a crash, when the model folders are missing."""
    import synth
    from animgen.engines.diffusers_engine import DiffusersEngine
    from animgen.session import Session
    from animgen.tasks import TaskRunner
    s = Session(DiffusersEngine(tmp_path / "models"), tmp_path / "data")
    hero = synth.character(tmp_path / "hero.png")
    ch = s.set_character(str(hero))        # pose reader missing -> standard proportions, facing assumed
    assert ch.facing_source == "assumed"
    r = TaskRunner()
    t = r.wait(r.start("g", lambda task: s.generate(task, "idle", None)).id)
    assert t.state == "failed" and t.error["code"] == "models_missing"
    s.close()
