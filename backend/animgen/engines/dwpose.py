"""DWPose body skeleton reader (ONNX Runtime, CPU), converted to the 18 OpenPose points ControlNet uses.

Lessons carried over from chargen (§14, Inherited—Verify):
- the person detector (YOLOX-L) takes BGR input;
- NMS overlap 0.6, otherwise two people close together merge into one box;
- the pose model takes RGB normalised with ImageNet mean/std, 288x384 crops, SimCC decoding.

The pre- and post-processing functions are pure numpy and tested in the cloud build; the ONNX sessions
themselves can only be tested with the model files on the owner's PC.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..skeleton import Skeleton
from .base import EngineError

DET_SIZE = (640, 640)          # (w, h)
POSE_SIZE = (288, 384)         # (w, h)
SIMCC_SPLIT = 2.0
NMS_THRESHOLD = 0.6
SCORE_THRESHOLD = 0.3
KPT_THRESHOLD = 0.3
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)

# COCO-WholeBody body points (first 17) -> OpenPose 18 order. Index 1 (neck) is the shoulder midpoint.
COCO_TO_OPENPOSE = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]


def letterbox(img_bgr: np.ndarray, size=DET_SIZE) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    r = min(size[1] / h, size[0] / w)
    resized = cv2.resize(img_bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    out = np.full((size[1], size[0], 3), 114, np.uint8)
    out[: resized.shape[0], : resized.shape[1]] = resized
    return out, r


def yolox_decode(raw: np.ndarray, size=DET_SIZE, strides=(8, 16, 32)) -> np.ndarray:
    """Raw YOLOX head output (N, 5 + classes) -> boxes in the letterboxed image (cx, cy, w, h, ...)."""
    grids, expanded = [], []
    for s in strides:
        gh, gw = size[1] // s, size[0] // s
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        expanded.append(np.full((gh * gw, 1), s))
    grid = np.concatenate(grids, 0).astype(np.float32)
    stride = np.concatenate(expanded, 0).astype(np.float32)
    out = raw.copy().astype(np.float32)
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = NMS_THRESHOLD) -> list[int]:
    """boxes: (N, 4) x1 y1 x2 y2."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def person_boxes(decoded: np.ndarray, ratio: float) -> np.ndarray:
    """Person boxes (x1, y1, x2, y2) in original image pixels, best first."""
    obj = decoded[:, 4]
    person = decoded[:, 5] if decoded.shape[1] > 5 else np.ones_like(obj)
    score = obj * person
    m = score > SCORE_THRESHOLD
    if not m.any():
        return np.zeros((0, 4), np.float32)
    d, sc = decoded[m], score[m]
    boxes = np.stack([d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2, d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2], 1)
    boxes /= ratio
    keep = nms(boxes, sc)
    return boxes[keep]


def box_to_center_scale(box: np.ndarray, aspect: float = POSE_SIZE[0] / POSE_SIZE[1], padding: float = 1.25):
    x1, y1, x2, y2 = box
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    w, h = (x2 - x1) * padding, (y2 - y1) * padding
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return c, np.array([w, h], np.float32)


def crop_affine(center: np.ndarray, scale: np.ndarray, out=POSE_SIZE) -> np.ndarray:
    """Affine that maps the box (center, scale) onto the pose input of size ``out``."""
    src = np.array([center, center + [0, -scale[1] / 2], center + [-scale[0] / 2, 0]], np.float32)
    dst = np.array([[out[0] / 2, out[1] / 2], [out[0] / 2, 0], [0, out[1] / 2]], np.float32)
    return cv2.getAffineTransform(src, dst)


def simcc_decode(simcc_x: np.ndarray, simcc_y: np.ndarray, split: float = SIMCC_SPLIT):
    """(K, Wx), (K, Wy) -> keypoints (K, 2) in pose-input pixels and scores (K,)."""
    x = simcc_x.argmax(axis=1).astype(np.float32)
    y = simcc_y.argmax(axis=1).astype(np.float32)
    sx, sy = simcc_x.max(axis=1), simcc_y.max(axis=1)
    scores = np.minimum(sx, sy)
    kpts = np.stack([x, y], 1) / split
    kpts[scores <= 0] = -1
    return kpts, scores


def to_openpose(kpts: np.ndarray, scores: np.ndarray, thr: float = KPT_THRESHOLD) -> Skeleton:
    pts: list = []
    conf: list[float] = []
    for src in COCO_TO_OPENPOSE:
        if src is None:  # neck
            ok = scores[5] > thr and scores[6] > thr
            pts.append(((kpts[5][0] + kpts[6][0]) / 2, (kpts[5][1] + kpts[6][1]) / 2) if ok else None)
            conf.append(float(min(scores[5], scores[6])) if ok else 0.0)
        else:
            ok = scores[src] > thr
            pts.append((float(kpts[src][0]), float(kpts[src][1])) if ok else None)
            conf.append(float(scores[src]) if ok else 0.0)
    return Skeleton(pts, conf)


class DWPoseReader:
    """Reads one figure's skeleton. Model files: ``<models>/dwpose/yolox_l.onnx`` and ``dw-ll_ucoco_384.onnx``."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self._det = None
        self._pose = None

    def _load(self) -> None:
        if self._det is not None:
            return
        det, pose = self.folder / "yolox_l.onnx", self.folder / "dw-ll_ucoco_384.onnx"
        missing = [str(p) for p in (det, pose) if not p.is_file()]
        if missing:
            raise EngineError("models_missing", ", ".join(missing))
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self._det = ort.InferenceSession(str(det), opts, providers=providers)
        self._pose = ort.InferenceSession(str(pose), opts, providers=providers)

    def read(self, image: Image.Image) -> Skeleton | None:
        self._load()
        # Flatten transparency onto white: DWPose was trained on photos/illustrations, not alpha.
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        rgb = np.asarray(bg.convert("RGB"))
        bgr = rgb[..., ::-1].copy()

        lb, ratio = letterbox(bgr)
        inp = lb.transpose(2, 0, 1)[None].astype(np.float32)  # BGR, 0..255, no normalisation
        raw = self._det.run(None, {self._det.get_inputs()[0].name: inp})[0][0]
        if raw.shape[-1] >= 6 and raw[:, 2:4].max() < 20:   # raw head output: decode it
            raw = yolox_decode(raw)
        boxes = person_boxes(raw, ratio)
        if len(boxes) == 0:
            # A figure on a plain background can be missed by the detector; use the whole opaque area.
            a = np.asarray(rgba.getchannel("A"))
            ys, xs = np.nonzero(a > 128)
            if len(xs) == 0:
                return None
            boxes = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], np.float32)
        center, scale = box_to_center_scale(boxes[0])
        m = crop_affine(center, scale)
        crop = cv2.warpAffine(rgb, m, POSE_SIZE, flags=cv2.INTER_LINEAR)
        x = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        outs = self._pose.run(None, {self._pose.get_inputs()[0].name: x})
        simcc_x, simcc_y = outs[0][0], outs[1][0]
        kpts, scores = simcc_decode(simcc_x, simcc_y)
        inv = cv2.invertAffineTransform(m)
        k = np.concatenate([kpts, np.ones((len(kpts), 1), np.float32)], 1) @ inv.T
        return to_openpose(k, scores)
