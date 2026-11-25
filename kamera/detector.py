"""Детектор объектов YOLOX (Apache-2.0) через onnxruntime.

Модель по умолчанию — yolox_s.onnx (COCO, 80 классов). Свою модель (огонь, номера)
можно подключить тем же классом: формат выхода YOLOX, свой список классов.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 в пикселях кадра
    track_id: int = -1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def bottom(self) -> tuple[float, float]:
        """Точка «ноги/колёса» — по ней проверяем попадание в зону."""
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2


class YoloxDetector:
    def __init__(self, model_path: str, classes: list[str] | None = None,
                 conf: float = 0.3, nms: float = 0.45):
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = (int(inp.shape[2]), int(inp.shape[3]))  # h, w
        self.classes = classes or COCO
        self.conf = conf
        self.nms = nms
        self._grids = self._make_grids()
        self._lock = threading.Lock()

    def _make_grids(self):
        h, w = self.size
        grids, strides = [], []
        for s in (8, 16, 32):
            hs, ws = h // s, w // s
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((hs * ws, 1), s))
        return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(np.float32)

    def detect(self, frame: np.ndarray, only: set[str] | None = None) -> list[Detection]:
        h, w = self.size
        r = min(h / frame.shape[0], w / frame.shape[1])
        resized = cv2.resize(frame, (int(frame.shape[1] * r), int(frame.shape[0] * r)),
                             interpolation=cv2.INTER_LINEAR)
        padded = np.full((h, w, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)

        with self._lock:
            out = self.session.run(None, {self.input_name: blob})[0][0]

        grids, strides = self._grids
        out[:, :2] = (out[:, :2] + grids) * strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * strides

        scores = out[:, 4:5] * out[:, 5:]
        cls_ids = scores.argmax(1)
        cls_scores = scores[np.arange(len(scores)), cls_ids]
        keep = cls_scores >= self.conf
        if not keep.any():
            return []
        boxes = out[keep, :4]
        cls_ids, cls_scores = cls_ids[keep], cls_scores[keep]
        xywh = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 2], boxes[:, 3]], 1) / r

        result: list[Detection] = []
        for cid in np.unique(cls_ids):  # NMS по каждому классу
            name = self.classes[cid] if cid < len(self.classes) else str(cid)
            if only and name not in only:
                continue
            m = cls_ids == cid
            idx = cv2.dnn.NMSBoxes(xywh[m].tolist(), cls_scores[m].tolist(), self.conf, self.nms)
            for i in np.array(idx).flatten():
                x, y, bw, bh = xywh[m][i]
                result.append(Detection(name, float(cls_scores[m][i]),
                                        (float(x), float(y), float(x + bw), float(y + bh))))
        return result
