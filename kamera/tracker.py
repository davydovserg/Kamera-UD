"""Простой трекер по IoU: даёт объекту постоянный id, время появления и путь."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .detector import Detection


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


@dataclass
class Track:
    id: int
    cls: str
    box: tuple
    first_seen: float
    last_seen: float
    hits: int = 1
    path: list = field(default_factory=list)  # точки «низа» рамки, последние 50


class IouTracker:
    def __init__(self, iou_min: float = 0.2, max_lost_sec: float = 8.0):
        self.iou_min = iou_min
        self.max_lost = max_lost_sec
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, dets: list[Detection], now: float | None = None) -> list[Track]:
        now = now or time.time()
        used: set[int] = set()
        for d in sorted(dets, key=lambda d: -d.conf):
            best, best_iou = None, self.iou_min
            for t in self.tracks.values():
                if t.id in used or t.cls != d.cls:
                    continue
                v = iou(t.box, d.box)
                if v > best_iou:
                    best, best_iou = t, v
            if best is None:  # анализ ~1 к/с: быстрый объект уходит дальше своей рамки — ищем ближайший
                dx_max = 1.5 * max(d.box[2] - d.box[0], d.box[3] - d.box[1])
                cx, cy = d.center
                best_dist = dx_max
                for t in self.tracks.values():
                    if t.id in used or t.cls != d.cls or now - t.last_seen > 3:
                        continue
                    tx, ty = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
                    dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                    if dist < best_dist:
                        best, best_dist = t, dist
            if best is None:
                best = Track(self._next, d.cls, d.box, now, now)
                self.tracks[best.id] = best
                self._next += 1
            else:
                best.box, best.last_seen = d.box, now
                best.hits += 1
            best.path.append(d.bottom)
            del best.path[:-50]
            d.track_id = best.id
            used.add(best.id)
        for tid in [t.id for t in self.tracks.values() if now - t.last_seen > self.max_lost]:
            del self.tracks[tid]
        return [self.tracks[i] for i in used]
