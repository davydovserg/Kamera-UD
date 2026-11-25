"""Зоны и линии. В конфиге координаты нормированы 0..1 — не зависят от разрешения камеры."""
from __future__ import annotations

from datetime import datetime, time as dtime


def to_px(points, w: int, h: int):
    return [(x * w, y * h) for x, y in points]


def in_polygon(pt, poly) -> bool:
    if not poly or len(poly) < 3:
        return True  # зона не задана — весь кадр
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def side(pt, line) -> float:
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)


def crossed(p_prev, p_now, line) -> int:
    """+1 / -1 — пересёк линию в одну или другую сторону, 0 — нет."""
    a, b = side(p_prev, line), side(p_now, line)
    if a == 0 or b == 0 or (a > 0) == (b > 0):
        return 0
    return 1 if a < 0 < b else -1


def parse_hours(spec: str) -> tuple[dtime, dtime]:
    a, b = spec.split("-")
    return dtime.fromisoformat(a.strip()), dtime.fromisoformat(b.strip())


def in_hours(now: datetime, spec: str) -> bool:
    """'20:00-08:00' — через полночь тоже работает."""
    start, end = parse_hours(spec)
    t = now.time()
    return start <= t < end if start < end else (t >= start or t < end)
