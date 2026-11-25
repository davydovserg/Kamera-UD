"""Сценарии KADR. Каждый сценарий — класс в реестре SCENARIOS.

Сценарий получает кадр с найденными объектами (FrameCtx) и возвращает список текстов событий.
Сообщение собирает движок: «14:32 · Камера 3 · Сценарий: текст».
Новый сценарий = новый класс с @register — панель покажет его автоматически.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .geometry import crossed, in_hours, in_polygon, to_px

VEHICLES = ["car", "truck", "bus", "motorcycle"]


@dataclass
class FrameCtx:
    cam: dict
    now: datetime
    ts: float
    frame: np.ndarray
    dets: list
    tracks: dict = field(default_factory=dict)

    @property
    def wh(self):
        return self.frame.shape[1], self.frame.shape[0]


SCENARIOS: dict[str, type["Scenario"]] = {}


def register(cls):
    SCENARIOS[cls.key] = cls
    return cls


class Scenario:
    key = ""
    title = ""
    description = ""
    needs = None          # "zone" | "line" | None — что нарисовать на кадре
    classes: list = []    # какие объекты нужны детектору
    params: list = []     # [{name, label, type, default}] — поля в панели
    model_param = None    # имя параметра с путём к своей модели (огонь, номера)
    available = True

    def __init__(self, cam: dict, cfg: dict, base_dir: str = "."):
        self.cam = cam
        self.base_dir = base_dir
        self.cfg = {p["name"]: p["default"] for p in self.params}
        self.cfg.update({k: v for k, v in cfg.items() if k not in ("enabled",)})
        self.zone = cfg.get("zone") or []
        self.line = cfg.get("line") or []
        self._alerted: dict[int, float] = {}
        self._last_alert = 0.0

    # --- что вернуть панели ---
    @classmethod
    def meta(cls) -> dict:
        return {"key": cls.key, "title": cls.title, "description": cls.description,
                "needs": cls.needs, "params": cls.params, "available": cls.available,
                "model_param": cls.model_param}

    def status(self) -> str:
        if self.model_param:
            path = self.cfg.get(self.model_param) or ""
            if not path or not os.path.isfile(os.path.join(self.base_dir, path)):
                return "нужна модель: положите файл и укажите путь"
        return "работает"

    # --- работа ---
    def wanted_classes(self) -> set:
        return set(self.cfg.get("classes") or self.classes)

    def process(self, ctx: FrameCtx) -> list[str]:
        return []

    def tick(self, now: datetime) -> list[str]:
        """Вызывается раз в ~секунду без кадра: сводки по расписанию."""
        return []

    def _once_per_track(self, tid: int, now: float, ttl: float = 3600) -> bool:
        for k in [k for k, t in self._alerted.items() if now - t > ttl]:
            del self._alerted[k]
        if tid in self._alerted:
            return False
        self._alerted[tid] = now
        return True

    def _cooldown(self, now: float) -> bool:
        if now - self._last_alert < float(self.cfg.get("cooldown_sec", 60)):
            return False
        self._last_alert = now
        return True


def _summary_due(state: dict, now: datetime, at: str) -> bool:
    if not at:
        return False
    hh, mm = map(int, at.split(":"))
    day = now.date().isoformat()
    if (now.hour, now.minute) >= (hh, mm) and state.get("sent") != day:
        state["sent"] = day
        return True
    return False


@register
class Intrusion(Scenario):
    key = "intrusion"
    title = "Человек в зоне"
    description = ("Человек (или машина) в выделенной зоне. С расписанием — «нерабочее время»; "
                   "без расписания — всегда (крыша, забор, техзона).")
    needs = "zone"
    classes = ["person"]
    params = [
        {"name": "hours", "label": "Часы охраны (пусто — всегда)", "type": "text", "default": "20:00-08:00"},
        {"name": "min_sec", "label": "Сколько секунд в зоне до тревоги", "type": "number", "default": 3},
        {"name": "classes", "label": "Объекты", "type": "classes", "default": ["person"]},
        {"name": "cooldown_sec", "label": "Пауза между сообщениями, с", "type": "number", "default": 120},
    ]

    def process(self, ctx):
        hours = (self.cfg.get("hours") or "").strip()
        if hours and not in_hours(ctx.now, hours):
            return []
        w, h = ctx.wh
        zone = to_px(self.zone, w, h)
        for d in ctx.dets:
            if d.cls not in self.wanted_classes() or not in_polygon(d.bottom, zone):
                continue
            t = ctx.tracks.get(d.track_id)
            if not t or ctx.ts - t.first_seen < float(self.cfg["min_sec"]):
                continue
            if self._once_per_track(t.id, ctx.ts) and self._cooldown(ctx.ts):
                who = "человек" if d.cls == "person" else d.cls
                return [f"{who} в зоне" + (" в нерабочее время" if hours else "")]
        return []


@register
class Parking(Scenario):
    key = "parking"
    title = "Стоянка в запрещённой зоне"
    description = "Машина стоит в зоне (пожарный проезд, разворотная площадка, у ворот) дольше заданного."
    needs = "zone"
    classes = VEHICLES
    params = [
        {"name": "max_minutes", "label": "Сколько минут можно стоять", "type": "number", "default": 10},
        {"name": "classes", "label": "Объекты", "type": "classes", "default": VEHICLES},
    ]

    def process(self, ctx):
        w, h = ctx.wh
        zone = to_px(self.zone, w, h)
        out = []
        for d in ctx.dets:
            if d.cls not in self.wanted_classes() or not in_polygon(d.bottom, zone):
                continue
            t = ctx.tracks.get(d.track_id)
            if not t:
                continue
            minutes = (ctx.ts - t.first_seen) / 60
            if minutes >= float(self.cfg["max_minutes"]) and self._once_per_track(t.id, ctx.ts, ttl=6 * 3600):
                out.append(f"машина стоит в зоне {int(minutes)} мин")
        return out


@register
class LineCross(Scenario):
    key = "line_cross"
    title = "Заезд / выезд"
    description = "Считает машины через линию ворот: въезд и выезд. Сообщение на каждый проезд или только сводка."
    needs = "line"
    classes = VEHICLES
    params = [
        {"name": "in_label", "label": "Название направления «туда»", "type": "text", "default": "въезд"},
        {"name": "out_label", "label": "Название направления «обратно»", "type": "text", "default": "выезд"},
        {"name": "notify_each", "label": "Сообщать о каждом проезде", "type": "bool", "default": False},
        {"name": "summary_at", "label": "Сводка за день в (ЧЧ:ММ, пусто — нет)", "type": "text", "default": "18:00"},
        {"name": "classes", "label": "Объекты", "type": "classes", "default": VEHICLES},
    ]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.counts = {"in": 0, "out": 0}
        self.log: list[str] = []
        self._sum = {}
        self._crossed: dict[int, int] = {}

    def process(self, ctx):
        if len(self.line) != 2:
            return []
        w, h = ctx.wh
        line = to_px(self.line, w, h)
        out = []
        for d in ctx.dets:
            t = ctx.tracks.get(d.track_id)
            if d.cls not in self.wanted_classes() or not t or len(t.path) < 2:
                continue
            s = crossed(t.path[-2], t.path[-1], line)
            if s and self._crossed.get(t.id) != s:
                self._crossed[t.id] = s
                key = "in" if s > 0 else "out"
                self.counts[key] += 1
                label = self.cfg["in_label"] if s > 0 else self.cfg["out_label"]
                self.log.append(f"{ctx.now:%H:%M} {label}")
                if self.cfg["notify_each"]:
                    out.append(label)
        return out

    def tick(self, now):
        if _summary_due(self._sum, now, self.cfg.get("summary_at")):
            c, self.counts, self.log = self.counts, {"in": 0, "out": 0}, []
            return [f"сводка за день: {self.cfg['in_label']} {c['in']}, {self.cfg['out_label']} {c['out']}"]
        return []


@register
class Loading(Scenario):
    key = "loading"
    title = "Сводка погрузок"
    description = "Грузовик стоит в зоне погрузки дольше N минут — считаем погрузку. Вечером — список за день."
    needs = "zone"
    classes = ["truck", "car", "bus"]
    params = [
        {"name": "min_minutes", "label": "Минимум минут стоянки = погрузка", "type": "number", "default": 5},
        {"name": "notify_each", "label": "Сообщать о каждой погрузке", "type": "bool", "default": False},
        {"name": "summary_at", "label": "Сводка за день в (ЧЧ:ММ)", "type": "text", "default": "18:00"},
        {"name": "classes", "label": "Объекты", "type": "classes", "default": ["truck"]},
    ]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.active: dict[int, datetime] = {}
        self.done: list[str] = []
        self._sum = {}

    def process(self, ctx):
        w, h = ctx.wh
        zone = to_px(self.zone, w, h)
        out, seen = [], set()
        for d in ctx.dets:
            t = ctx.tracks.get(d.track_id)
            if d.cls not in self.wanted_classes() or not t or not in_polygon(d.bottom, zone):
                continue
            seen.add(t.id)
            if t.id not in self.active and ctx.ts - t.first_seen >= float(self.cfg["min_minutes"]) * 60:
                self.active[t.id] = datetime.fromtimestamp(t.first_seen, ctx.now.tzinfo)
                if self.cfg["notify_each"]:
                    out.append(f"началась погрузка (с {self.active[t.id]:%H:%M})")
        for tid in [k for k in self.active if k not in seen and k not in ctx.tracks]:
            start = self.active.pop(tid)
            mins = int((ctx.now - start).total_seconds() // 60)
            self.done.append(f"погрузка — {self.cam.get('name', self.cam['id'])} — {start:%H:%M} ({mins} мин)")
        return out

    def tick(self, now):
        if _summary_due(self._sum, now, self.cfg.get("summary_at")):
            items, self.done = self.done, []
            if not items:
                return ["сводка погрузок за день: погрузок не было"]
            return [f"сводка погрузок за день ({len(items)}):\n" + "\n".join(items)]
        return []


@register
class Fire(Scenario):
    key = "fire"
    title = "Огонь"
    description = ("Открытое пламя. Базовый режим работает сразу: цвет пламени + мерцание (лампы и фары "
                   "не мерцают так), подтверждение по нескольким кадрам. Если положить обученную модель "
                   "models/fire.onnx (YOLOX, классы fire, smoke) — используется она.")
    needs = "zone"
    params = [
        {"name": "model", "label": "Файл модели огня (нет файла — базовый режим)", "type": "text", "default": "models/fire.onnx"},
        {"name": "conf", "label": "Уверенность модели (0–1)", "type": "number", "default": 0.45},
        {"name": "min_area", "label": "Мин. площадь пламени, % кадра (базовый режим)", "type": "number", "default": 0.05},
        {"name": "confirm", "label": "Кадров с огнём из последних 5", "type": "number", "default": 3},
        {"name": "cooldown_sec", "label": "Пауза между сообщениями, с", "type": "number", "default": 300},
    ]
    _models: dict = {}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.hist = deque(maxlen=5)
        self.prev_mask = None

    def status(self):
        return "работает (модель)" if self._detector() else "работает (базовый режим, без модели)"

    def _detector(self):
        rel = (self.cfg.get("model") or "").strip()
        path = os.path.join(self.base_dir, rel) if rel else ""
        if not path or not os.path.isfile(path):
            return None
        if path not in Fire._models:
            from .detector import YoloxDetector
            Fire._models[path] = YoloxDetector(path, classes=["fire", "smoke"], conf=float(self.cfg["conf"]))
        return Fire._models[path]

    def _basic(self, frame, zone_px) -> bool:
        import cv2
        k = 320 / frame.shape[1]
        small = cv2.resize(frame, (320, int(frame.shape[0] * k)))
        b, g, r = [c.astype(np.int16) for c in cv2.split(small)]
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = (r > 180) & (r >= g) & (g > b) & ((r - b) > 70) & (hsv[..., 1] > 70) & (hsv[..., 2] > 170)
        if zone_px and len(zone_px) >= 3:
            zm = np.zeros(mask.shape, np.uint8)
            cv2.fillPoly(zm, [np.array([(x * k, y * k) for x, y in zone_px], np.int32)], 1)
            mask &= zm.astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        prev, self.prev_mask = self.prev_mask, mask
        if mask.mean() * 100 < float(self.cfg["min_area"]) or prev is None:
            return False
        union = (mask | prev).sum()
        flicker = (mask ^ prev).sum() / union if union else 0
        return flicker > 0.25  # пламя меняет форму между кадрами, лампа и фара — нет

    def process(self, ctx):
        w, h = ctx.wh
        zone = to_px(self.zone, w, h)
        det = self._detector()
        if det:
            seen = any(in_polygon(d.center, zone) for d in det.detect(ctx.frame, only={"fire"}))
        else:
            seen = self._basic(ctx.frame, zone)
        self.hist.append(seen)
        if sum(self.hist) >= int(self.cfg["confirm"]) and self._cooldown(ctx.ts):
            return ["обнаружен огонь"]
        return []


LAT2CYR = str.maketrans("ABEKMHOPCTYX", "АВЕКМНОРСТУХ")
CYR2LAT = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")


def norm_plate(t: str) -> str:
    return "".join(ch for ch in (t or "").upper().translate(CYR2LAT) if ch.isalnum())


@register
class Plates(Scenario):
    key = "plates"
    title = "Номера машин"
    description = ("Читает номера машин в зоне, ведёт журнал и сверяет со списком жителей. "
                   "Модели номеров и OCR (MIT) скачиваются сами при первом включении. "
                   "Камера: номер от 120 px по ширине, угол до 30°.")
    needs = "zone"
    classes = VEHICLES
    params = [
        {"name": "residents", "label": "Номера жителей через запятую (пусто — без сверки)", "type": "text", "default": ""},
        {"name": "notify", "label": "Сообщать", "type": "select", "default": "unknown",
         "options": [["unknown", "только о чужих"], ["all", "о каждой машине"], ["none", "не сообщать, только сводка"]]},
        {"name": "min_reads", "label": "Сколько раз прочитать одинаково", "type": "number", "default": 2},
        {"name": "repeat_min", "label": "Не повторять тот же номер, мин", "type": "number", "default": 10},
        {"name": "summary_at", "label": "Сводка номеров за день в (ЧЧ:ММ, пусто — нет)", "type": "text", "default": ""},
    ]
    _alpr = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reads: dict[str, list[float]] = {}
        self.last_sent: dict[str, float] = {}
        self.day: list[str] = []
        self._sum = {}

    @classmethod
    def alpr(cls):
        if cls._alpr is None:
            from fast_alpr import ALPR
            cls._alpr = ALPR(ocr_model="global-plates-mobile-vit-v2-model", ocr_device="auto")
        return cls._alpr

    def status(self):
        try:
            import fast_alpr  # noqa: F401
            return "работает"
        except ImportError:
            return "нужен пакет: pip install fast-alpr"

    def residents(self) -> set:
        return {norm_plate(x) for x in str(self.cfg.get("residents") or "").split(",") if x.strip()}

    def process(self, ctx):
        w, h = ctx.wh
        zone = to_px(self.zone, w, h)
        if not any(d.cls in self.wanted_classes() and in_polygon(d.bottom, zone) for d in ctx.dets):
            return []
        out = []
        for r in self.alpr().predict(ctx.frame):
            if not r.ocr or not r.ocr.text:
                continue
            bb = r.detection.bounding_box
            if zone and not in_polygon(((bb.x1 + bb.x2) / 2, (bb.y1 + bb.y2) / 2), zone):
                continue
            conf = r.ocr.confidence
            conf = float(np.mean(conf)) if isinstance(conf, (list, tuple, np.ndarray)) else float(conf or 0)
            plate = norm_plate(r.ocr.text)
            if len(plate) < 6 or conf < 0.5:
                continue
            hist = [t for t in self.reads.get(plate, []) if ctx.ts - t < 60] + [ctx.ts]
            self.reads[plate] = hist
            if len(hist) < int(self.cfg["min_reads"]):
                continue
            if ctx.ts - self.last_sent.get(plate, 0) < float(self.cfg["repeat_min"]) * 60:
                continue
            self.last_sent[plate] = ctx.ts
            shown = plate.translate(LAT2CYR)
            res = self.residents()
            known = (plate in res) if res else None
            tag = "" if known is None else (" — житель" if known else " — НЕТ в списке жителей")
            self.day.append(f"{ctx.now:%H:%M} {shown}{tag}")
            mode = self.cfg["notify"]
            if mode == "all" or (mode == "unknown" and known is False):
                out.append(f"номер {shown}{tag}")
        for p in [p for p, v in self.reads.items() if ctx.ts - v[-1] > 120]:
            del self.reads[p]
        return out

    def tick(self, now):
        if _summary_due(self._sum, now, self.cfg.get("summary_at")):
            items, self.day = self.day, []
            return [f"номера за день ({len(items)}):\n" + ("\n".join(items) or "машин не было")]
        return []


CARRY = ["suitcase", "backpack", "handbag", "bicycle", "chair", "tv", "laptop", "microwave",
         "refrigerator", "oven", "couch", "bench"]


@register
class CarryOut(Scenario):
    key = "carry_out"
    title = "Вынос"
    description = ("Человек выходит через линию проходной или ворот с предметом: сумка, техника, мебель — "
                   "или несёт что-то крупное перед собой (широкий силуэт). "
                   "Стрелка на линии показывает направление «туда».")
    needs = "line"
    classes = ["person"] + CARRY
    params = [
        {"name": "exit_dir", "label": "Выход — это", "type": "select", "default": "in",
         "options": [["in", "по стрелке («туда»)"], ["out", "против стрелки"]]},
        {"name": "hours", "label": "Когда следить (пусто — всегда)", "type": "text", "default": ""},
        {"name": "wide_ratio", "label": "Широкий силуэт: ширина/высота больше", "type": "number", "default": 0.62},
        {"name": "cooldown_sec", "label": "Пауза между сообщениями, с", "type": "number", "default": 30},
    ]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._done: dict[int, float] = {}
        self.ratios: dict[int, deque] = {}

    def process(self, ctx):
        if len(self.line) != 2:
            return []
        hours = (self.cfg.get("hours") or "").strip()
        if hours and not in_hours(ctx.now, hours):
            return []
        from .tracker import iou
        w, h = ctx.wh
        line = to_px(self.line, w, h)
        want = 1 if self.cfg["exit_dir"] == "in" else -1
        items = [d for d in ctx.dets if d.cls in CARRY]
        out = []
        for d in ctx.dets:
            t = ctx.tracks.get(d.track_id)
            if d.cls != "person" or not t:
                continue
            x1, y1, x2, y2 = d.box
            self.ratios.setdefault(t.id, deque(maxlen=5)).append((x2 - x1) / max(1.0, y2 - y1))
            if len(t.path) < 2 or t.id in self._done or crossed(t.path[-2], t.path[-1], line) != want:
                continue
            self._done[t.id] = ctx.ts
            pad = 0.3 * (x2 - x1)
            near = [i.cls for i in items if iou(i.box, d.box) > 0.05
                    or (x1 - pad < i.center[0] < x2 + pad and y1 < i.center[1] < y2)]
            wide = float(np.median(self.ratios[t.id])) > float(self.cfg["wide_ratio"])
            if (near or wide) and self._cooldown(ctx.ts):
                what = ", ".join(sorted(set(near))) if near else "крупный предмет в руках"
                out.append(f"человек вышел с предметом: {what}")
        for k in [k for k, v in self._done.items() if ctx.ts - v > 600]:
            self._done.pop(k, None)
            self.ratios.pop(k, None)
        return out


@register
class Health(Scenario):
    key = "health"
    title = "Исправность камеры"
    description = "Камера не в сети, картинка зависла, закрыта или слишком тёмная. Сообщение и при восстановлении."
    params = [
        {"name": "offline_sec", "label": "Нет сигнала дольше, с", "type": "number", "default": 120},
        {"name": "frozen_sec", "label": "Картинка не меняется дольше, с", "type": "number", "default": 300},
        {"name": "dark_sec", "label": "Темно/закрыто дольше, с", "type": "number", "default": 600},
    ]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prev_small = None
        self.since = {"frozen": None, "dark": None}
        self.problem: str | None = None

    def check_stream(self, online: bool, frame_time: float, now_ts: float) -> list[str]:
        if not online and frame_time and now_ts - frame_time > float(self.cfg["offline_sec"]):
            return self._set("камера не в сети")
        if not online and not frame_time and now_ts - self._started() > float(self.cfg["offline_sec"]):
            return self._set("камера не в сети (не было ни одного кадра)")
        return []

    def _started(self):
        if not hasattr(self, "_t0"):
            self._t0 = time.time()
        return self._t0

    def _set(self, problem: str | None) -> list[str]:
        if problem == self.problem:
            return []
        old, self.problem = self.problem, problem
        if problem:
            return [problem]
        return [f"камера снова работает (было: {old})"] if old else []

    def process(self, ctx):
        import cv2
        small = cv2.cvtColor(cv2.resize(ctx.frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        frozen = self.prev_small is not None and float(np.abs(small - self.prev_small).mean()) < 0.3
        self.prev_small = small
        dark = float(small.mean()) < 12 or float(small.std()) < 3
        checks = [("frozen", frozen, "картинка зависла", self.cfg["frozen_sec"]),
                  ("dark", dark, "камера закрыта или слишком темно", self.cfg["dark_sec"])]
        for key, bad, text, limit in checks:
            if not bad:
                self.since[key] = None
                continue
            self.since[key] = self.since[key] or ctx.ts
            if ctx.ts - self.since[key] >= float(limit):
                return self._set(text)
        return self._set(None)


PRESETS = {
    "ЖК": ["fire", "parking", "plates", "intrusion", "health"],
    "Склад": ["line_cross", "intrusion", "loading", "carry_out", "health"],
    "Стройплощадка": ["intrusion", "line_cross", "health"],
}
