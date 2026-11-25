"""Движок: на каждую камеру — поток чтения и поток анализа с включёнными сценариями."""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from .camera import CameraStream
from .config import Config
from .detector import YoloxDetector
from .notify import Journal, Notifier
from .scenarios import SCENARIOS, FrameCtx, Health
from .tracker import IouTracker

log = logging.getLogger("kamera.engine")


class CameraWorker:
    def __init__(self, engine: "Engine", cam: dict):
        self.engine = engine
        self.cam = cam
        self.stream = CameraStream(cam["id"], cam["url"]).start()
        self.tracker = IouTracker(max_lost_sec=30)
        self.scenarios = {}
        self.last_dets = []
        self.fps_an = 0.0
        self.error = ""
        self._stop = threading.Event()
        self.build_scenarios()
        threading.Thread(target=self._run, name=f"an-{cam['id']}", daemon=True).start()

    def build_scenarios(self):
        out = {}
        for key, params in (self.cam.get("scenarios") or {}).items():
            cls = SCENARIOS.get(key)
            if cls and params.get("enabled") and cls.available:
                out[key] = cls(self.cam, params, self.engine.base_dir)
        self.scenarios = out

    def stop(self):
        self._stop.set()
        self.stream.stop()

    def _emit(self, key, texts, now: datetime):
        title = SCENARIOS[key].title
        name = self.cam.get("name") or self.cam["id"]
        for t in texts:
            self.engine.notifier.send(time.time(), name, title, f"{now:%d.%m %H:%M} · {name} · {title}: {t}")

    def _run(self):
        interval = float(self.engine.cfg.data["detector"].get("interval_sec", 1.0))
        last_frame_t, n, t_count = 0.0, 0, time.time()
        while not self._stop.is_set():
            t0 = time.time()
            now = datetime.now(self.engine.tz)
            try:
                for key, sc in list(self.scenarios.items()):
                    if isinstance(sc, Health):
                        self._emit(key, sc.check_stream(self.stream.online, self.stream.frame_time, t0), now)
                    self._emit(key, sc.tick(now), now)

                frame, ft = self.stream.latest()
                if frame is not None and ft != last_frame_t and self.scenarios:
                    last_frame_t = ft
                    want = set()
                    for sc in self.scenarios.values():
                        want |= sc.wanted_classes()
                    dets = self.engine.detector.detect(frame, only=want) if want else []
                    self.tracker.update(dets, t0)
                    self.last_dets = dets
                    ctx = FrameCtx(self.cam, now, t0, frame, dets, self.tracker.tracks)
                    for key, sc in list(self.scenarios.items()):
                        self._emit(key, sc.process(ctx), now)
                    n += 1
                self.error = ""
            except Exception as e:
                self.error = str(e)
                log.exception("camera %s", self.cam["id"])
            if time.time() - t_count >= 10:
                self.fps_an, n, t_count = n / (time.time() - t_count), 0, time.time()
            self._stop.wait(max(0.05, interval - (time.time() - t0)))


class Engine:
    def __init__(self, config_path: str):
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.cfg = Config(config_path)
        self.tz = ZoneInfo(self.cfg.data.get("timezone", "Europe/Moscow"))
        d = self.cfg.data["detector"]
        model = os.path.join(self.base_dir, d["model"])
        if not os.path.isfile(model):
            raise SystemExit(f"Нет модели {model}. Запустите: python scripts/download_models.py")
        self.detector = YoloxDetector(model, conf=float(d["conf"]), nms=float(d["nms"]))
        self.journal = Journal(os.path.join(self.base_dir, "data", "journal.db"),
                               int(self.cfg.data.get("journal_days", 30)))
        self.notifier = Notifier(self.cfg.data["notify"], self.journal)
        self.workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()

    def start(self):
        for cam in self.cfg.cameras:
            self.restart_camera(cam["id"])
        return self

    def restart_camera(self, cam_id: str):
        with self._lock:
            old = self.workers.pop(cam_id, None)
            if old:
                old.stop()
            cam = self.cfg.camera(cam_id)
            if cam and cam.get("enabled", True) and cam.get("url"):
                self.workers[cam_id] = CameraWorker(self, cam)

    def reload_scenarios(self, cam_id: str):
        w = self.workers.get(cam_id)
        if w:
            w.build_scenarios()
        else:
            self.restart_camera(cam_id)

    def remove_camera(self, cam_id: str):
        with self._lock:
            w = self.workers.pop(cam_id, None)
            if w:
                w.stop()
