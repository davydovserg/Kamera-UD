"""Конфиг объекта: камеры, включённые сценарии, уведомления. Хранится в config.yaml."""
from __future__ import annotations

import copy
import os
import threading

import yaml

DEFAULTS = {
    "site": "Объект",
    "timezone": "Europe/Moscow",
    "panel": {"host": "0.0.0.0", "port": 8080, "user": "admin", "password": "change-me"},
    "detector": {"model": "models/yolox_s.onnx", "conf": 0.3, "nms": 0.45, "interval_sec": 1.0},
    "notify": {
        "email": {"enabled": False, "smtp_host": "", "smtp_port": 465, "ssl": True,
                  "user": "", "password": "", "from": "", "to": []},
        "telegram": {"enabled": False, "token": "", "chat_id": ""},
    },
    "journal_days": 30,
    "cameras": [],
}


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Config:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        data = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        self.data = _merge(DEFAULTS, data)

    def save(self):
        with self._lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.data, f, allow_unicode=True, sort_keys=False)
            os.replace(tmp, self.path)

    @property
    def cameras(self) -> list[dict]:
        return self.data["cameras"]

    def camera(self, cam_id: str) -> dict | None:
        return next((c for c in self.cameras if c["id"] == cam_id), None)

    def upsert_camera(self, cam: dict) -> dict:
        with self._lock:
            old = self.camera(cam["id"])
            if old:
                old.update({k: v for k, v in cam.items() if k != "scenarios"})
                cam = old
            else:
                cam.setdefault("scenarios", {})
                cam.setdefault("enabled", True)
                self.cameras.append(cam)
            self.save()
            return cam

    def delete_camera(self, cam_id: str):
        with self._lock:
            self.data["cameras"] = [c for c in self.cameras if c["id"] != cam_id]
            self.save()

    def set_scenario(self, cam_id: str, key: str, params: dict):
        with self._lock:
            cam = self.camera(cam_id)
            cam.setdefault("scenarios", {})[key] = params
            self.save()
