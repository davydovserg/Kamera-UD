"""Панель управления: камеры, выбор сценариев, зоны на кадре, журнал."""
from __future__ import annotations

import os
import re
import secrets

import cv2
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from .camera import VENDORS, build_url, mask_url, probe
from .detector import COCO
from .engine import Engine
from .scenarios import PRESETS, SCENARIOS

STATIC = os.path.join(os.path.dirname(__file__), "static")


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="KADR Камеры", docs_url=None, redoc_url=None)
    security = HTTPBasic()
    cfg = engine.cfg

    def auth(c: HTTPBasicCredentials = Depends(security)):
        p = cfg.data["panel"]
        ok = secrets.compare_digest(c.username, str(p["user"])) and \
            secrets.compare_digest(c.password, str(os.environ.get("KADR_PANEL_PASSWORD", p["password"])))
        if not ok:
            raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})

    A = [Depends(auth)]

    def cam_or_404(cam_id):
        cam = cfg.camera(cam_id)
        if not cam:
            raise HTTPException(404, "нет такой камеры")
        return cam

    @app.get("/", dependencies=A)
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/meta", dependencies=A)
    def meta():
        return {"site": cfg.data["site"], "scenarios": [c.meta() for c in SCENARIOS.values()],
                "presets": PRESETS, "vendors": list(VENDORS), "classes": COCO}

    @app.get("/api/cameras", dependencies=A)
    def cameras():
        out = []
        for c in cfg.cameras:
            w = engine.workers.get(c["id"])
            sc_status = {k: s.status() for k, s in (w.scenarios.items() if w else [])}
            out.append({
                "id": c["id"], "name": c.get("name", c["id"]), "url": mask_url(c.get("url", "")),
                "enabled": c.get("enabled", True), "scenarios": c.get("scenarios", {}),
                "online": bool(w and w.stream.online), "error": (w and (w.stream.error or w.error)) or "",
                "fps": round(w.fps_an, 2) if w else 0, "scenario_status": sc_status,
            })
        return out

    @app.post("/api/cameras", dependencies=A)
    def add_camera(body: dict):
        name = (body.get("name") or "").strip()
        url = (body.get("url") or "").strip()
        cam_id = body.get("id") or re.sub(r"\W+", "-", name.lower()).strip("-") or f"cam{len(cfg.cameras) + 1}"
        old = cfg.camera(cam_id)
        if old and "***" in url:  # из панели пришла маска — пароль не менялся
            url = old["url"]
        if not url:
            raise HTTPException(400, "нужна ссылка на поток")
        cam = cfg.upsert_camera({"id": cam_id, "name": name or cam_id, "url": url,
                                 "enabled": body.get("enabled", True)})
        preset = body.get("preset")
        if preset in PRESETS and not old:
            for key in PRESETS[preset]:
                if SCENARIOS[key].available:
                    cfg.set_scenario(cam_id, key, {"enabled": True})
        engine.restart_camera(cam_id)
        return {"id": cam["id"]}

    @app.post("/api/nvr", dependencies=A)
    def add_nvr(body: dict):
        """Регистратор целиком: каналы с ch_from по ch_to."""
        vendor = body["vendor"]
        if vendor not in VENDORS:
            raise HTTPException(400, "неизвестный производитель")
        prefix = body.get("prefix") or "Камера"
        added = []
        for ch in range(int(body.get("ch_from", 1)), int(body.get("ch_to", 1)) + 1):
            url = build_url(vendor, body["host"], body["user"], body["password"], ch, int(body.get("port", 554)))
            r = add_camera({"id": f"{re.sub(r'[^0-9a-z]+', '-', body['host'].lower())}-ch{ch}",
                            "name": f"{prefix} {ch}", "url": url, "preset": body.get("preset")})
            added.append(r["id"])
        return {"added": added}

    @app.post("/api/probe", dependencies=A)
    def probe_url(body: dict):
        url = body.get("url", "")
        if body.get("id") and "***" in url:
            url = cam_or_404(body["id"])["url"]
        ok, info = probe(url)
        return {"ok": ok, "info": info}

    @app.delete("/api/cameras/{cam_id}", dependencies=A)
    def delete_camera(cam_id: str):
        cam_or_404(cam_id)
        engine.remove_camera(cam_id)
        cfg.delete_camera(cam_id)
        return {"ok": True}

    @app.put("/api/cameras/{cam_id}/scenarios/{key}", dependencies=A)
    def set_scenario(cam_id: str, key: str, params: dict):
        cam_or_404(cam_id)
        if key not in SCENARIOS:
            raise HTTPException(404, "нет такого сценария")
        cfg.set_scenario(cam_id, key, params)
        engine.reload_scenarios(cam_id)
        return {"ok": True}

    @app.get("/api/cameras/{cam_id}/snapshot.jpg", dependencies=A)
    def snapshot(cam_id: str, boxes: int = 1):
        """Текущий кадр с рамками. Ничего не сохраняется на диск."""
        cam_or_404(cam_id)
        w = engine.workers.get(cam_id)
        frame, _ = w.stream.latest() if w else (None, 0)
        if frame is None:
            raise HTTPException(503, "нет кадра от камеры")
        if boxes:
            for d in list(w.last_dets):
                x1, y1, x2, y2 = map(int, d.box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 120), 2)
                cv2.putText(frame, f"{d.cls} {d.conf:.2f} #{d.track_id}", (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 120), 1)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(jpg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/events", dependencies=A)
    def events(limit: int = 200):
        return engine.journal.recent(limit)

    @app.get("/api/notify", dependencies=A)
    def get_notify():
        n = cfg.data["notify"]
        e = dict(n["email"], password="***" if n["email"].get("password") else "")
        t = dict(n["telegram"], token="***" if n["telegram"].get("token") else "")
        return {"email": e, "telegram": t, "site": cfg.data["site"]}

    @app.put("/api/notify", dependencies=A)
    def put_notify(body: dict):
        n = cfg.data["notify"]
        for ch in ("email", "telegram"):
            for k, v in (body.get(ch) or {}).items():
                if v == "***":
                    continue
                if k == "to" and isinstance(v, str):
                    v = [x.strip() for x in v.split(",") if x.strip()]
                n[ch][k] = v
        if body.get("site"):
            cfg.data["site"] = body["site"]
        cfg.save()
        return {"ok": True}

    @app.post("/api/notify/test", dependencies=A)
    def test_notify():
        import time
        engine.notifier.send(time.time(), "панель", "проверка", f"KADR · {cfg.data['site']}: проверка уведомлений")
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
