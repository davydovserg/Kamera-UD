"""Демо «Вдоль забора»: прогоняет детектор, трекер и сценарий FenceWalk по видео
и сохраняет кадры вокруг первой тревоги вместе с рамками и путём человека.

python train/make_fence_demo.py <видео> <папка demo>
"""
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kamera.detector import YoloxDetector  # noqa: E402
from kamera.scenarios import FenceWalk, FrameCtx  # noqa: E402
from kamera.tracker import IouTracker  # noqa: E402

video, out_dir = sys.argv[1], sys.argv[2]
ZONE = [[0, 0.30], [0.45, 0.335], [0.78, 0.25], [0.80, 0.37], [0.45, 0.45], [0, 0.41]]
STEP = 10  # видео 10 к/с → анализ раз в секунду, как в продукте
root = os.path.join(os.path.dirname(__file__), "..")
det = YoloxDetector(os.path.join(root, "models", "yolox_s.onnx"), conf=0.35)
cam = {"id": "fence", "name": "Камера 12 · Забор, южная сторона"}
sc = FenceWalk(cam, {"enabled": True, "zone": ZONE, "min_sec": 15, "min_path": 18, "cooldown_sec": 5})
tr = IouTracker(max_lost_sec=3)
now = datetime.now(ZoneInfo("Europe/Moscow"))
cap = cv2.VideoCapture(video)
frames, i, t0, alarm_at = [], 0, time.time(), None
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i % STEP == 0:
        ts = t0 + i / 10
        dets = det.detect(f, only={"person"})
        tr.update(dets, ts)
        ev = sc.process(FrameCtx(cam, now, ts, f, dets, tr.tracks))
        h, w = f.shape[:2]
        frames.append({"img": f, "ev": ev, "objects": [
            {"id": d.track_id, "box": [round(d.box[0] / w, 4), round(d.box[1] / h, 4),
                                       round(d.box[2] / w, 4), round(d.box[3] / h, 4)]} for d in dets],
            "path": {str(t.id): [[round(x / w, 4), round(y / h, 4)] for x, y in t.path[-12:]]
                     for t in tr.tracks.values() if t.last_seen == ts}})
        if ev and alarm_at is None:
            alarm_at = len(frames) - 1
            alarm_id = next(d.track_id for d in dets if sc._alerted.get(d.track_id) == ts)
            print("тревога на кадре", alarm_at, ev, "объект", alarm_id)
    i += 1
if alarm_at is None:
    sys.exit("тревоги не было — поправьте зону")

seq = frames[max(0, alarm_at - 7): alarm_at + 2]
os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
meta = {"zone": ZONE, "alarm_id": alarm_id, "alarm_index": alarm_at - max(0, alarm_at - 7), "frames": []}
for k, fr in enumerate(seq):
    name = f"fence_{k}.jpg"
    cv2.imwrite(os.path.join(out_dir, "img", name), fr["img"], [cv2.IMWRITE_JPEG_QUALITY, 80])
    meta["frames"].append({"img": name, "objects": fr["objects"], "path": fr["path"], "ev": fr["ev"]})
h, w = seq[0]["img"].shape[:2]
meta.update(w=w, h=h)
json.dump(meta, open(os.path.join(out_dir, "fence.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("сохранено кадров:", len(seq))
