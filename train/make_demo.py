"""Готовит кадры для демо-страницы: прогоняет реальные модели Kamera-UD и рисует результат.

python train/make_demo.py <папка с исходными кадрами> <папка demo/img>
Пишет demo/data.json с найденными объектами — страница показывает ровно то, что увидела модель.
"""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kamera.detector import YoloxDetector  # noqa: E402
from kamera.scenarios import VEHICLES, norm_plate, LAT2CYR  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
root = os.path.join(os.path.dirname(__file__), "..")
coco = YoloxDetector(os.path.join(root, "models", "yolox_s.onnx"), conf=0.35)
fire_path = os.path.join(root, "models", "fire.onnx")
fire = YoloxDetector(fire_path, classes=["fire", "smoke"], conf=0.35) if os.path.exists(fire_path) else None
out = {}


def save(name, img):
    cv2.imwrite(os.path.join(dst, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])


for f in sorted(os.listdir(src)):
    if not f.lower().endswith((".jpg", ".png")):
        continue
    key = os.path.splitext(f)[0]
    img = cv2.imread(os.path.join(src, f))
    if img.shape[1] > 1280:
        img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])))
    h, w = img.shape[:2]
    rec = {"w": w, "h": h, "objects": []}
    if key.startswith("fire") and not fire:
        # базовый режим: цветовая маска пламени (на одном кадре без проверки мерцания)
        import numpy as np
        b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = ((r > 180) & (r >= g) & (g > b) & ((r - b) > 70) & (hsv[..., 1] > 70) & (hsv[..., 2] > 170)).astype(np.uint8)
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.ones((25, 25), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        from kamera.detector import Detection
        dets = [Detection("fire", 0.0, (x, y, x + bw, y + bh)) for x, y, bw, bh, a in stats[1:]
                if a > 0.002 * w * h]
    elif key.startswith("fire"):
        dets = fire.detect(img)
    else:
        dets = coco.detect(img, only=set(VEHICLES) | {"person"})
    for d in dets:
        x1, y1, x2, y2 = d.box
        rec["objects"].append({"cls": d.cls, "conf": round(d.conf, 2),
                               "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]})
    if key.startswith("plate"):
        from fast_alpr import ALPR
        alpr = ALPR(ocr_model="global-plates-mobile-vit-v2-model", ocr_device="cpu")
        rec["plates"] = []
        for r in alpr.predict(img):
            bb = r.detection.bounding_box
            text = norm_plate(r.ocr.text if r.ocr else "").translate(LAT2CYR)
            rec["plates"].append({"text": text, "box": [bb.x1 / w, bb.y1 / h, bb.x2 / w, bb.y2 / h]})
    save(key, img)  # чистый кадр: рамки и зоны рисует страница
    out[key] = rec
    print(key, len(rec["objects"]), "объектов", rec.get("plates", ""))

json.dump(out, open(os.path.join(dst, "..", "data.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
