"""D-Fire (YOLO-разметка: 0=smoke, 1=fire) → COCO JSON для YOLOX. Наши классы: 0=fire, 1=smoke.

python dfire_to_coco.py <папка dfire_yolo>
Создаёт <папка>/annotations/{train,val,test}.json
"""
import json
import os
import sys

from PIL import Image

root = sys.argv[1]
REMAP = {1: 0, 0: 1}  # fire первым — так ждёт сценарий «Огонь»
os.makedirs(os.path.join(root, "annotations"), exist_ok=True)

for split in ("train", "val", "test"):
    img_dir, lbl_dir = os.path.join(root, split, "images"), os.path.join(root, split, "labels")
    images, anns, aid = [], [], 1
    for i, name in enumerate(sorted(os.listdir(img_dir))):
        with Image.open(os.path.join(img_dir, name)) as im:
            w, h = im.size
        images.append({"id": i + 1, "file_name": name, "width": w, "height": h})
        lp = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if not os.path.exists(lp):
            continue
        for line in open(lp):
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            x, y, ww, hh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            if ww < 1 or hh < 1:
                continue
            anns.append({"id": aid, "image_id": i + 1, "category_id": REMAP[c], "bbox": [x, y, ww, hh],
                         "area": ww * hh, "iscrowd": 0})
            aid += 1
    json.dump({"images": images, "annotations": anns,
               "categories": [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}]},
              open(os.path.join(root, "annotations", f"{split}.json"), "w"))
    print(split, len(images), "кадров,", len(anns), "рамок")
