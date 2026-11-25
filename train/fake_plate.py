"""Заменяет настоящий номер на демо-кадре вымышленным (регион 00 не выдаётся) и закрывает рамку дилера.

python train/fake_plate.py <исходный plate.jpg> <куда сохранить>
"""
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, dst = sys.argv[1], sys.argv[2]
im = cv2.imread(src)
im = cv2.resize(im, (1280, int(im.shape[0] * 1280 / im.shape[1])))
OX, OY = 280, 300  # смещение области номера в кадре 1280 px

pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
d = ImageDraw.Draw(pil)
WHITE, INK = (232, 234, 233), (28, 28, 30)
d.rounded_rectangle([OX + 32, OY + 46, OX + 556, OY + 188], 6, fill=WHITE)      # основное поле
d.rectangle([OX + 572, OY + 48, OX + 786, OY + 141], fill=WHITE)               # цифры региона (RUS и флаг остаются)
d.rectangle([OX + 58, OY + 196, OX + 772, OY + 234], fill=(34, 35, 38))        # рамка дилера — без названия и телефона

font = "C:/Windows/Fonts/bahnschrift.ttf"
big = ImageFont.truetype(font, 136)
small = ImageFont.truetype(font, 108)
try:
    big.set_variation_by_name("SemiBold")
    small.set_variation_by_name("SemiBold")
except Exception:
    pass

base = OY + 168
chars = [(ch, big if ch.isdigit() else small) for ch in "Е123КХ"]
widths = [d.textbbox((0, 0), ch, font=f)[2] - d.textbbox((0, 0), ch, font=f)[0] for ch, f in chars]
gap = (470 - sum(widths)) / (len(chars) - 1)
x = OX + 62
for (ch, f), wch in zip(chars, widths):
    bbox = d.textbbox((0, 0), ch, font=f)
    d.text((x - bbox[0], base - bbox[3]), ch, font=f, fill=INK)
    x += wch + gap
x = OX + 600
for ch in "00":
    bbox = d.textbbox((0, 0), ch, font=small)
    d.text((x - bbox[0], OY + 130 - bbox[3]), ch, font=small, fill=INK)
    x += (bbox[2] - bbox[0]) + 18

out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
out = cv2.GaussianBlur(out, (3, 3), 0.6)  # чтобы вписалось в фото
cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
