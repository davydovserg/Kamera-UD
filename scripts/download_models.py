"""Скачивает YOLOX-S (Apache-2.0, официальный релиз Megvii) в models/."""
import os
import urllib.request

URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
dst = os.path.join(os.path.dirname(__file__), "..", "models", "yolox_s.onnx")
os.makedirs(os.path.dirname(dst), exist_ok=True)
if os.path.exists(dst):
    print("уже есть:", os.path.abspath(dst))
else:
    print("качаю", URL)
    urllib.request.urlretrieve(URL, dst)
    print("готово:", os.path.abspath(dst))
