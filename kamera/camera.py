"""Чтение потока камеры в отдельном потоке: держим только последний кадр, сами переподключаемся."""
from __future__ import annotations

import os
import re
import threading
import time

import cv2

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

# Шаблоны ссылок популярных регистраторов/камер. {ch} — номер канала.
VENDORS = {
    "hikvision": "rtsp://{user}:{password}@{host}:{port}/Streaming/Channels/{ch}02",
    "hikvision-main": "rtsp://{user}:{password}@{host}:{port}/Streaming/Channels/{ch}01",
    "dahua": "rtsp://{user}:{password}@{host}:{port}/cam/realmonitor?channel={ch}&subtype=1",
    "dahua-main": "rtsp://{user}:{password}@{host}:{port}/cam/realmonitor?channel={ch}&subtype=0",
    "trassir": "rtsp://{user}:{password}@{host}:{port}/live/main",
    "uniview": "rtsp://{user}:{password}@{host}:{port}/unicast/c{ch}/s1/live",
}


def build_url(vendor: str, host: str, user: str, password: str, ch: int = 1, port: int = 554) -> str:
    return VENDORS[vendor].format(user=user, password=password, host=host, port=port, ch=ch)


def mask_url(url: str) -> str:
    return re.sub(r"(//[^:/@]+:)[^@]*@", r"\1***@", url or "")


def probe(url: str, timeout_sec: float = 10) -> tuple[bool, str]:
    """Проверка доступа: открыть поток и получить кадр."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    t0 = time.time()
    try:
        while time.time() - t0 < timeout_sec:
            ok, frame = cap.read()
            if ok and frame is not None:
                return True, f"{frame.shape[1]}×{frame.shape[0]}"
            if not cap.isOpened():
                break
        return False, "нет кадра: проверьте адрес, логин/пароль и что порт 554 открыт"
    finally:
        cap.release()


class CameraStream:
    def __init__(self, cam_id: str, url: str):
        self.cam_id = cam_id
        self.url = url
        self.is_file = os.path.isfile(url)
        self.frame = None
        self.frame_time = 0.0
        self.online = False
        self.error = ""
        self.fps_in = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"cam-{cam_id}", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def latest(self):
        with self._lock:
            return (None if self.frame is None else self.frame.copy()), self.frame_time

    def _run(self):
        backoff = 2
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.online, self.error = False, "не удалось открыть поток"
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60)
                continue
            backoff = 2
            file_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            n, t_count = 0, time.time()
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.is_file:  # тестовое видео — по кругу
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.online, self.error = False, "поток оборвался, переподключение"
                    break
                with self._lock:
                    self.frame, self.frame_time = frame, time.time()
                self.online, self.error = True, ""
                n += 1
                if time.time() - t_count >= 5:
                    self.fps_in, n, t_count = n / (time.time() - t_count), 0, time.time()
                if self.is_file:
                    time.sleep(1 / file_fps)
            cap.release()
