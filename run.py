"""Запуск: python run.py [config.yaml]"""
import logging
import os
import shutil
import sys

import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kamera.engine import Engine  # noqa: E402
from kamera.web import create_app  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "config.yaml"))
    if not os.path.exists(path):
        shutil.copy(os.path.join(HERE, "config.example.yaml"), path)
        logging.info("Создан %s из примера — поменяйте пароль панели", path)
    os.makedirs(os.path.join(os.path.dirname(path), "data"), exist_ok=True)
    engine = Engine(path).start()
    p = engine.cfg.data["panel"]
    uvicorn.run(create_app(engine), host=p["host"], port=int(p["port"]), log_level="warning")


if __name__ == "__main__":
    main()
