"""Уведомления (только текст: время + камера + сценарий) и журнал отправок (по умолчанию 30 дней)."""
from __future__ import annotations

import json
import logging
import queue
import smtplib
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from email.mime.text import MIMEText

log = logging.getLogger("kamera.notify")


class Journal:
    def __init__(self, path: str, keep_days: int = 30):
        self.path, self.keep_days = path, keep_days
        self._lock = threading.Lock()
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY, ts REAL, camera TEXT, scenario TEXT, text TEXT, sent TEXT)""")

    def _db(self):
        return sqlite3.connect(self.path, timeout=10)

    def add(self, ts, camera, scenario, text, sent) -> int:
        with self._lock, self._db() as db:
            db.execute("DELETE FROM events WHERE ts < ?", (time.time() - self.keep_days * 86400,))
            cur = db.execute("INSERT INTO events(ts,camera,scenario,text,sent) VALUES(?,?,?,?,?)",
                             (ts, camera, scenario, text, sent))
            return cur.lastrowid

    def set_sent(self, row_id: int, sent: str):
        with self._lock, self._db() as db:
            db.execute("UPDATE events SET sent=? WHERE id=?", (sent, row_id))

    def recent(self, limit=200, camera=None):
        q, args = "SELECT ts,camera,scenario,text,sent FROM events", []
        if camera:
            q, args = q + " WHERE camera=?", [camera]
        with self._db() as db:
            rows = db.execute(q + " ORDER BY ts DESC LIMIT ?", args + [limit]).fetchall()
        return [dict(zip(("ts", "camera", "scenario", "text", "sent"), r)) for r in rows]


class Notifier:
    def __init__(self, cfg: dict, journal: Journal):
        self.cfg = cfg  # ссылка на config.data["notify"] — правки из панели видны сразу
        self.journal = journal
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="notifier", daemon=True).start()

    def send(self, ts, camera, scenario, text):
        row = self.journal.add(ts, camera, scenario, text, "в очереди")
        self.q.put((row, text))

    def _run(self):
        while True:
            row, text = self.q.get()
            results = []
            for name, fn in (("почта", self._email), ("telegram", self._telegram)):
                c = self.cfg.get("email" if name == "почта" else "telegram", {})
                if not c.get("enabled"):
                    continue
                for attempt in range(3):
                    try:
                        fn(c, text)
                        results.append(f"{name}: ок")
                        break
                    except Exception as e:  # сеть/SMTP — пробуем ещё
                        if attempt == 2:
                            results.append(f"{name}: ошибка {e}")
                            log.warning("notify %s failed: %s", name, e)
                        time.sleep(3 * (attempt + 1))
            self.journal.set_sent(row, "; ".join(results) or "каналы не настроены")

    @staticmethod
    def _email(c, text):
        msg = MIMEText(text, "plain", "utf-8")
        msg["Subject"] = "KADR: " + text.split("\n")[0][:120]
        msg["From"] = c.get("from") or c["user"]
        to = c["to"] if isinstance(c["to"], list) else [c["to"]]
        msg["To"] = ", ".join(to)
        cls = smtplib.SMTP_SSL if c.get("ssl", True) else smtplib.SMTP
        with cls(c["smtp_host"], int(c["smtp_port"]), timeout=20) as s:
            if not c.get("ssl", True):
                s.starttls()
            if c.get("user"):
                s.login(c["user"], c["password"])
            s.sendmail(msg["From"], to, msg.as_string())

    @staticmethod
    def _telegram(c, text):
        data = urllib.parse.urlencode({"chat_id": c["chat_id"], "text": text}).encode()
        url = f"https://api.telegram.org/bot{c['token']}/sendMessage"
        with urllib.request.urlopen(url, data=data, timeout=20) as r:
            if not json.loads(r.read()).get("ok"):
                raise RuntimeError("telegram вернул ошибку")
