"""
═══════════════════════════════════════════════════════════════
SQLite Database Helper
═══════════════════════════════════════════════════════════════
Stores detection history, predictions, and timestamps.
Thread-safe (each call opens its own connection).
═══════════════════════════════════════════════════════════════
"""

import sqlite3
from datetime import datetime
from typing import Optional


class Database:
    """Lightweight SQLite wrapper for detection records."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Create tables if they don't exist."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                hazard TEXT,
                recyclable INTEGER,
                image_path TEXT,
                gemini_explanation TEXT,
                disposal_method TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(timestamp DESC)")
        conn.commit()
        conn.close()

    def insert_detection(self, record: dict) -> int:
        """Insert a detection record. Returns inserted row id."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO detections
              (class_name, confidence, hazard, recyclable, image_path,
               gemini_explanation, disposal_method, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("class_name", ""),
            float(record.get("confidence", 0.0)),
            record.get("hazard", "low"),
            1 if record.get("recyclable") else 0,
            record.get("image_path", ""),
            record.get("gemini_explanation", ""),
            record.get("disposal_method", ""),
            record.get("timestamp", datetime.now().isoformat()),
        ))
        conn.commit()
        last_id = cur.lastrowid
        conn.close()
        return last_id

    def get_all_detections(self, limit: int = 100) -> list[dict]:
        """Retrieve recent detections."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_detection(self, record_id: int) -> Optional[dict]:
        """Retrieve a single detection."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM detections WHERE id = ?", (record_id,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None

    def delete_detection(self, record_id: int):
        """Delete a record by id."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM detections WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()

    def clear_all(self):
        """Delete all records."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM detections")
        conn.commit()
        conn.close()

    def get_stats(self) -> dict:
        """Aggregate statistics for the dashboard."""
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) AS c FROM detections")
        total = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM detections WHERE hazard = 'high'")
        hazardous = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM detections WHERE recyclable = 1")
        recyclable = cur.fetchone()["c"]

        cur.execute("""
            SELECT class_name, COUNT(*) AS c
            FROM detections
            GROUP BY class_name
            ORDER BY c DESC
            LIMIT 10
        """)
        by_class = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT AVG(confidence) AS avg_c FROM detections")
        avg_conf = cur.fetchone()["avg_c"] or 0.0

        conn.close()

        recyc_pct = round((recyclable / total * 100) if total else 0, 1)
        co2_saved = round(recyclable * 0.03, 2)  # rough estimate (tons)

        return {
            "total": total,
            "hazardous": hazardous,
            "recyclable": recyclable,
            "recyclable_pct": recyc_pct,
            "avg_confidence": round(avg_conf, 2),
            "co2_saved_tons": co2_saved,
            "by_class": by_class,
        }
