import aiosqlite
import json
from datetime import datetime

DB_PATH = "database/wastevision.db"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    waste_class TEXT,
    confidence REAL,
    recyclable INTEGER,
    hazard_level TEXT,
    bboxes TEXT,
    ocr_text TEXT,
    timestamp TEXT
)
"""

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_TABLE)
        await db.commit()

async def save_detection(data: dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO detections
               (filename, waste_class, confidence, recyclable, hazard_level, bboxes, ocr_text, timestamp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                data.get("filename", ""),
                data.get("class", ""),
                data.get("confidence", 0.0),
                1 if data.get("recyclable") else 0,
                data.get("hazard_level", "low"),
                json.dumps(data.get("bboxes", [])),
                data.get("ocr_text", ""),
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()

async def get_history(limit: int = 50):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
