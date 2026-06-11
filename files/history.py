from fastapi import APIRouter, Query
from database.db import get_history

router = APIRouter()


@router.get("/history")
async def history(limit: int = Query(50, ge=1, le=500)):
    rows = await get_history(limit)
    return {"count": len(rows), "detections": rows}
