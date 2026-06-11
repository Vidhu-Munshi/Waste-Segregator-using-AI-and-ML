from fastapi import APIRouter
from fastapi.responses import Response
from database.db import get_history
from services.report_generator import generate_pdf

router = APIRouter()


@router.get("/report/pdf")
async def pdf_report(limit: int = 100):
    detections = await get_history(limit)
    pdf_bytes = generate_pdf(detections)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=wastevision-report.pdf"},
    )
