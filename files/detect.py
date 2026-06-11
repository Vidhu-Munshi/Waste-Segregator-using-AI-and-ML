from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import os
import uuid
import aiofiles

from services.classifier import classify
from services.detector import detect_objects
from database.db import save_detection

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    data = await file.read()
    image = Image.open(io.BytesIO(data))

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(UPLOAD_DIR, filename)
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)

    clf = classify(image)
    bboxes = detect_objects(image)

    result = {
        "filename": filename,
        "class": clf["class"],
        "confidence": clf["confidence"],
        "recyclable": clf["recyclable"],
        "hazard_level": clf["hazard_level"],
        "all_scores": clf["all_scores"],
        "bboxes": bboxes,
    }

    await save_detection(result)
    return result
