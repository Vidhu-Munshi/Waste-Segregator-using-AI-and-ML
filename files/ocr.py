from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from services.ocr_engine import extract_text

router = APIRouter()


@router.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    data = await file.read()
    image = Image.open(io.BytesIO(data))
    return extract_text(image)
