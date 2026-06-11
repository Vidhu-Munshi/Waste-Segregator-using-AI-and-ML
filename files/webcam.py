from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
import base64
import io
import json

from services.classifier import classify
from services.detector import detect_objects
from database.db import save_detection

router = APIRouter()


@router.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)

            # Expect: {"image": "<base64 jpeg/png>"}
            img_b64 = payload.get("image", "")
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(img_bytes))

            clf = classify(image)
            bboxes = detect_objects(image)

            result = {
                "class": clf["class"],
                "confidence": clf["confidence"],
                "recyclable": clf["recyclable"],
                "hazard_level": clf["hazard_level"],
                "bboxes": bboxes,
            }

            await save_detection({**result, "filename": "webcam"})
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()
