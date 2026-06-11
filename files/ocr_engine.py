import numpy as np
from PIL import Image

_reader = None


def load_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text(image: Image.Image) -> dict:
    reader = load_reader()
    arr = np.array(image.convert("RGB"))
    results = reader.readtext(arr)
    texts = [{"text": t, "confidence": round(c, 4)} for (_, t, c) in results]
    combined = " ".join(r["text"] for r in texts)
    return {"text": combined, "details": texts}
