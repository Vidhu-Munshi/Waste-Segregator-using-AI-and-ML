"""
═══════════════════════════════════════════════════════════════
Image Preprocessing Utilities
═══════════════════════════════════════════════════════════════
Handles image loading, decoding, resizing, and normalization
for TensorFlow model inference.
═══════════════════════════════════════════════════════════════
"""

import io
import base64
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


def preprocess_image(pil_image: Image.Image, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Convert a PIL Image to a normalized TensorFlow-ready batch tensor.

    Args:
        pil_image: input PIL.Image
        target_size: (width, height) the model expects

    Returns:
        np.ndarray with shape (1, H, W, 3), dtype float32, values in [0, 1]
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize using high-quality bilinear filter
    img = pil_image.resize(target_size, Image.BILINEAR)

    # Convert to numpy array
    arr = np.asarray(img, dtype=np.float32)

    # Normalize using MobileNetV2's preprocess_input → scales pixels to [-1, 1]
    # This MUST match what was used during training (train_model.py uses the same)
    arr = mobilenet_preprocess(arr)

    # Add batch dimension -> (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr


def decode_base64_image(b64_string: str) -> Image.Image:
    """
    Decode a base64-encoded image (with or without data URL prefix)
    into a PIL.Image.

    Args:
        b64_string: e.g. "data:image/jpeg;base64,xxxx..." or pure base64

    Returns:
        PIL.Image in RGB mode
    """
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def pil_to_base64(pil_image: Image.Image, fmt: str = "JPEG") -> str:
    """Convert a PIL.Image to a base64 data URL string."""
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format=fmt, quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"
