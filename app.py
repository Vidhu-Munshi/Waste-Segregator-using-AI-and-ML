"""
═══════════════════════════════════════════════════════════════
WasteVision AI - Flask Backend
═══════════════════════════════════════════════════════════════
Main application server handling:
- Image uploads
- Webcam capture predictions
- TensorFlow inference
- Gemini AI explanations
- SQLite history storage
═══════════════════════════════════════════════════════════════
"""

import os
import io
import base64
import uuid
import json
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, abort
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv

# Local utilities
from utils.model_loader import ModelManager
from utils.preprocessing import preprocess_image, decode_base64_image
from utils.gemini_helper import GeminiHelper
from utils.database import Database

# ───────────────────────────────────────────────────────────
# Load environment variables from .env
# ───────────────────────────────────────────────────────────
load_dotenv()

# ───────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

# Ensure required folders exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "static" / "images").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "uploads").mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────────
# Flask app initialization
# ───────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "wastevision-secret-2026")
CORS(app)

# ───────────────────────────────────────────────────────────
# Initialize core services
# ───────────────────────────────────────────────────────────
print("=" * 60)
print("  WasteVision AI Starting...")
print("=" * 60)

# Load TensorFlow model (singleton, only once)
model_manager = ModelManager(
    model_path=str(BASE_DIR / "model.h5"),
    class_names_path=str(BASE_DIR / "class_names.json"),
)

# Gemini AI helper
gemini = GeminiHelper(api_key=os.getenv("GEMINI_API_KEY", ""))

# SQLite database
db = Database(db_path=str(BASE_DIR / "database.db"))
db.init_db()

print(f"[OK] Model loaded:  {model_manager.is_loaded}")
print(f"[OK] Classes:       {len(model_manager.class_names) if model_manager.class_names else 0}")
print(f"[OK] Gemini ready:  {gemini.is_configured}")
print(f"[OK] Database:      {BASE_DIR / 'database.db'}")
print("=" * 60)


# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_pil_image(pil_img: Image.Image, prefix: str = "img") -> tuple[str, str]:
    """Save a PIL image to /static/uploads and return (filename, public_url)."""
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.jpg"
    filepath = UPLOAD_FOLDER / filename
    pil_img.convert("RGB").save(filepath, "JPEG", quality=88)
    return filename, f"/static/uploads/{filename}"


def build_disposal_guidance(class_name: str) -> dict:
    """Quick rule-based disposal guidance (used as fallback/supplement)."""
    cl = class_name.lower()

    # Recyclables
    if any(k in cl for k in ["plastic", "glass", "metal", "paper", "cardboard", "recycl"]):
        return {
            "hazard": "low",
            "recyclable": True,
            "icon": "♻️",
            "bin": "Blue / Recycling Bin",
            "tips": "Rinse and clean before recycling. Separate by material if possible.",
        }
    # Organic
    if any(k in cl for k in ["organic", "food", "compost"]):
        return {
            "hazard": "low",
            "recyclable": True,
            "icon": "🍎",
            "bin": "Green / Compost Bin",
            "tips": "Compost at home or use municipal organic waste collection.",
        }
    # E-waste
    if any(k in cl for k in [
        "battery", "keyboard", "microwave", "mobile", "mouse",
        "pcb", "player", "printer", "television", "washing", "ewaste", "e-waste",
    ]):
        return {
            "hazard": "high",
            "recyclable": True,
            "icon": "💻",
            "bin": "Certified E-Waste Drop-off",
            "tips": "Never dispose in regular bins. Contact certified e-waste recyclers.",
        }
    # Hazardous
    if any(k in cl for k in ["hazard", "chemical", "medical", "toxic"]):
        return {
            "hazard": "high",
            "recyclable": False,
            "icon": "⚠️",
            "bin": "Hazardous Waste Facility",
            "tips": "Requires specialized disposal. Contact local HazMat services.",
        }
    # Trash / non-recyclable
    if any(k in cl for k in ["trash", "non-recycl", "non_recycl"]):
        return {
            "hazard": "medium",
            "recyclable": False,
            "icon": "🗑️",
            "bin": "General Waste Bin",
            "tips": "Goes to landfill. Consider if any parts can be salvaged.",
        }
    # Default
    return {
        "hazard": "low",
        "recyclable": True,
        "icon": "🗑️",
        "bin": "Check local guidelines",
        "tips": "Verify recyclability with local waste authority.",
    }


# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "model_loaded": model_manager.is_loaded,
        "gemini_configured": gemini.is_configured,
        "classes": model_manager.class_names or [],
        "version": "2.4.0",
    })


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return list of class names the model can predict."""
    return jsonify({"classes": model_manager.class_names or []})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict waste category from uploaded image OR base64 image.
    Accepts:
      - multipart/form-data with key 'image' (file upload)
      - JSON {"image_base64": "data:image/...;base64,..."}  (webcam capture)
    Returns prediction + Gemini explanation + disposal guidance.
    """
    try:
        pil_image = None
        public_url = None
        filename = None

        # ─── (A) File upload ────────────────────────────────
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "Empty filename"}), 400
            if not allowed_file(file.filename):
                return jsonify({"error": "File type not allowed"}), 400

            safe_name = secure_filename(file.filename)
            unique = f"{uuid.uuid4().hex[:10]}_{safe_name}"
            save_path = UPLOAD_FOLDER / unique
            file.save(save_path)

            pil_image = Image.open(save_path).convert("RGB")
            filename = unique
            public_url = f"/static/uploads/{unique}"

        # ─── (B) Base64 (webcam capture) ────────────────────
        else:
            data = request.get_json(silent=True) or {}
            b64 = data.get("image_base64", "")
            if not b64:
                return jsonify({"error": "No image provided"}), 400
            pil_image = decode_base64_image(b64)
            filename, public_url = save_pil_image(pil_image, prefix="cam")

        # ─── Run TensorFlow prediction ──────────────────────
        if not model_manager.is_loaded:
            # Use a fallback simulated prediction so UI still demos
            prediction = model_manager.fallback_predict()
        else:
            input_tensor = preprocess_image(pil_image, target_size=model_manager.input_size)
            prediction = model_manager.predict(input_tensor)

        class_name = prediction["class_name"]
        confidence = prediction["confidence"]
        all_probs = prediction["all_probabilities"]

        # ─── Build disposal guidance ────────────────────────
        guidance = build_disposal_guidance(class_name)

        # ─── Gemini AI explanation ──────────────────────────
        gemini_response = gemini.explain_waste(
            class_name=class_name,
            confidence=confidence,
        )

        # ─── Save to database history ───────────────────────
        record_id = db.insert_detection({
            "class_name": class_name,
            "confidence": confidence,
            "hazard": guidance["hazard"],
            "recyclable": guidance["recyclable"],
            "image_path": public_url or "",
            "gemini_explanation": gemini_response.get("explanation", ""),
            "disposal_method": gemini_response.get("disposal", guidance["tips"]),
            "timestamp": datetime.now().isoformat(),
        })

        # ─── Final response ─────────────────────────────────
        return jsonify({
            "success": True,
            "id": record_id,
            "prediction": {
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "all_probabilities": all_probs,
            },
            "guidance": guidance,
            "gemini": gemini_response,
            "image_url": public_url,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat with Gemini AI about a waste item or general waste topics.
    Body: { "message": "...", "context": { "class_name": "...", ... }, "history": [...] }
    """
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        context = data.get("context") or {}
        history = data.get("history") or []

        if not message:
            return jsonify({"error": "Empty message"}), 400

        reply = gemini.chat(
            user_message=message,
            context=context,
            history=history,
        )
        return jsonify({"success": True, "reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history():
    """Retrieve detection history."""
    limit = request.args.get("limit", 100, type=int)
    rows = db.get_all_detections(limit=limit)
    return jsonify({"history": rows, "count": len(rows)})


@app.route("/api/history/<int:record_id>", methods=["GET"])
def history_detail(record_id):
    """Retrieve a specific detection record."""
    row = db.get_detection(record_id)
    if not row:
        return jsonify({"error": "Not found"}), 404
    return jsonify(row)


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id):
    """Delete a specific detection record."""
    db.delete_detection(record_id)
    return jsonify({"success": True})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear all detection history."""
    db.clear_all()
    return jsonify({"success": True})


@app.route("/api/stats", methods=["GET"])
def stats():
    """Dashboard statistics."""
    return jsonify(db.get_stats())


@app.route("/static/uploads/<path:filename>")
def serve_upload(filename):
    """Serve uploaded files."""
    return send_from_directory(str(UPLOAD_FOLDER), filename)


# ───────────────────────────────────────────────────────────
# Error handlers
# ───────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (max 10MB)"}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ───────────────────────────────────────────────────────────
# Run
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    print(f"\n>> Server starting on http://localhost:5000  (debug={debug_mode})\n")
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=False)
