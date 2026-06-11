"""
═══════════════════════════════════════════════════════════════
WasteVision AI - Command-Line Prediction Script
═══════════════════════════════════════════════════════════════
Predict a single image without running the Flask server.

Usage:
    python predict.py path/to/image.jpg
═══════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
from PIL import Image

from utlis.model_loader import ModelManager
from utlis.preprocessing import preprocess_image


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    BASE_DIR = Path(__file__).resolve().parent
    mm = ModelManager(
        model_path=str(BASE_DIR / "model.h5"),
        class_names_path=str(BASE_DIR / "class_names.json"),
    )

    print(f"\n🔍 Analyzing: {image_path}")
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess_image(img, target_size=mm.input_size)
    result = mm.predict(tensor)

    print("\n═══ Prediction Result ═══")
    print(f"  Class:      {result['class_name']}")
    print(f"  Confidence: {result['confidence']}%")
    print("\n  Top probabilities:")
    for p in result["all_probabilities"]:
        bar = "█" * int(p["prob"] / 2)
        print(f"    {p['name']:<25} {p['prob']:>6.2f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
