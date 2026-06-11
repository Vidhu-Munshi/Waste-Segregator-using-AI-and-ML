import numpy as np
import os
from PIL import Image

CLASSES = [
    "plastic", "metal", "glass", "organic", "paper",
    "cardboard", "battery", "e-waste", "medical_waste", "hazardous",
]

RECYCLABLE = {"plastic", "metal", "glass", "paper", "cardboard"}

HAZARD_MAP = {
    "battery": "high",
    "e-waste": "high",
    "medical_waste": "high",
    "hazardous": "critical",
    "organic": "low",
    "paper": "low",
    "cardboard": "low",
    "plastic": "low",
    "metal": "low",
    "glass": "medium",
}

MODEL_PATH = "models/classifier.h5"
_model = None


def _build_model():
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(base.input, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_classifier():
    global _model
    import tensorflow as tf
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        _model = tf.keras.models.load_model(MODEL_PATH)
    else:
        _model = _build_model()
    return _model


def preprocess(image: Image.Image) -> np.ndarray:
    import tensorflow as tf
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    return np.expand_dims(arr, 0)


def classify(image: Image.Image) -> dict:
    model = load_classifier()
    arr = preprocess(image)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    waste_class = CLASSES[idx]
    return {
        "class": waste_class,
        "confidence": round(confidence, 4),
        "recyclable": waste_class in RECYCLABLE,
        "hazard_level": HAZARD_MAP.get(waste_class, "low"),
        "all_scores": {c: round(float(p), 4) for c, p in zip(CLASSES, preds)},
    }
