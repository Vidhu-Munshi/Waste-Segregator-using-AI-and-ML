"""
Train the waste classifier with transfer learning.

Usage:
    python training/train_classifier.py --data datasets/waste --epochs 20

datasets/waste/ must follow ImageFolder structure:
    datasets/waste/
        plastic/  *.jpg ...
        metal/    *.jpg ...
        ...
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASSES = [
    "plastic", "metal", "glass", "organic", "paper",
    "cardboard", "battery", "e-waste", "medical_waste", "hazardous",
]
IMG_SIZE = (224, 224)
BATCH = 32
MODEL_OUT = "models/classifier.h5"


def build_model(num_classes: int):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(base.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def main(data_dir: str, epochs: int, fine_tune_epochs: int):
    os.makedirs("models", exist_ok=True)

    train_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
    )

    train = train_gen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH, subset="training"
    )
    val = train_gen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH, subset="validation"
    )

    num_classes = len(train.class_indices)
    model, base = build_model(num_classes)

    print(f"Training head — {epochs} epochs")
    model.fit(train, validation_data=val, epochs=epochs)

    # Fine-tune last 30 layers
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Fine-tuning — {fine_tune_epochs} epochs")
    model.fit(
        train,
        validation_data=val,
        epochs=fine_tune_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        ],
    )

    model.save(MODEL_OUT)
    print(f"Saved → {MODEL_OUT}")
    print("Class mapping:", train.class_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/waste")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.data, args.epochs, args.fine_tune_epochs)
