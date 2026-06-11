"""
═══════════════════════════════════════════════════════════════
WasteVision AI - Model Training Script  (v3 - Multi-Source)
═══════════════════════════════════════════════════════════════
Handles the nested Capstone dataset at E:\\Capstone_4thsem\\:

  dataset/           ← mixed flat + double-nested categories
  modified-dataset/  ← pre-split e-waste classes (merged back)

Workflow:
  1. Scan & collect all images → dict[class → [paths]]
  2. Flatten & split (70 / 15 / 15) → prepared_dataset/
  3. Train MobileNetV2 (head freeze → fine-tune)
  4. Evaluate: classification report + confusion matrix + curves

Usage:
  python train_model.py                        # full run
  python train_model.py --dry-run              # scan only, no copy/train
  python train_model.py --epochs 20 --batch 32
  python train_model.py --force-prepare        # re-copy even if folder exists
═══════════════════════════════════════════════════════════════
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no GUI required
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)

# sklearn for evaluation metrics
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠ scikit-learn not installed — skipping detailed metrics.")
    SKLEARN_AVAILABLE = False


# ───────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
CAPSTONE_DIR  = Path("E:/Capstone_4thsem")
DATASET_DIR   = CAPSTONE_DIR / "dataset"
MODIFIED_DIR  = CAPSTONE_DIR / "modified-dataset"
PREPARED_DIR  = BASE_DIR / "prepared_dataset"   # flattened + split output
MODEL_PATH    = BASE_DIR / "model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
CHECKPOINT_DIR   = BASE_DIR / "models"
METRICS_DIR      = BASE_DIR / "metrics"

CHECKPOINT_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ───────────────────────────────────────────────────────────
# Flat categories (images live directly inside the folder)
# ───────────────────────────────────────────────────────────
FLAT_CATEGORIES = [
    "cardboard", "glass", "metal", "paper", "plastic", "trash",
]

# ───────────────────────────────────────────────────────────
# Double-nested categories: dataset/<top>/<top>/<sub>/images
# Each sub-folder becomes its own class.
# ───────────────────────────────────────────────────────────
NESTED_ROOTS = [
    "hazardous",
    "non-recyclable",
    "organic",
    "recyclable",
]


# ═══════════════════════════════════════════════════════════
# Phase 1 — Data collection
# ═══════════════════════════════════════════════════════════

def _images_in(folder: Path) -> list:
    """Return all image paths under *folder* recursively."""
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def collect_all_images(dataset_dir: Path, modified_dir: Path) -> dict:
    """
    Walk both source roots and return:
        { class_name: [Path, ...], ... }

    Rules
    -----
    - Flat category   → class = folder name
    - Double-nested   → class = leaf sub-folder name
    - modified-dataset → all train/val/test splits merged by class name
      (so we can re-split uniformly; avoids data-leakage from the
       pre-existing split boundaries)
    """
    class_images = defaultdict(list)

    print("\n📂  Scanning dataset/ (flat categories)")
    print("─" * 55)
    for cat in FLAT_CATEGORIES:
        cat_dir = dataset_dir / cat
        if not cat_dir.exists():
            print(f"  ⚠  Missing: {cat_dir}")
            continue
        imgs = _images_in(cat_dir)
        class_images[cat].extend(imgs)
        print(f"  ✓  {cat:<28} {len(imgs):>5} images")

    print("\n📂  Scanning dataset/ (nested categories)")
    print("─" * 55)
    for root_name in NESTED_ROOTS:
        # Structure: dataset/<root>/<root>/<sub>/
        nested_dir = dataset_dir / root_name / root_name
        if not nested_dir.exists():
            # Try one level up in case the double-nesting doesn't apply
            nested_dir = dataset_dir / root_name
        if not nested_dir.exists():
            print(f"  ⚠  Missing nested root: {nested_dir}")
            continue
        sub_dirs = [d for d in nested_dir.iterdir() if d.is_dir()]
        if not sub_dirs:
            # All images are directly inside — treat as flat class
            imgs = _images_in(nested_dir)
            class_images[root_name].extend(imgs)
            print(f"  ✓  {root_name:<28} {len(imgs):>5} images  (flat fallback)")
        else:
            for sub in sorted(sub_dirs):
                imgs = _images_in(sub)
                class_images[sub.name].extend(imgs)
                print(f"  ✓  {sub.name:<28} {len(imgs):>5} images  [{root_name}]")

    print("\n📂  Scanning modified-dataset/ (merging all splits)")
    print("─" * 55)
    if not modified_dir.exists():
        print(f"  ⚠  modified-dataset not found at {modified_dir} — skipping.")
    else:
        merged_counts = defaultdict(int)
        for split in ("train", "val", "test"):
            split_dir = modified_dir / split
            if not split_dir.exists():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    imgs = _images_in(cls_dir)
                    class_images[cls_dir.name].extend(imgs)
                    merged_counts[cls_dir.name] += len(imgs)
        for cls, cnt in sorted(merged_counts.items()):
            print(f"  ✓  {cls:<28} {cnt:>5} images  [modified-dataset]")

    print()
    return dict(class_images)


# ═══════════════════════════════════════════════════════════
# Phase 2 — Dataset preparation (flatten + split)
# ═══════════════════════════════════════════════════════════

def prepare_dataset(
    class_images: dict,
    output_dir: Path,
    ratios: tuple = (0.70, 0.15, 0.15),
    seed: int = 42,
    force: bool = False,
) -> list:
    """
    Copy images into output_dir/train|val|test/<class>/.
    Returns the sorted list of class names.

    Skips the copy step if the folder already exists and *force* is False.
    """
    class_names = sorted(class_images.keys())

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        print(f"✓ Prepared dataset already exists at {output_dir}")
        print("  Use --force-prepare to re-copy.\n")
        return class_names

    print(f"\n▶ Preparing dataset → {output_dir}")
    print(f"  Split ratios  train={ratios[0]:.0%}  val={ratios[1]:.0%}  test={ratios[2]:.0%}")
    print("─" * 60)

    random.seed(seed)
    train_r, val_r, _ = ratios
    total_copied = 0

    for class_name in class_names:
        images = list(class_images[class_name])
        random.shuffle(images)
        n       = len(images)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        for split, imgs in splits.items():
            dest = output_dir / split / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                # Avoid name collisions by prefixing with parent folder name
                new_name = f"{img_path.parent.name}_{img_path.name}"
                dst = dest / new_name
                if not dst.exists():
                    shutil.copy2(img_path, dst)
            total_copied += len(imgs)

        print(
            f"  {class_name:<28} "
            f"train={len(splits['train']):<5} "
            f"val={len(splits['val']):<5} "
            f"test={len(splits['test'])}"
        )

    print(f"\n✓ Total images copied: {total_copied}\n")
    return class_names


# ═══════════════════════════════════════════════════════════
# Phase 3 — Model
# ═══════════════════════════════════════════════════════════

def build_model(num_classes: int, input_shape=(224, 224, 3)) -> tf.keras.Model:
    """MobileNetV2 transfer-learning classifier."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune(model: tf.keras.Model, learning_rate: float = 1e-5):
    """Unfreeze last 30 layers of the base for fine-tuning."""
    base = model.layers[0]
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


# ═══════════════════════════════════════════════════════════
# Phase 4 — Metrics & Reporting
# ═══════════════════════════════════════════════════════════

def plot_training_curves(history1, history2=None):
    """Save accuracy and loss curves to metrics/training_curves.png."""
    acc  = history1.history["accuracy"]
    val_acc  = history1.history["val_accuracy"]
    loss = history1.history["loss"]
    val_loss = history1.history["val_loss"]

    if history2:
        acc      += history2.history["accuracy"]
        val_acc  += history2.history["val_accuracy"]
        loss     += history2.history["loss"]
        val_loss += history2.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    phase_split = len(history1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("WasteVision AI — Training History", fontsize=14, fontweight="bold")

    # Accuracy
    ax1.plot(epochs, acc,     label="Train Accuracy",  color="#2196F3")
    ax1.plot(epochs, val_acc, label="Val Accuracy",    color="#FF9800", linestyle="--")
    if history2:
        ax1.axvline(phase_split, color="gray", linestyle=":", label="Fine-tune start")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(epochs, loss,     label="Train Loss",  color="#F44336")
    ax2.plot(epochs, val_loss, label="Val Loss",    color="#9C27B0", linestyle="--")
    if history2:
        ax2.axvline(phase_split, color="gray", linestyle=":", label="Fine-tune start")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = METRICS_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves saved → {out}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    """Save a colour-coded confusion matrix to metrics/confusion_matrix.png."""
    n = len(class_names)
    fig_size = max(10, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label",      fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = METRICS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix saved → {out}")


def evaluate_model(model, test_gen, class_names: list):
    """
    Run full evaluation on the test set:
      - Loss & accuracy
      - Per-class classification report (precision, recall, F1)
      - Confusion matrix image
    """
    print("\n▶ Evaluating on test set...")
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\n  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {acc * 100:.2f}%")

    if not SKLEARN_AVAILABLE:
        return

    print("\n▶ Generating detailed metrics...")

    # Collect predictions
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # Guard: align class_names with generator's ordering
    gen_class_names = [class_names[i] for i in range(len(class_names))]

    # Classification report
    report = classification_report(y_true, y_pred, target_names=gen_class_names, zero_division=0)
    print("\n" + "═" * 60)
    print("  Classification Report")
    print("═" * 60)
    print(report)

    report_path = METRICS_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Loss:     {loss:.4f}\n")
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"  ✓ Report saved → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, gen_class_names)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="WasteVision AI Trainer")
    parser.add_argument("--epochs",           type=int,   default=15,    help="Phase-1 training epochs")
    parser.add_argument("--fine-tune-epochs", type=int,   default=5,     help="Phase-2 fine-tune epochs")
    parser.add_argument("--batch",            type=int,   default=32,    help="Batch size")
    parser.add_argument("--img-size",         type=int,   default=224,   help="Image size (square)")
    parser.add_argument("--dry-run",          action="store_true",       help="Scan only, no copy or training")
    parser.add_argument("--force-prepare",    action="store_true",       help="Re-copy dataset even if prepared_dataset/ exists")
    parser.add_argument("--train-ratio",      type=float, default=0.70)
    parser.add_argument("--val-ratio",        type=float, default=0.15)
    args = parser.parse_args()

    test_ratio  = round(1.0 - args.train_ratio - args.val_ratio, 4)
    split_ratios = (args.train_ratio, args.val_ratio, test_ratio)
    img_size     = (args.img_size, args.img_size)

    print("=" * 60)
    print("🧠  WasteVision AI — Model Training")
    print("=" * 60)
    print(f"  Dataset dir:  {DATASET_DIR}")
    print(f"  Modified dir: {MODIFIED_DIR}")
    print(f"  Output dir:   {PREPARED_DIR}")
    print(f"  Image size:   {img_size}")
    print(f"  Batch size:   {args.batch}")
    print(f"  Epochs:       {args.epochs} + {args.fine_tune_epochs} fine-tune")
    print(f"  Split:        {split_ratios[0]:.0%} / {split_ratios[1]:.0%} / {split_ratios[2]:.0%}")
    print("=" * 60)

    # ── Phase 1: Collect images ──────────────────────────────
    class_images = collect_all_images(DATASET_DIR, MODIFIED_DIR)

    print("\n📊  Dataset summary")
    print("─" * 55)
    total_images = sum(len(v) for v in class_images.values())
    for cls, imgs in sorted(class_images.items()):
        print(f"  {cls:<28} {len(imgs):>5} images")
    print(f"  {'TOTAL':<28} {total_images:>5} images")
    print(f"  {'CLASSES':<28} {len(class_images):>5}")

    if args.dry_run:
        print("\n✅  Dry run complete — no files copied or training started.")
        return

    # ── Phase 2: Prepare (split + copy) ─────────────────────
    class_names = prepare_dataset(
        class_images,
        output_dir=PREPARED_DIR,
        ratios=split_ratios,
        force=args.force_prepare,
    )
    num_classes = len(class_names)

    # Save class names
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"✓ class_names.json saved ({num_classes} classes)")

    # ── Phase 3: Data generators ─────────────────────────────
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    eval_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_gen.flow_from_directory(
        PREPARED_DIR / "train",
        target_size=img_size,
        batch_size=args.batch,
        class_mode="categorical",
        shuffle=True,
    )
    val_data = eval_gen.flow_from_directory(
        PREPARED_DIR / "val",
        target_size=img_size,
        batch_size=args.batch,
        class_mode="categorical",
        shuffle=False,
    )
    test_data = eval_gen.flow_from_directory(
        PREPARED_DIR / "test",
        target_size=img_size,
        batch_size=args.batch,
        class_mode="categorical",
        shuffle=False,
    )

    # Use the generator's class order (alphabetical, matches our sorted list)
    class_names = list(train_data.class_indices.keys())
    num_classes  = train_data.num_classes
    print(f"\n✓ Classes detected by generator: {num_classes}")
    print(f"  {class_names}\n")

    # ── Phase 4: Build & train ───────────────────────────────
    model = build_model(num_classes, input_shape=(*img_size, 3))
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(
            str(CHECKPOINT_DIR / "best_model.h5"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", min_lr=1e-7),
    ]

    print("\n▶ Phase 1: Training classification head...\n")
    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history2 = None
    if args.fine_tune_epochs > 0:
        print("\n▶ Phase 2: Fine-tuning base layers...\n")
        fine_tune(model, learning_rate=1e-5)
        history2 = model.fit(
            train_data,
            validation_data=val_data,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )

    # ── Phase 5: Save model ──────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n✓ Model saved → {MODEL_PATH}")

    # ── Phase 6: Metrics ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("📈  Metrics & Evaluation")
    print("=" * 60)

    plot_training_curves(history1, history2)
    evaluate_model(model, test_data, class_names)

    print("\n" + "=" * 60)
    print("✅  Training complete!")
    print(f"   Model:    {MODEL_PATH}")
    print(f"   Metrics:  {METRICS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
