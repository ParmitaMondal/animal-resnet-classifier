#!/usr/bin/env python3
# Train a multiclass animal classifier with ResNet50 transfer learning.

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from utils import (
    set_seed,
    ensure_dir,
    get_datasets,
    compute_class_weights,
    plot_history_curves,
    evaluate_dataset_macro_f1,
)

def build_model(img_size: int, num_classes: int):
    """ResNet50 backbone + classification head."""
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Light augmentations
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")
    x = aug(inputs)

    # ResNet50 preprocessing
    x = layers.Lambda(preprocess_input, name="preprocess")(x)

    base = ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False  # Stage 1: freeze backbone

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="AnimalResNet50")
    return model, base

def parse_args():
    p = argparse.ArgumentParser(description="Animal Classification with ResNet50 (transfer learning)")
    p.add_argument("--data_dir", type=str, default="data", help="Root with train/ val/ (and optional test/)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_finetune", type=float, default=1e-5)
    p.add_argument("--fine_tune_at", type=int, default=143, help="Unfreeze from this layer index (ResNet50 has ~175)")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "plots"))

    print(f"[INFO] img_size={args.img_size}, batch_size={args.batch_size}, epochs={args.epochs}")

    # Datasets
    train_ds, val_ds, class_names = get_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
    num_classes = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")

    # Class weights (optional but helpful for imbalance)
    class_weights = compute_class_weights(train_ds, num_classes)
    print(f"[INFO] Class weights: {class_weights}")

    # Save class names mapping
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Model
    model, base = build_model(args.img_size, num_classes)

    metrics = [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")
    ]

    # Stage 1: train head only
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_head),
        loss="categorical_crossentropy",
        metrics=metrics,
    )

    ckpt_best = os.path.join(args.output_dir, "best_val_acc.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_best, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    print("[INFO] Stage 1: training classification head (backbone frozen)...")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Save history + plots
    with open(os.path.join(args.output_dir, "history_stage1.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist1.history.items()}, f)
    plot_history_curves(hist1.history, os.path.join(args.output_dir, "plots"), prefix="stage1")

    macro_f1_val = evaluate_dataset_macro_f1(model, val_ds, class_names)
    print(f"[INFO] Stage 1 validation macro-F1: {macro_f1_val:.4f}")

    # Stage 2: fine-tune top of backbone
    print(f"[INFO] Stage 2: fine-tuning from layer index {args.fine_tune_at} ...")
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= args.fine_tune_at)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_finetune),
        loss="categorical_crossentropy",
        metrics=metrics,
    )

    callbacks_ft = [
        keras.callbacks.ModelCheckpoint(
            ckpt_best, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks_ft,
        verbose=1
    )

    with open(os.path.join(args.output_dir, "history_stage2.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist2.history.items()}, f)
    plot_history_curves(hist2.history, os.path.join(args.output_dir, "plots"), prefix="stage2")

    macro_f1_val2 = evaluate_dataset_macro_f1(model, val_ds, class_names)
    print(f"[INFO] Stage 2 validation macro-F1: {macro_f1_val2:.4f}")

    # Save final and best
    last_path = os.path.join(args.output_dir, "last.h5")
    model.save(last_path)
    print(f"[INFO] Saved final model to {last_path} and best-by-val-acc to {ckpt_best}")

if __name__ == "__main__":
    main()
