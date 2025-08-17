#!/usr/bin/env python3
# Batch inference: predict class probabilities for a folder of images.

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

def parse_args():
    p = argparse.ArgumentParser(description="Predict animal classes with a trained ResNet50 model.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--class_names", type=str, required=True, help="Path to class_names.json saved during training")
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--out", type=str, default="predictions.csv")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()

def list_image_files(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    return [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.lower().endswith(exts)]

def load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # ResNet50 preprocessing
    return img

def main():
    args = parse_args()

    with open(args.class_names, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    files = list_image_files(args.images_dir)
    if not files:
        raise SystemExit(f"No images found in {args.images_dir}")

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(lambda p: (load_image(p, args.img_size), p), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = keras.models.load_model(args.model_path, compile=False)

    probs = []
    paths = []
    for batch_imgs, batch_paths in ds:
        p = model.predict(batch_imgs, verbose=0)
        probs.append(p)
        paths.extend([x.numpy().decode("utf-8") for x in batch_paths])

    probs = np.concatenate(probs, axis=0)  # (N, C)
    preds = probs.argmax(axis=1)
    pred_labels = [class_names[i] for i in preds]

    # Build DataFrame with per-class probabilities
    df = pd.DataFrame({"filepath": paths, "pred_label": pred_labels})
    for i, cname in enumerate(class_names):
        df[f"prob_{cname}"] = probs[:, i]

    df.to_csv(args.out, index=False)
    print(f"[INFO] Saved predictions to {args.out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
