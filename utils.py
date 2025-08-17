#!/usr/bin/env python3
# Utilities: dataset loading, class weights, plots, evaluation

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _make_ds_from_dir(dir_path, img_size, batch_size, shuffle=True):
    return keras.preprocessing.image_dataset_from_directory(
        dir_path,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",  # MULTICLASS
        shuffle=shuffle,
    ).prefetch(tf.data.AUTOTUNE)

def get_datasets(data_dir: str, img_size: int, batch_size: int):
    """
    Returns: train_ds, val_ds, class_names
    Expects:
      data/
        train/
          classA/
          classB/
          ...
        val/   (optional; if missing, uses validation_split on train/)
          classA/
          classB/
          ...
    """
    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")

    if os.path.isdir(val_root):
        train_ds = _make_ds_from_dir(train_root, img_size, batch_size, shuffle=True)
        val_ds   = _make_ds_from_dir(val_root, img_size, batch_size, shuffle=False)
        class_names = train_ds.class_names
    else:
        # Use validation_split on train
        train_ds = keras.preprocessing.image_dataset_from_directory(
            train_root,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True,
        )
        val_ds = keras.preprocessing.image_dataset_from_directory(
            train_root,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False,
        )
        class_names = train_ds.class_names

    # cache+prefetch
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names

def compute_class_weights(train_ds, num_classes: int):
    """
    Compute class weights from a tf.data Dataset where labels are one-hot (categorical).
    Returns dict: {class_index: weight}
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in train_ds.unbatch():
        idx = int(tf.argmax(y).numpy())
        counts[idx] += 1
    total = counts.sum()
    # Inverse frequency weighting
    weights = total / (num_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}

def plot_history_curves(history_dict, out_dir: str, prefix: str = ""):
    os.makedirs(out_dir, exist_ok=True)

    # Loss
    fig = plt.figure()
    plt.plot(history_dict.get("loss", []), label="train")
    plt.plot(history_dict.get("val_loss", []), label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("CategoricalCrossentropy"); plt.legend()
    fig.savefig(os.path.join(out_dir, f"{prefix}_loss.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Accuracy
    if "accuracy" in history_dict:
        fig = plt.figure()
        plt.plot(history_dict.get("accuracy", []), label="train")
        plt.plot(history_dict.get("val_accuracy", []), label="val")
        plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
        fig.savefig(os.path.join(out_dir, f"{prefix}_accuracy.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Top-5
    if "top5_acc" in history_dict:
        fig = plt.figure()
        plt.plot(history_dict.get("top5_acc", []), label="train")
        plt.plot(history_dict.get("val_top5_acc", []), label="val")
        plt.title("Top-5 Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Top-5 Acc"); plt.legend()
        fig.savefig(os.path.join(out_dir, f"{prefix}_top5.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

def evaluate_dataset_macro_f1(model, ds, class_names):
    """Compute macro-F1 on a dataset with one-hot labels."""
    y_true = []
    y_pred = []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(p, axis=1).tolist())
    f1 = f1_score(y_true, y_pred, average="macro")
    return f1
