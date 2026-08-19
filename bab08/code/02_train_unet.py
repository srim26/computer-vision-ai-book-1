"""
Bab 10 - Image Segmentation
Praktik: Mini U-Net (encoder-decoder + skip connections) pada dataset sintetis.
Loss: weighted BCE + soft Dice. Evaluasi: hard Dice, hard IoU pada threshold 0.5.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import os, json, glob

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

IMG_SIZE = 128

def load_split(img_dir, mask_dir):
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    X, Y = [], []
    for f in files:
        img = np.array(Image.open(f).convert("L"), dtype=np.float32) / 255.0
        mname = os.path.join(mask_dir, os.path.basename(f))
        mask = np.array(Image.open(mname).convert("L"), dtype=np.float32) / 255.0
        X.append(img[..., None])
        Y.append(mask[..., None])
    return np.array(X), np.array(Y)

def soft_dice_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def hard_dice_metric(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    return soft_dice_metric(y_true, y_pred_bin, smooth)

def hard_iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    dice_loss = 1.0 - soft_dice_metric(y_true, y_pred)
    return bce + dice_loss

def build_mini_unet():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    p3 = layers.MaxPooling2D()(c3)

    bn = layers.Conv2D(128, 3, activation="relu", padding="same")(p3)

    u3 = layers.UpSampling2D()(bn)
    u3 = layers.Concatenate()([u3, c3])
    d3 = layers.Conv2D(64, 3, activation="relu", padding="same")(u3)

    u2 = layers.UpSampling2D()(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = layers.Conv2D(32, 3, activation="relu", padding="same")(u2)

    u1 = layers.UpSampling2D()(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = layers.Conv2D(16, 3, activation="relu", padding="same")(u1)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return keras.Model(inputs, outputs)

def main():
    X_train, y_train = load_split("../datasets/sample/images/train", "../datasets/sample/masks/train")
    X_val, y_val = load_split("../datasets/sample/images/val", "../datasets/sample/masks/val")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    model = build_mini_unet()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=["accuracy", soft_dice_metric, hard_dice_metric, hard_iou_metric],
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20, batch_size=4, verbose=2,
    )

    os.makedirs("../output", exist_ok=True)
    metrics = {
        "seed": SEED,
        "epochs": 20,
        "final_train_hard_dice": float(history.history["hard_dice_metric"][-1]),
        "final_val_hard_dice": float(history.history["val_hard_dice_metric"][-1]),
        "final_train_hard_iou": float(history.history["hard_iou_metric"][-1]),
        "final_val_hard_iou": float(history.history["val_hard_iou_metric"][-1]),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }
    with open("../output/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    model.save("../output/mini_unet.keras")
    print("Selesai. Val hard Dice:", metrics["final_val_hard_dice"], "| Val hard IoU:", metrics["final_val_hard_iou"])

if __name__ == "__main__":
    main()
