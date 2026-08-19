"""
Bab 8 - Image Segmentation
Menghasilkan Output 8.1 (kurva Dice/IoU dari clean-run) dan
Output 8.2 (prediction mask vs ground truth pada data validation).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import glob, os

C_TRAIN = "#1f5fa8"
C_VAL = "#e2711d"

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

with open("../output/metrics.json") as f:
    metrics = json.load(f)
hist = metrics["history"]
epochs = list(range(1, len(hist["hard_dice_metric"]) + 1))

# Output 8.1 - Kurva Dice dan IoU (train vs validation)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)

ax = axes[0]
ax.plot(epochs, hist["hard_dice_metric"], marker="o", ms=3, color=C_TRAIN, label="Training Dice")
ax.plot(epochs, hist["val_hard_dice_metric"], marker="o", ms=3, color=C_VAL, label="Validation Dice")
ax.set_title("Dice Coefficient", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Dice")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
ax.plot(epochs, hist["hard_iou_metric"], marker="o", ms=3, color=C_TRAIN, label="Training IoU")
ax.plot(epochs, hist["val_hard_iou_metric"], marker="o", ms=3, color=C_VAL, label="Validation IoU")
ax.set_title("IoU (Intersection over Union)", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False, fontsize=9)

fig.suptitle("Output 8.1 \u2014 Metrik Validation Mini U-Net dari Clean-Run Bab 8", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../output/output_08_01_metrik_validation.png", facecolor="white")
plt.close()
print("Tersimpan: output_08_01_metrik_validation.png")

# Output 8.2 - Prediction mask vs ground truth pada data validation
model = keras.models.load_model(
    "../output/mini_unet.keras",
    custom_objects={
        "combined_loss": lambda yt, yp: 0.0,  # placeholder, hanya untuk load bobot inference
    },
    compile=False,
)
X_val, Y_val = load_split("../datasets/sample/images/val", "../datasets/sample/masks/val")
preds = model.predict(X_val, verbose=0)

n_show = 4
fig, axes = plt.subplots(3, n_show, figsize=(10, 7.5), dpi=200)
row_titles = ["Citra Input", "Ground Truth Mask", "Prediction Mask"]
for col in range(n_show):
    axes[0, col].imshow(X_val[col, ..., 0], cmap="gray")
    axes[1, col].imshow(Y_val[col, ..., 0], cmap="gray")
    pred_bin = (preds[col, ..., 0] > 0.5).astype(np.float32)
    axes[2, col].imshow(pred_bin, cmap="gray")
    for row in range(3):
        axes[row, col].axis("off")
for row in range(3):
    axes[row, 0].set_ylabel(row_titles[row], fontsize=10)
    axes[row, 0].axis("on")
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    for spine in axes[row, 0].spines.values():
        spine.set_visible(False)

fig.suptitle("Output 8.2 \u2014 Prediction Mask vs Ground Truth (Data Validation)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../output/output_08_02_prediction_mask.png", facecolor="white")
plt.close()
print("Tersimpan: output_08_02_prediction_mask.png")
