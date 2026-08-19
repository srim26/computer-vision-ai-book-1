"""
Bab 7 - Object Detection
Menghasilkan Output 7.1 (inference sebelum training) dan
Output 7.3 (prediction setelah fine-tuning) pada citra sintetis asli
(bukan foto jalan raya) -- memperbaiki bug aset lama yang salah.
"""
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = {0: "background", 1: "persegi", 2: "lingkaran"}
CLASS_COLORS = {1: "#1F5FA8", 2: "#E2711D"}

def load_manifest_image(idx=0, split="val"):
    with open(f"data/{split}/manifest.json") as f:
        manifest = json.load(f)
    item = manifest[idx]
    img = Image.open(f"data/{split}/{item['file']}").convert("RGB")
    return img, item

def draw_predictions(ax, img, boxes, labels, scores=None, conf_thresh=0.05, title=""):
    ax.imshow(img)
    n_drawn = 0
    if scores is None:
        scores = [None] * len(boxes)
    for box, label, score in zip(boxes, labels, scores):
        if score is not None and score < conf_thresh:
            continue
        x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(int(label), "#888888")
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        label_text = CLASS_NAMES.get(int(label), str(label))
        if score is not None:
            label_text += f" {score:.2f}"
        ax.text(x1, max(y1 - 4, 4), label_text, color="white", fontsize=10, fontweight="bold",
                 bbox=dict(facecolor=color, edgecolor="none", pad=1.5))
        n_drawn += 1
    ax.set_title(f"{title}\n({n_drawn} deteksi ditampilkan, threshold={conf_thresh})", fontsize=12)
    ax.axis("off")

def main():
    img, item = load_manifest_image(idx=0, split="val")
    img_tensor = F.to_tensor(img)

    # --- Output 7.1: model BELUM dilatih (random init) ---
    model_untrained = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=MobileNet_V3_Large_Weights.DEFAULT, num_classes=3
    )
    model_untrained.eval()
    with torch.no_grad():
        pred_before = model_untrained([img_tensor])[0]

    # terapkan NMS supaya visualisasi tidak penuh sesak box tumpang-tindih
    from torchvision.ops import nms
    keep = nms(pred_before["boxes"], pred_before["scores"], iou_threshold=0.3)
    keep = keep[:8]  # batasi maksimal 8 box supaya tetap terbaca
    boxes_before = pred_before["boxes"][keep].numpy()
    labels_before = pred_before["labels"][keep].numpy()
    scores_before = pred_before["scores"][keep].numpy()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    draw_predictions(ax, img, boxes_before, labels_before, scores_before, conf_thresh=0.0,
                      title="Output 7.1 — Inference Faster R-CNN (belum dilatih)")
    plt.tight_layout()
    plt.savefig("../output/output_07_01_sebelum_training.png", facecolor="white")
    plt.close()
    print(f"Output 7.1 (setelah NMS, top-8): {len(boxes_before)} box, scores: {scores_before.tolist()}")

    # --- Output 7.3: model SESUDAH fine-tuning ---
    model_trained = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=MobileNet_V3_Large_Weights.DEFAULT, num_classes=3
    )
    state_dict = torch.load("../output/fasterrcnn_bab07.pt", map_location="cpu")
    model_trained.load_state_dict(state_dict)
    model_trained.eval()

    # coba beberapa citra validation, pilih yang menghasilkan deteksi paling representatif
    with open("data/val/manifest.json") as f:
        val_manifest = json.load(f)

    best_idx, best_img, best_pred, best_n = 0, None, None, -1
    for idx in range(len(val_manifest)):
        img_i, item_i = load_manifest_image(idx=idx, split="val")
        tensor_i = F.to_tensor(img_i)
        with torch.no_grad():
            pred_i = model_trained([tensor_i])[0]
        n_conf = int((pred_i["scores"] > 0.2).sum())
        if n_conf > best_n:
            best_n = n_conf
            best_idx, best_img, best_pred = idx, img_i, pred_i
        if best_n >= len(item_i["labels"]):
            break

    print(f"Citra val index {best_idx} dipilih ({best_n} deteksi >0.2 dari {len(val_manifest)} kandidat dicoba)")
    img, pred_after = best_img, best_pred

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    draw_predictions(ax, img, pred_after["boxes"].numpy(), pred_after["labels"].numpy(),
                      pred_after["scores"].numpy(), conf_thresh=0.05,
                      title="Output 7.3 — Prediction setelah fine-tuning")
    plt.tight_layout()
    plt.savefig("../output/output_07_03_setelah_training.png", facecolor="white")
    plt.close()
    print(f"Output 7.3: {len(pred_after['boxes'])} raw box, scores: {pred_after['scores'][:5].tolist()}")

    print("Ground truth kelas untuk citra ini:", item["labels"])
    print("Selesai. Kedua file PNG siap ditanam ke naskah.")

if __name__ == "__main__":
    main()
