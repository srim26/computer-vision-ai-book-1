"""
Bab 7 - Object Detection
Praktik: Faster R-CNN (torchvision, Apache 2.0) pada dataset sintetis.
Detector tidak memakai pretrained COCO weights. Backbone MobileNetV3 memakai
pretrained ImageNet weights sebagai representasi awal. Konsep tetap sama: RPN, anchor box, NMS, IoU.
"""
import json, os, random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

class SyntheticDetectionDataset(Dataset):
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "manifest.json")) as f:
            self.manifest = json.load(f)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        img = Image.open(os.path.join(self.root, item["file"])).convert("RGB")
        img = F.to_tensor(img)
        boxes = torch.tensor(item["boxes"], dtype=torch.float32)
        labels = torch.tensor(item["labels"], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    train_ds = SyntheticDetectionDataset("data/train")
    val_ds = SyntheticDetectionDataset("data/val")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cpu")
    # num_classes = 3 (background + persegi + lingkaran)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=MobileNet_V3_Large_Weights.DEFAULT, num_classes=3, box_score_thresh=0.01)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    EPOCHS = 10
    history = {"train_loss": []}
    model.train()
    for epoch in range(EPOCHS):
        epoch_losses = []
        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        history["train_loss"].append(mean_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {mean_loss:.4f}", flush=True)

    # Evaluasi sederhana: rata-rata IoU terbaik per gambar val pada threshold confidence 0.5
    model.eval()
    ious = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)
            for pred, target in zip(preds, targets):
                gt_boxes = target["boxes"]
                if len(pred["boxes"]) == 0 or len(gt_boxes) == 0:
                    continue
                for gt in gt_boxes:
                    best_iou = 0.0
                    for pb, score in zip(pred["boxes"], pred["scores"]):
                        if score < 0.05:
                            continue
                        xa = max(gt[0], pb[0]); ya = max(gt[1], pb[1])
                        xb = min(gt[2], pb[2]); yb = min(gt[3], pb[3])
                        inter = max(0, (xb - xa)) * max(0, (yb - ya))
                        area_gt = (gt[2]-gt[0]) * (gt[3]-gt[1])
                        area_pb = (pb[2]-pb[0]) * (pb[3]-pb[1])
                        union = area_gt + area_pb - inter
                        iou = (inter / union).item() if union > 0 else 0.0
                        best_iou = max(best_iou, iou)
                    ious.append(best_iou)
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    result = {
        "seed": SEED,
        "epochs": EPOCHS,
        "train_loss_per_epoch": history["train_loss"],
        "mean_iou_val_conf0.05": mean_iou,
        "n_val_gt_boxes_evaluated": len(ious),
        "torch_version": torch.__version__,
    }
    compact_metrics = compact_validation_summary(model, val_loader, device)
    result.update(compact_metrics)

    torch.save(model.state_dict(), "../output/fasterrcnn_bab07.pt")

    with open("../output/metrics_bab07.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


# --- Compact Edition Bab 7: robust validation summary ---
def compact_validation_summary(model, val_loader, device):
    model.eval()
    best_ious = []
    n_images_with_pred = 0
    n_predictions_total = 0
    n_gt_total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            for pred, target in zip(preds, targets):
                gt_boxes = target["boxes"].cpu()
                pred_boxes = pred["boxes"].cpu()
                scores = pred["scores"].cpu()
                n_gt_total += len(gt_boxes)
                n_predictions_total += len(pred_boxes)
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue
                n_images_with_pred += 1
                # Compare the highest-score prediction with all GT boxes.
                top = int(torch.argmax(scores))
                pb = pred_boxes[top].unsqueeze(0)
                # torchvision.ops.box_iou imported in original script or import locally.
                from torchvision.ops import box_iou
                ious = box_iou(pb, gt_boxes)[0]
                best_ious.append(float(torch.max(ious).item()))
    return {
        "mean_best_iou_val": float(sum(best_ious) / len(best_ious)) if best_ious else 0.0,
        "n_val_images_with_prediction": int(n_images_with_pred),
        "n_val_predictions_total": int(n_predictions_total),
        "n_val_gt_boxes_total": int(n_gt_total),
    }

if __name__ == "__main__":
    main()
