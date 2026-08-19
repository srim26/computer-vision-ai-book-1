"""
Bab 9 - Object Detection
Generator dataset sintetis: kotak & lingkaran dengan bounding box.
Sepenuhnya deterministik (seed=42), tanpa aset pihak ketiga.
"""
import numpy as np
from PIL import Image, ImageDraw
import os, json

SEED = 42
IMG_SIZE = 128
N_TRAIN = 80
N_VAL = 20

def make_scene(rng):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    n_obj = rng.integers(1, 4)
    boxes, labels = [], []
    for _ in range(n_obj):
        shape = int(rng.integers(0, 2))  # 0=persegi(1), 1=lingkaran(2)
        size = rng.integers(15, 35)
        x = int(rng.integers(0, IMG_SIZE - size))
        y = int(rng.integers(0, IMG_SIZE - size))
        size = int(size)
        color = tuple(rng.integers(0, 180, size=3).tolist())
        if shape == 0:
            draw.rectangle([x, y, x + size, y + size], fill=color)
            labels.append(1)
        else:
            draw.ellipse([x, y, x + size, y + size], fill=color)
            labels.append(2)
        boxes.append([x, y, x + size, y + size])
    return img, boxes, labels

def build_split(n, seed_offset, out_dir):
    rng = np.random.default_rng(SEED + seed_offset)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i in range(n):
        img, boxes, labels = make_scene(rng)
        fname = f"scene_{i:03d}.png"
        img.save(os.path.join(out_dir, fname))
        manifest.append({"file": fname, "boxes": boxes, "labels": labels})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest

if __name__ == "__main__":
    build_split(N_TRAIN, 0, "data/train")
    build_split(N_VAL, 1000, "data/val")
    print(f"Dataset sintetis dibuat: {N_TRAIN} train, {N_VAL} val. Kelas: 1=persegi, 2=lingkaran.")
