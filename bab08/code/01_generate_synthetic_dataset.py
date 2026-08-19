"""
Bab 10 - Image Segmentation
Generator dataset sintetis: lingkaran pada latar gelap (image + binary mask).
Deterministik (seed=42), tanpa aset pihak ketiga. 48 pasang: 36 train, 12 val.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os, json

SEED = 42
IMG_SIZE = 128
N_TRAIN = 36
N_VAL = 12

def make_pair(rng):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=int(rng.integers(10, 30)))
    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    r = int(rng.integers(25, 45))
    cx = int(rng.integers(r + 10, IMG_SIZE - r - 10))
    cy = int(rng.integers(r + 10, IMG_SIZE - r - 10))
    fill_val = int(rng.integers(160, 230))

    draw_img.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill_val)
    draw_mask.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    # noise ringan supaya tidak terlalu sempurna (lebih realistis untuk segmentasi)
    noise = rng.normal(0, 6, (IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.5))

    return img, mask

def build_split(n, seed_offset, img_dir, mask_dir):
    rng = np.random.default_rng(SEED + seed_offset)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    manifest = []
    for i in range(n):
        img, mask = make_pair(rng)
        fname = f"sample_{i:03d}.png"
        img.save(os.path.join(img_dir, fname))
        mask.save(os.path.join(mask_dir, fname))
        manifest.append(fname)
    return manifest

if __name__ == "__main__":
    train_files = build_split(N_TRAIN, 0, "../datasets/sample/images/train", "../datasets/sample/masks/train")
    val_files = build_split(N_VAL, 1000, "../datasets/sample/images/val", "../datasets/sample/masks/val")
    with open("../datasets/sample/manifest.json", "w") as f:
        json.dump({"train": train_files, "val": val_files, "seed": SEED}, f, indent=2)
    print(f"Dataset sintetis dibuat: {N_TRAIN} train, {N_VAL} val.")
