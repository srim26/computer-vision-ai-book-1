"""
Bab 5 - Machine Learning
Generator dataset sintetis dua kelas (kelas_A, kelas_B), 120 citra per kelas,
resolusi 128x128, seed 42, dengan distribusi warna sengaja tumpang tindih
supaya SVM linear-ish kesulitan memisahkan sempurna (sesuai deskripsi naskah).
"""
import numpy as np
import cv2
from pathlib import Path

SEED = 42
N_PER_CLASS = 120
SIZE = 128

def make_image(rng, base_color, color_spread, shape_type):
    img = np.full((SIZE, SIZE, 3), 245, dtype=np.uint8)  # latar hampir putih
    # variasi iluminasi latar
    illum = rng.integers(-15, 15)
    img = np.clip(img.astype(int) + illum, 0, 255).astype(np.uint8)

    # posisi & ukuran objek bervariasi
    cx = rng.integers(40, SIZE - 40)
    cy = rng.integers(40, SIZE - 40)
    r = rng.integers(20, 45)

    # warna objek: base_color + noise besar (penyebab overlap antar kelas)
    color = np.clip(
        np.array(base_color) + rng.normal(0, color_spread, 3), 0, 255
    ).astype(int).tolist()
    color_bgr = (color[2], color[1], color[0])

    if shape_type == "circle":
        cv2.circle(img, (cx, cy), r, color_bgr, -1)
    else:
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color_bgr, -1)

    # noise sensor
    noise = rng.normal(0, 8, img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return img

def build_dataset(out_dir="dataset"):
    rng = np.random.default_rng(SEED)
    out_dir = Path(out_dir)

    # kelas_A: warna hangat (oranye-merah), sebagian shape lingkaran
    # kelas_B: warna sedang-hangat juga (kuning-oranye) -> sengaja overlap dgn kelas_A
    class_configs = {
        "kelas_A": {"base_color": (210, 110, 60), "spread": 35, "shapes": ["circle", "rectangle"]},
        "kelas_B": {"base_color": (190, 150, 70), "spread": 35, "shapes": ["circle", "rectangle"]},
    }

    for label, cfg in class_configs.items():
        folder = out_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(N_PER_CLASS):
            shape = cfg["shapes"][i % 2]
            img = make_image(rng, cfg["base_color"], cfg["spread"], shape)
            cv2.imwrite(str(folder / f"{label}_{i:03d}.png"), img)

    print(f"Dataset dibuat: {N_PER_CLASS} citra x 2 kelas di '{out_dir}'")

if __name__ == "__main__":
    build_dataset()
