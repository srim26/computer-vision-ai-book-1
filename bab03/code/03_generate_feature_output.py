"""
Bab 3 Compact Edition - Fitur dan Representasi
Generator output visual fitur: contour dan bounding box hasil deteksi objek
pada citra sintetis resmi Bab 3/4 (img_03_01_practice_scene.png).

Menggunakan citra sintetis yang sama dengan Bab 3 (lihat bab03/code/
00_generate_synthetic_scene.py) untuk menjaga konsistensi feature vector
yang dilaporkan pada Output 4.3 dan Tabel 4.1.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# citra sintetis resmi dibuat oleh bab03/code/00_generate_synthetic_scene.py
ROOT = Path(__file__).resolve().parent.parent
IMG_PATH = ROOT / "images" / "img_03_01_practice_scene.png"


def main(out_path=None):
    if out_path is None:
        out_path = ROOT / "output" / "contour_bbox.png"
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        raise FileNotFoundError(
            f"Citra tidak ditemukan di {IMG_PATH}. "
            "Jalankan python code/00_generate_synthetic_scene.py terlebih dahulu."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    result = img.copy()
    cv2.drawContours(result, [c], -1, (0, 255, 0), 3)   # kontur hijau
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)  # bounding box biru

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, result)
    print(f"Output visual fitur disimpan ke {out_path}")


if __name__ == "__main__":
    main()
