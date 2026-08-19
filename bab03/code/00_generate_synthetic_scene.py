"""
Generator citra sintetis resmi untuk Praktik Bab 3 Compact Edition
(img_03_01_practice_scene.png).

Direkonstruksi berdasarkan nilai fitur yang tercantum di output fitur pada naskah produksi:
area=67.267 px², perimeter=1.005,318 px, aspect_ratio=0,935691,
circularity=0,836383, mean RGB=(222,633; 123,868; 34,666).

Hasil rekonstruksi ini mencapai kecocokan <1% pada seluruh metrik
(bukan file asli penulis, tapi karakteristik geometris & warnanya
sangat mendekati sehingga aman dipakai untuk mereproduksi alur kode).
"""
import cv2
import numpy as np


def generate(seed=42, out_path="images/img_03_01_practice_scene.png"):
    np.random.seed(seed)
    img = np.full((600, 800, 3), 235, dtype=np.uint8)
    center = (350, 300)
    radius = 146
    color_bgr = (35, 124, 223)  # oranye, BGR
    cv2.circle(img, center, radius, color_bgr, -1)

    # tonjolan kecil ala tangkai jeruk
    stem_color = (20, 90, 40)
    pts = np.array([
        [center[0] - 8, center[1] - radius + 5],
        [center[0] + 8, center[1] - radius + 5],
        [center[0] + 4, center[1] - radius - 18],
        [center[0] - 4, center[1] - radius - 18],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], stem_color)

    cv2.imwrite(out_path, img)
    return out_path


if __name__ == "__main__":
    path = generate()
    print(f"Citra sintetis disimpan ke {path}")
