"""
Praktik Bab 3 — Feature Extraction — diekstrak dari naskah buku "Computer Vision & AI".

CATATAN STATUS: kode ini diekstrak dari cuplikan naskah dan disusun ulang
sesuai urutan kemunculan di buku, dengan baris non-Python (perintah shell,
diagram folder, notasi matematika, tampilan matriks mentah) sudah disaring
keluar. BELUM diuji ulang sebagai satu skrip end-to-end yang utuh dalam
repositori ini (berbeda dengan Bab 7, 9, dan 10 yang sudah dijalankan
penuh dan diverifikasi).

Sebelum digunakan: cek kembali dependency, path file, dan urutan eksekusi
sesuai konteks masing-masing bagian di bawah.
"""


# === 4.9 Praktik: Membuat Komputer “Mendeskripsikan” Sebuah Objek ===
import cv2
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
image = cv2.imread(str(ROOT / "images" / "jeruk.jpg"))
if image is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise ValueError("Kontur objek tidak ditemukan.")

contour = max(contours, key=cv2.contourArea)

# Mask khusus objek utama
object_mask = np.zeros(gray.shape, dtype=np.uint8)
cv2.drawContours(object_mask, [contour], -1, 255, -1)
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)

x, y, w, h = cv2.boundingRect(contour)
aspect_ratio = w / h

circularity = (
    4 * np.pi * area / (perimeter ** 2)
    if perimeter > 0 else 0
)

mean_bgr = cv2.mean(image, mask=object_mask)[:3]
mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])

feature_vector = np.array([
    area, perimeter, aspect_ratio, circularity,
    mean_rgb[0], mean_rgb[1], mean_rgb[2]
])

print(feature_vector)
result = image.copy()
cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
cv2.rectangle(result, (x, y), (x + w, y + h),
    (255, 0, 0), 2)
out_path = ROOT / "output" / "hasil_feature.jpg"
out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), result)
print(f"Hasil visual disimpan ke {out_path}")

