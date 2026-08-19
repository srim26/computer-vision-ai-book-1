"""
Praktik Bab 3 Compact Edition — diselaraskan dari naskah buku "Computer Vision & AI".

CATATAN STATUS: kode ini diekstrak dari cuplikan naskah dan disusun ulang
sesuai urutan kemunculan di buku, dengan baris non-Python (perintah shell,
diagram folder, notasi matematika, tampilan matriks mentah) sudah disaring
keluar. BELUM diuji ulang sebagai satu skrip end-to-end yang utuh dalam
repositori ini (berbeda dengan Bab 7, 9, dan 10 yang sudah dijalankan
penuh dan diverifikasi).

Sebelum digunakan: cek kembali dependency, path file, dan urutan eksekusi
sesuai konteks masing-masing bagian di bawah.
"""


# === Setup (ditambahkan agar file dapat dijalankan mandiri) ===
# Bagian ini tidak eksplisit di naskah karena praktik Bab 3 melanjutkan
# konteks pembacaan citra dari bab-bab sebelumnya (img sudah dimuat).
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
img = cv2.imread(str(ROOT / "images" / "jeruk.jpg"))
if img is None:
    raise FileNotFoundError("Sesuaikan path ke citra praktik Bab 3.")


# === Brightness dan Contrast ===
# img menggunakan citra input yang sama
brighter = cv2.convertScaleAbs(img, alpha=1.0, beta=40)


# === Histogram: Melihat Distribusi Intensitas ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.hist(gray.ravel(), bins=256, range=[0, 256])
plt.xlabel("Intensitas")
plt.ylabel("Jumlah pixel")
plt.show()


# === Histogram Equalization dan CLAHE ===
equalized = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)


# === RGB, Grayscale, dan HSV ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# === Eksperimen: Mencari Warna Tertentu ===
lower = np.array([35, 50, 50])
upper = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower, upper)


# === Mean Filter ===
blur_mean = cv2.blur(gray, (3, 3))


# === Gaussian Blur ===
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)


# === Median Filter ===
median = cv2.medianBlur(gray, 5)


# === Sharpening ===
kernel = np.array([[0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]])
sharp = cv2.filter2D(gray, -1, kernel)


# === Sobel dan Laplacian ===
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)


# === Canny Edge Detector ===
edges = cv2.Canny(gray, 100, 200)


# === Adaptive Thresholding ===
adaptive = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2)


# === Otsu Thresholding ===
# CATATAN: 'blur' tidak eksplisit didefinisikan pada cuplikan naskah di
# titik ini; ditambahkan Gaussian blur standar sebagai preprocessing umum
# sebelum Otsu thresholding.
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# === 3.8 Morfologi: Merapikan Bentuk ===
kernel = np.ones((3, 3), np.uint8)


# === Erosion dan Dilation ===
eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)


# === Opening dan Closing ===
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


# === 3.9 Dari Pixel Menuju Bentuk ===
contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)


# === Area, Perimeter, Bounding Rectangle, dan Centroid ===
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, True)
x, y, w, h = cv2.boundingRect(cnt)
M = cv2.moments(cnt)
if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])


# === Menyiapkan Lingkungan ===
import cv2
import numpy as np
import matplotlib.pyplot as plt


# === 1. Membaca Citra ===
IMAGE_PATH = ROOT / "images" / "img_03_01_practice_scene.png"
img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError("Citra tidak ditemukan")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# === 2. Grayscale dan Histogram ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.hist(gray.ravel(), bins=256, range=[0, 256])
plt.show()


# === 3. Filtering dan Edge ===
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 100, 200)


# === 4. Thresholding ===
_, binary = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# === 5. Morphology ===
kernel = np.ones((3, 3), np.uint8)
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

