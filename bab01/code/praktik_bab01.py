"""
Praktik Bab 1 — diekstrak dari naskah buku "Computer Vision & AI".

CATATAN STATUS: kode ini diekstrak dari cuplikan naskah dan disusun ulang
sesuai urutan kemunculan di buku, dengan baris non-Python (perintah shell,
diagram folder, notasi matematika, tampilan matriks mentah) sudah disaring
keluar. BELUM diuji ulang sebagai satu skrip end-to-end yang utuh dalam
repositori ini (berbeda dengan Bab 7, 9, dan 10 yang sudah dijalankan
penuh dan diverifikasi).

Sebelum digunakan: cek kembali dependency, path file, dan urutan eksekusi
sesuai konteks masing-masing bagian di bawah.
"""


# === Membaca Gambar ===
from pathlib import Path
import cv2
BASE_DIR = Path.cwd().parent if Path.cwd().name == "code" else Path.cwd()
IMAGE_PATH = BASE_DIR / "images" / "kucing.jpg"
image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")
print(type(image))
print(image.shape)
height, width, channels = image.shape
print("Tinggi :", height)
print("Lebar :", width)
print("Channel:", channels)


# === Mengintip Sebuah Pixel ===
pixel = image[100, 100]
print(pixel)
print(image[200, 300])
print(image[100:105, 100:105])


# === Tampilkan Gambarnya ===
# CATATAN: cv2.imshow() butuh lingkungan desktop dengan jendela GUI.
# Tidak akan berfungsi di Google Colab, server headless, atau Jupyter
# berbasis browser. Untuk lingkungan tsb, gunakan matplotlib sebagai
# gantinya:
#   import matplotlib.pyplot as plt
#   plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()
cv2.imshow("Gambar Pertama", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# === Coba Sendiri ===
print(image.shape)
print(image[50, 50])
print(image[100, 200])
print(image[300, 400])

