"""
Praktik Bab 2 — diekstrak dari naskah buku "Computer Vision & AI".

CATATAN STATUS: kode ini diekstrak dari cuplikan naskah dan disusun ulang
sesuai urutan kemunculan di buku, dengan baris non-Python (perintah shell,
diagram folder, notasi matematika, tampilan matriks mentah) sudah disaring
keluar. BELUM diuji ulang sebagai satu skrip end-to-end yang utuh dalam
repositori ini (berbeda dengan Bab 7, 9, dan 10 yang sudah dijalankan
penuh dan diverifikasi).

Sebelum digunakan: cek kembali dependency, path file, dan urutan eksekusi
sesuai konteks masing-masing bagian di bawah.
"""


# === 2.5 RGB, BGR, dan Channel Warna ===
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/jeruk.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off")
plt.show()


# === 2.6 Grayscale: Ketika Satu Pixel Cukup dengan Satu Nilai ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# === 2.7 Ketika Gambar Berubah Menjadi Matriks ===
# [baris non-Python disaring otomatis] 15 22 35 48 57
# [baris non-Python disaring otomatis] 26 34 47 61 72
# [baris non-Python disaring otomatis] 39 51 65 78 89
# [baris non-Python disaring otomatis] 53 67 81 94 108
# [baris non-Python disaring otomatis] 69 84 98 113 127
import numpy as np

# CATATAN: variabel diberi nama 'gray_contoh_5x5' (bukan 'gray') supaya tidak
# menimpa hasil konversi grayscale asli dari citra jeruk.jpg di bagian 2.6 di atas.
# Di naskah buku, ini ditampilkan sebagai ilustrasi array kecil yang berdiri sendiri.
gray_contoh_5x5 = np.array([
    [15, 22, 35, 48, 57],
    [26, 34, 47, 61, 72],
    [39, 51, 65, 78, 89],
    [53, 67, 81, 94, 108],
    [69, 84, 98, 113, 127]
], dtype=np.uint8)

print(gray_contoh_5x5.shape) # (5, 5)


# === 2.8 Koordinat Pixel: Di Mana Letak Sebuah Titik? ===
x = 200
y = 100
pixel = img[y, x]

area = img[50:150, 100:250]


# === Membaca dan memeriksa citra ===
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/jeruk.jpg")

if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")

print(type(img))
print(img.shape)


# === Mengintip satu pixel ===
x = 200
y = 100
pixel = img[y, x]
print(pixel)


# === Menampilkan warna dengan benar ===
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off")
plt.show()


# === Mengubah ke grayscale dan melihat array ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray.shape)
print(gray[y, x])
print(gray[0:5, 0:5])

plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.show()


# === 2.10 Eksperimen: Apa yang Terjadi Jika Pixel Kita Ubah? ===
ubah = img_rgb.copy()
ubah[50:150, 50:150] = [0, 0, 0]

plt.imshow(ubah)
plt.axis("off")
plt.show()
# Putih
ubah[50:150, 50:150] = [255, 255, 255]

# Merah pada citra RGB
ubah[50:150, 50:150] = [255, 0, 0]

# Satu pixel merah
x, y = 200, 100
ubah[y, x] = [255, 0, 0]

