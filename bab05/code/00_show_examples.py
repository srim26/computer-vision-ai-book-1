"""
Bab 7 - CNN
Skrip pelengkap untuk menghasilkan Output 7.1: contoh citra EuroSAT asli.

Kenapa skrip terpisah? Claude (asisten AI) tidak memiliki akses ke berkas
citra satelit EuroSAT yang sebenarnya (dataset tidak bisa diunduh dari
lingkungan verifikasinya), sehingga contoh citra pada Output 7.1 TIDAK
BOLEH dibuat oleh AI - harus dihasilkan langsung dari data asli di
komputer Anda agar benar-benar merepresentasikan dataset yang dipakai.

Jalankan setelah 01_cnn_eurosat.py (supaya folder data/2750 sudah ada).
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

SEED = 42
DATA_DIR = "data/2750"
IMG_SIZE = 64

def main():
    ds = keras.utils.image_dataset_from_directory(
        DATA_DIR, seed=SEED, image_size=(IMG_SIZE, IMG_SIZE), batch_size=16
    )
    class_names = ds.class_names

    images, labels = next(iter(ds))

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2), dpi=200)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(class_names[labels[i]], fontsize=10)
        ax.axis("off")
    fig.suptitle("Contoh Citra EuroSAT (Sentinel-2)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/output_7_1_contoh_citra.png", facecolor="white")
    print("Tersimpan: output/output_7_1_contoh_citra.png")
    print("Kirim file ini kembali agar bisa disisipkan ke naskah.")

if __name__ == "__main__":
    main()
