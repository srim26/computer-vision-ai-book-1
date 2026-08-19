"""
Bab 7 - CNN
Skrip pelengkap untuk Output 7.6 dan 7.7: feature map dari convolution
layer pertama dan kedua, pada satu citra validation.

Jalankan SETELAH 01_cnn_eurosat.py selesai (butuh output/cnn_eurosat.keras).
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

SEED = 42
DATA_DIR = "data/2750"
IMG_SIZE = 64

def main():
    model = keras.models.load_model("output/cnn_eurosat.keras")

    # Ambil satu citra contoh dari dataset (deterministik lewat seed yang sama)
    ds = keras.utils.image_dataset_from_directory(
        DATA_DIR, seed=SEED, image_size=(IMG_SIZE, IMG_SIZE), batch_size=1
    )
    sample_image, sample_label = next(iter(ds))
    class_names = ds.class_names
    print(f"Citra contoh: kelas '{class_names[int(sample_label[0])]}'")

    # Cari layer Conv2D pertama dan kedua secara otomatis
    conv_layers = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)]
    if len(conv_layers) < 2:
        raise RuntimeError(f"Model hanya punya {len(conv_layers)} layer Conv2D, butuh minimal 2.")
    conv1, conv2 = conv_layers[0], conv_layers[1]

    activation_model = keras.Model(
        inputs=model.inputs, outputs=[conv1.output, conv2.output]
    )
    feat1, feat2 = activation_model.predict(sample_image, verbose=0)
    print(f"Shape feature map conv1: {feat1.shape}")
    print(f"Shape feature map conv2: {feat2.shape}")

    os.makedirs("output", exist_ok=True)

    def plot_feature_maps(features, title, filename, n=16):
        n = min(n, features.shape[-1])
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(9, 2.3 * rows), dpi=200)
        for i, ax in enumerate(axes.flat):
            if i < n:
                ax.imshow(features[0, :, :, i], cmap="viridis")
            ax.axis("off")
        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"output/{filename}", facecolor="white")
        plt.close()
        print(f"Tersimpan: output/{filename}")

    plot_feature_maps(
        feat1, f"Feature Map Convolution Layer 1 (shape {feat1.shape[1:]})",
        "output_7_6_feature_map_conv1.png"
    )
    plot_feature_maps(
        feat2, f"Feature Map Convolution Layer 2 (shape {feat2.shape[1:]})",
        "output_7_7_feature_map_conv2.png"
    )
    print("Kirim kedua file PNG ini kembali untuk disisipkan ke naskah, "
          "beserta shape yang tercetak di atas (untuk teks 7.10).")

if __name__ == "__main__":
    main()
