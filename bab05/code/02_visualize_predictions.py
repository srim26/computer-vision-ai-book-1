"""
Bab 7 - CNN
Skrip pelengkap untuk Output 7.4 (contoh prediksi benar) dan
Output 7.5 (contoh prediksi salah / error analysis).

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
BATCH_SIZE = 64
VAL_SPLIT = 0.20

def main():
    model = keras.models.load_model("output/cnn_eurosat.keras")

    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=VAL_SPLIT, subset="validation",
        seed=SEED, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    )
    class_names = val_ds.class_names

    # Kumpulkan prediksi dari beberapa batch validation sampai cukup contoh benar & salah
    correct_examples, wrong_examples = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        for i in range(len(labels)):
            true_l = int(labels[i])
            pred_l = int(pred_labels[i])
            conf = float(preds[i][pred_l])
            entry = (images[i].numpy().astype("uint8"), true_l, pred_l, conf)
            if true_l == pred_l and len(correct_examples) < 10:
                correct_examples.append(entry)
            elif true_l != pred_l and len(wrong_examples) < 10:
                wrong_examples.append(entry)
        if len(correct_examples) >= 10 and len(wrong_examples) >= 10:
            break

    os.makedirs("output", exist_ok=True)

    # Output 7.4 - contoh prediksi benar
    n = min(10, len(correct_examples))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2), dpi=200)
    for i, ax in enumerate(axes.flat):
        if i < n:
            img, true_l, pred_l, conf = correct_examples[i]
            ax.imshow(img)
            ax.set_title(f"{class_names[pred_l]}\n({conf:.2f})", fontsize=9, color="#1a7a3c")
        ax.axis("off")
    fig.suptitle("Contoh Prediksi Benar - Validation Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/output_7_4_prediksi_benar.png", facecolor="white")
    plt.close()
    print(f"Tersimpan: output/output_7_4_prediksi_benar.png ({n} contoh)")

    # Output 7.5 - contoh prediksi salah
    n = min(10, len(wrong_examples))
    if n == 0:
        print("PERINGATAN: tidak ditemukan prediksi salah pada batch yang diperiksa. "
              "Coba jalankan lagi dengan lebih banyak batch, atau model sudah sangat akurat "
              "pada sampel ini.")
    else:
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5.2 * rows / 2), dpi=200)
        axes_flat = axes.flat if n > 1 else [axes]
        for i, ax in enumerate(axes_flat):
            if i < n:
                img, true_l, pred_l, conf = wrong_examples[i]
                ax.imshow(img)
                ax.set_title(
                    f"Prediksi: {class_names[pred_l]} ({conf:.2f})\nSebenarnya: {class_names[true_l]}",
                    fontsize=8, color="#c0392b"
                )
            ax.axis("off")
        fig.suptitle("Contoh Prediksi Salah - Validation Set", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("output/output_7_5_prediksi_salah.png", facecolor="white")
        plt.close()
        print(f"Tersimpan: output/output_7_5_prediksi_salah.png ({n} contoh)")

    print("Kirim kedua file PNG ini kembali untuk disisipkan ke naskah.")

if __name__ == "__main__":
    main()
