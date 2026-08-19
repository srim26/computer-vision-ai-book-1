"""
Bab 7 - Convolutional Neural Network (CNN)
Praktik: CNN sederhana pada dataset EuroSAT.

STATUS: Kode ini BELUM dieksekusi di lingkungan penulis (dataset EuroSAT
tidak dapat diunduh dari sandbox verifikasi karena pembatasan jaringan).
Jalankan skrip ini di lingkungan cvai-book biasa untuk menghasilkan
Output 7.1-7.7 dan metrics.json yang sebenarnya sebelum Draft final dikunci.

Dataset : EuroSAT (Helber et al., 2018) - MIT License
Sumber  : https://github.com/phelber/EuroSAT
Kelas   : 10 (AnnualCrop, Forest, HerbaceousVegetation, Highway,
          Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os, json, hashlib, zipfile, urllib.request

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

DATA_ROOT = "data"
ZIP_PATH = os.path.join(DATA_ROOT, "EuroSAT.zip")
DATA_DIR = os.path.join(DATA_ROOT, "2750")  # struktur folder resmi setelah diekstrak
EUROSAT_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
EUROSAT_MD5 = "c8fa014336c82ac7804f0398fcb19387"  # checksum resmi (sumber: torchvision)
IMG_SIZE = 64
BATCH_SIZE = 64
VAL_SPLIT = 0.20

def _md5(path, chunk=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def download_eurosat():
    if os.path.isdir(DATA_DIR):
        print(f"Dataset sudah ada di {DATA_DIR}, lewati unduhan.")
        return
    os.makedirs(DATA_ROOT, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"Mengunduh EuroSAT dari {EUROSAT_URL} ...")
        try:
            urllib.request.urlretrieve(EUROSAT_URL, ZIP_PATH)
        except Exception as e:
            raise RuntimeError(
                "Gagal mengunduh EuroSAT dari madm.dfki.de. Server sumber "
                "kadang tidak stabil. Coba lagi beberapa saat, atau unduh manual "
                f"dari {EUROSAT_URL} lalu ekstrak ke folder '{DATA_ROOT}/'."
            ) from e
    actual_md5 = _md5(ZIP_PATH)
    if actual_md5 != EUROSAT_MD5:
        raise RuntimeError(
            f"Checksum EuroSAT.zip tidak cocok (dapat {actual_md5}, "
            f"seharusnya {EUROSAT_MD5}). File mungkin rusak/korup - hapus dan unduh ulang."
        )
    print("Checksum cocok. Mengekstrak...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA_ROOT)
    print(f"Selesai. Dataset tersedia di {DATA_DIR}")


def load_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

def build_model(num_classes):
    model = keras.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    download_eurosat()
    train_ds, val_ds, class_names = load_datasets()
    model = build_model(num_classes=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    os.makedirs("output", exist_ok=True)
    metrics = {
        "seed": SEED,
        "classes": class_names,
        "train_accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
    }
    with open("output/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    model.save("output/cnn_eurosat.keras")
    print("Selesai. Lihat output/metrics.json untuk angka clean-run yang sebenarnya.")

if __name__ == "__main__":
    main()
