# Computer Vision & AI — Dari Pixel hingga Analisis Citra Medis

Repository pendamping untuk buku **Computer Vision & AI: Dari Pixel hingga Analisis Citra Medis**  
oleh **Sri Mulyono, Kusworo Adi, dan Aris Sugiharto**.

Repository ini berisi kode praktik, aset yang diizinkan untuk dibagikan, dataset sintetis,
dan output eksperimen yang mendukung **Compact Edition 10 bab**. Struktur repository
mengikuti nomor bab pada buku, bukan struktur produksi 13 bab sebelumnya.

## Struktur

| Folder | Materi utama |
|---|---|
| `bab01/` | Citra sebagai data numerik |
| `bab02/` | Pixel dan citra digital |
| `bab03/` | Image processing, fitur, dan feature vector |
| `bab04/` | Machine Learning dan classification |
| `bab05/` | Convolutional Neural Network (CNN) |
| `bab06/` | Transfer Learning; Vision Transformer dibahas konseptual dalam buku |
| `bab07/` | Object Detection dengan Faster R-CNN |
| `bab08/` | Image Segmentation dengan Mini U-Net |
| `bab09/` | Medical Imaging, CT sintetis, windowing, dan measurement |
| `bab10/` | Medical AI dan evaluasi sintetis |

## Environment

Buku menggunakan satu environment utama bernama `cvai-book`.

```bash
conda env create -f environment.yml
conda activate cvai-book
```

Alternatif:

```bash
pip install -r requirements.txt
```

> Beberapa praktik mengunduh dataset atau pretrained weights dari sumber resmi dan
> karena itu memerlukan koneksi internet saat pertama dijalankan.

## Cara menggunakan

1. Aktifkan environment `cvai-book`.
2. Masuk ke folder bab yang ingin dipelajari.
3. Baca komentar pada skrip dan README bab (jika tersedia).
4. Jalankan kode dari direktori bab agar path relatif tetap sesuai.
5. Cocokkan output dengan penjelasan pada buku.

Contoh:

```bash
cd bab07
python code/generate_dataset.py
python code/train_fasterrcnn.py
```

## Reproducibility

Output numerik yang dicantumkan sebagai hasil eksperimen harus berasal dari kode,
aset, dan konfigurasi yang terdokumentasi. Dataset sintetis menggunakan seed tetap
bila tersedia. Praktik Medical Imaging/Medical AI dalam repository ini bersifat
pedagogis dan **bukan** validasi klinis.

## Data dan aset

- `kucing.jpg` dan `jeruk.jpg`: foto milik penulis.
- Dataset sintetis: dibuat programatik untuk praktik buku.
- Dataset/model pihak ketiga hanya digunakan pada bagian yang memang memerlukannya
  dan ketentuannya diringkas di `ASSET_LICENSES.md`.
- Dataset atau metode lama yang tidak lagi dipakai pada Compact Edition tidak
  dicantumkan sebagai bagian aktif repository ini.

## Object Detection

Bab 7 menggunakan **Faster R-CNN (torchvision)** dan dataset sintetis. Repository
Compact Edition tidak menggunakan Ultralytics YOLO/COCO8 sebagai praktik buku.

## Medical AI

Materi dan kode pada Bab 9–10 ditujukan untuk pendidikan. Output model, mask,
measurement, probability, atau visualisasi tidak boleh diperlakukan sebagai
diagnosis medis atau pengganti tenaga kesehatan.

## Status Release Candidate

Paket ini adalah **GitHub Release Candidate v1.0**. Clean-run Bab 1–10 telah dilakukan
bertahap pada environment penulis. Pemeriksaan struktur tambahan tersedia melalui
`python verify_repository.py`. **Jangan ubah repository menjadi Public sebelum keputusan
lisensi kode dan audit lisensi aset final selesai.**

## Sitasi

Informasi bibliografis final buku/ISBN dapat ditambahkan setelah diterbitkan.

## Lisensi

Lisensi kode dan lisensi aset dipisahkan. Lihat `LICENSE` dan `ASSET_LICENSES.md`.
