# Computer Vision & AI
## Dari Pixel hingga Analisis Citra Medis

Repository pendamping buku **Computer Vision & AI: Dari Pixel hingga Analisis Citra Medis**.

Repository ini menyediakan kode Python, contoh citra, dataset sintetis, serta praktik yang digunakan untuk membantu pembaca memahami perjalanan dari representasi pixel hingga penerapan Computer Vision dan Artificial Intelligence pada citra medis.

Buku dirancang dengan pendekatan bertahap:

**Pixel → Citra Digital → Image Processing → Representasi Fitur → Machine Learning → Deep Learning → CNN → Transfer Learning → Object Detection → Image Segmentation → Medical Imaging → Medical AI**

Repository ini bukan pengganti pembahasan dalam buku. Kode dibuat ringkas agar pembaca dapat mencoba kembali konsep dan eksperimen yang dibahas pada masing-masing bab.

---

## Struktur Repository

```text
.
├── bab01/
├── bab02/
├── bab03/
├── bab04/
├── bab05/
├── bab06/
├── bab07/
├── bab08/
├── bab09/
├── bab10/
├── ASSET_LICENSES.md
├── environment.yml
├── requirements.txt
└── verify_repository.py
```

### Ringkasan Praktik

| Bab | Topik praktik |
|---|---|
| `bab01` | Membaca citra dan mengenali citra sebagai data numerik |
| `bab02` | Pixel, array, warna, dan citra grayscale |
| `bab03` | Image processing dan ekstraksi fitur |
| `bab04` | Machine Learning untuk klasifikasi citra |
| `bab05` | Convolutional Neural Network (CNN) |
| `bab06` | Transfer Learning dengan MobileNetV2 |
| `bab07` | Object Detection dengan Faster R-CNN |
| `bab08` | Image Segmentation dengan Mini U-Net |
| `bab09` | Medical Imaging, CT, windowing, dan pengukuran |
| `bab10` | Evaluasi Medical AI, threshold, sensitivity, specificity, dan calibration |

Tidak semua bagian konseptual dalam buku memerlukan praktik tersendiri. Karena itu, struktur repository berfokus pada eksperimen yang paling membantu pembaca memahami konsep melalui kode.

---

## Environment

Seluruh praktik dirancang menggunakan satu environment utama:

```text
cvai-book
```

Environment dapat dibuat menggunakan Conda:

```bash
conda env create -f environment.yml
conda activate cvai-book
```

Alternatifnya, dependency Python dapat dipasang menggunakan:

```bash
pip install -r requirements.txt
```

Versi library dapat berubah dari waktu ke waktu. Jika terjadi perbedaan perilaku pada versi yang lebih baru, gunakan konfigurasi pada `environment.yml` sebagai acuan utama.

---

## Menjalankan Praktik

Masuk ke folder bab yang ingin dicoba.

Contoh:

```bash
cd bab03
python run_all_bab03.py
```

Beberapa bab memiliki `run_all_babXX.py` untuk menjalankan rangkaian eksperimen secara berurutan.

Bab yang hanya memiliki satu praktik dapat dijalankan langsung dari folder `code`.

Contoh:

```bash
cd bab01
python code/praktik_bab01.py
```

Output yang dihasilkan program dapat berupa nilai numerik, file JSON, model, atau visualisasi. Sebagian output tidak disimpan di repository dan akan dibuat kembali ketika praktik dijalankan.

---

## Dataset dan Aset

Repository menggunakan kombinasi:

- citra milik penulis;
- citra atau dataset sintetis yang dibuat melalui kode;
- dataset publik dengan sumber yang terdokumentasi.

Beberapa praktik dapat mengunduh dataset publik ketika pertama kali dijalankan. Dataset berukuran besar tidak disertakan langsung di repository.

Informasi sumber dan status penggunaan aset tersedia pada:

`ASSET_LICENSES.md`

---

## Catatan Praktik

### CNN

Praktik CNN menggunakan dataset **EuroSAT** untuk memperkenalkan proses pembelajaran fitur visual dan klasifikasi citra.

Dataset tidak disimpan di repository dan dapat diunduh oleh script ketika praktik pertama kali dijalankan.

### Transfer Learning

Praktik Transfer Learning menggunakan **MobileNetV2** dan subset dari **Oxford-IIIT Pet**.

Bobot pretrained dan dataset dapat diunduh ketika praktik dijalankan untuk pertama kali.

### Object Detection

Praktik Object Detection menggunakan **Faster R-CNN (`torchvision`)** dan dataset sintetis yang dibuat secara programatik.

Pendekatan ini memungkinkan pembaca mempelajari konsep bounding box, prediksi objek, dan evaluasi deteksi tanpa harus menyimpan dataset berukuran besar di repository.

### Image Segmentation

Praktik Image Segmentation menggunakan **Mini U-Net** dan pasangan citra-mask sintetis.

Dataset dibuat secara deterministik melalui kode sehingga eksperimen dapat direproduksi.

### Medical Imaging dan Medical AI

Praktik pada bagian Medical Imaging menggunakan data sintetis dan tidak menggunakan data pasien nyata.

Contoh Medical AI ditujukan untuk mempelajari konsep seperti sensitivity, specificity, threshold, calibration, dan interpretasi hasil evaluasi model.

Contoh tersebut **bukan perangkat diagnosis dan bukan sistem untuk pengambilan keputusan klinis**.

---

## Reproducibility

Repository dirancang agar praktik utama dapat dijalankan kembali menggunakan:

**kode yang sama + data yang sama atau proses pembentukan data yang terdokumentasi + environment yang terdokumentasi.**

Untuk melakukan pemeriksaan struktur repository:

```bash
python verify_repository.py
```

Beberapa hasil numerik dapat sedikit berbeda akibat versi library, hardware, backend komputasi, atau operasi floating-point.

---

## Tentang Buku

**Computer Vision & AI: Dari Pixel hingga Analisis Citra Medis** mengajak pembaca memahami bagaimana komputer mulai dari membaca angka-angka pixel hingga mampu menemukan pola dan menghasilkan informasi dari sebuah citra.

Pembahasan bergerak dari konsep dasar menuju Machine Learning, Deep Learning, CNN, Transfer Learning, Object Detection, Image Segmentation, Vision Transformer, Medical Imaging, dan Medical AI.

Fokus buku bukan sekadar menjalankan model, tetapi memahami hubungan:

**masalah → data → representasi → metode → model → output → evaluasi → keterbatasan → penggunaan**

Dalam konteks Medical AI, perhatian juga diberikan pada validasi, generalisasi, bias, keselamatan, konteks klinis, dan peran manusia dalam penggunaan AI.

---

## Lisensi dan Atribusi

Kode, dataset, model pretrained, dan aset visual dapat memiliki ketentuan penggunaan yang berbeda.

Informasi sumber dan atribusi aset repository tersedia pada `ASSET_LICENSES.md`.

Kode sumber yang dibuat khusus untuk repository ini tersedia di bawah MIT License. Lihat file `LICENSE`.

Lisensi MIT tersebut tidak otomatis berlaku untuk foto, dataset pihak ketiga, pretrained weights, teks buku, ilustrasi, atau aset lainnya. Ketentuan sumber dan penggunaan aset dijelaskan dalam `ASSET_LICENSES.md`.

---

## Citation

Jika repository ini digunakan dalam kegiatan akademik, pembelajaran, atau penelitian, silakan merujuk pada buku:

**Sri Mulyono. _Computer Vision & AI: Dari Pixel hingga Analisis Citra Medis_.**

Informasi bibliografi lengkap akan diperbarui setelah buku diterbitkan.

---

## Disclaimer

Repository ini dibuat untuk tujuan **pendidikan dan pembelajaran**.

Contoh yang berkaitan dengan citra medis dan Medical AI tidak dimaksudkan sebagai alat diagnosis, rekomendasi terapi, atau pengganti pertimbangan tenaga kesehatan.