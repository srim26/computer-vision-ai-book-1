# Sumber dan Lisensi Aset

Dokumen ini mencatat sumber, kepemilikan, dan ketentuan penggunaan aset yang digunakan dalam repository pendamping buku **Computer Vision & AI: Dari Pixel hingga Analisis Citra Medis**.

## Aset Milik Penulis

| Aset | Penggunaan | Status |
|---|---|---|
| `kucing.jpg` | Praktik membaca citra dan representasi pixel | Foto milik penulis |
| `jeruk.jpg` | Praktik pixel, image processing, dan ekstraksi fitur | Foto milik penulis |
| Dataset sintetis | Praktik Machine Learning, object detection, image segmentation, Medical Imaging, dan Medical AI | Dibuat secara programatik melalui kode repository |

Aset milik penulis tetap merupakan bagian dari karya buku dan tidak otomatis mengikuti lisensi kode repository.

## Dataset dan Model Pihak Ketiga

### EuroSAT

Digunakan pada praktik CNN di Bab 5.

Sumber utama:
- Patrick Helber, Benjamin Bischke, Andreas Dengel, dan Damian Borth.
- Dataset EuroSAT berbasis citra Sentinel-2.
- Dataset tersedia melalui repository resmi EuroSAT.

Lisensi dataset: **MIT License**.

Dataset tidak disimpan langsung dalam repository ini. Script praktik dapat mengunduh dataset dari sumber resminya.

### Oxford-IIIT Pet

Digunakan pada praktik Transfer Learning di Bab 6.

Repository menggunakan subset dua kelas:
- `Abyssinian`
- `american_bulldog`

Lisensi dataset: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

Dataset tidak didistribusikan ulang melalui repository ini. Script hanya membantu pengguna memperoleh data dari sumber resminya.

### MobileNetV2 dan Keras Applications

Digunakan sebagai backbone pada praktik Transfer Learning Bab 6.

Implementasi Keras menggunakan lisensi **Apache License 2.0**.

Pretrained weights dapat diunduh otomatis ketika model dibuat. Pengguna tetap perlu memperhatikan ketentuan penggunaan dataset asal yang digunakan untuk melatih bobot tersebut.

### Faster R-CNN / torchvision

Digunakan pada praktik Object Detection Bab 7.

Implementasi berasal dari `torchvision`, yang menggunakan lisensi **BSD-3-Clause**.

Praktik dalam repository menggunakan dataset sintetis yang dibuat secara programatik. Detector tidak menggunakan pretrained COCO detector weights. Backbone MobileNetV3 menggunakan pretrained weights yang disediakan melalui torchvision.

## Dataset Sintetis

Beberapa praktik menggunakan dataset yang dibentuk secara programatik, antara lain:

- dataset dua kelas pada Bab 4;
- objek geometris pada Bab 7;
- pasangan image-mask pada Bab 8;
- CT sintetis pada Bab 9;
- data probabilitas sintetis pada Bab 10.

Dataset tersebut dibuat hanya untuk tujuan pembelajaran dan reproducibility.

Data sintetis pada Bab 9 dan Bab 10 bukan data pasien dan bukan data klinis.

## Prinsip Penggunaan

- Dataset pihak ketiga tidak didistribusikan ulang jika tidak diperlukan.
- Repository lebih mengutamakan script untuk memperoleh data dari sumber resminya.
- Lisensi kode repository tidak otomatis berlaku pada foto, dataset, pretrained weights, teks buku, atau aset visual lain.
- Pengguna bertanggung jawab mematuhi ketentuan sumber upstream ketika menggunakan dataset atau pretrained weights pihak ketiga.