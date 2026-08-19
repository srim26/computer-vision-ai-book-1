# Lisensi dan Sumber Aset — Compact Edition

Dokumen ini hanya mencatat aset yang relevan dengan **Compact Edition 10 bab**.

## Aset milik penulis

| Aset | Penggunaan | Status |
|---|---|---|
| `kucing.jpg` | Praktik/ilustrasi citra | Foto milik penulis |
| `jeruk.jpg` | Praktik pixel, image processing, dan fitur | Foto milik penulis |
| Dataset sintetis | ML/detection/segmentation/medical imaging/medical AI sesuai bab | Dibuat programatik untuk praktik buku |

## Dataset/model pihak ketiga yang masih relevan

| Aset | Bab Compact | Catatan |
|---|---:|---|
| EuroSAT | 5 | Digunakan oleh praktik CNN yang berasal dari repository produksi; periksa kembali ketentuan sumber sebelum release final. |
| Oxford-IIIT Pet | 6 | Dataset target praktik transfer learning pada naskah Compact; dataset tidak perlu dikomit ke repository. |
| MobileNetV2 / Keras Applications | 6 | Pretrained ImageNet weights diunduh saat praktik; periksa dokumentasi/lisensi upstream pada release final. |
| Faster R-CNN / torchvision | 7 | Implementasi torchvision; praktik Compact menggunakan dataset sintetis dan tidak memakai pretrained COCO weights. |

## Tidak termasuk release aktif

Materi produksi lama seperti YOLO/COCO8, Fashion-MNIST, dan praktik Vision Transformer
pretrained terpisah tidak menjadi praktik aktif Compact Edition dan karena itu tidak
ditonjolkan pada README publik ini.

## Prinsip

- Jangan redistribusikan dataset pihak ketiga bila tidak diperlukan.
- Lebih aman menyediakan skrip/instruksi unduh dari sumber resmi.
- Lisensi kode repository tidak otomatis berlaku untuk foto, dataset, pretrained
  weights, teks buku, atau ilustrasi.
- Audit lisensi final tetap dilakukan sebelum repository diubah menjadi Public.
