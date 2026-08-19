# Bab 7 — Object Detection dengan Faster R-CNN

Praktik Compact Edition menggunakan dataset sintetis programatik:
80 citra training dan 20 citra validation, dua kelas objek (persegi dan lingkaran).

Detector menggunakan `fasterrcnn_mobilenet_v3_large_fpn` dari torchvision.
**COCO detector weights tidak digunakan.** Backbone MobileNetV3 menggunakan
pretrained ImageNet weights agar feature extractor memiliki representasi awal yang
lebih stabil pada praktik kecil ini.

## Clean-run

```bash
python run_all_bab07.py
```

Training menggunakan CPU dan 10 epoch pada konfigurasi RC1.4.
Output utama:
- `output/metrics_bab07.json`
- `output/fasterrcnn_bab07.pt`
- `output/output_07_03_setelah_training.png`

Praktik ini adalah sanity check pedagogis, bukan benchmark object detection dunia nyata.
