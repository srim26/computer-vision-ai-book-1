# Bab 10 — Medical AI

Praktik Compact Edition menggunakan **data sintetis deterministik** untuk
mendemonstrasikan:

- trade-off sensitivity dan specificity terhadap threshold;
- confusion matrix/metrics pada threshold 0,50;
- calibration curve;
- Brier score.

Tidak ada angka klinis yang diklaim. Output pada folder `output/` berasal langsung
dari script yang sama.

Jalankan:

```bash
python run_all_bab10.py
```

Output utama:
- `output/metrics_bab10.json`
- `output/output_10_01_threshold_tradeoff.png`
- `output/output_10_02_calibration_curve.png`

Semua hasil bersifat pedagogis dan **bukan validasi klinis**.
