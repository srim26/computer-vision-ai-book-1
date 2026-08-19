# Bab 9 — Medical Imaging

Praktik Compact Edition menggunakan **citra CT sintetis**, bukan data pasien dan
bukan file DICOM klinis. Tujuannya memperlihatkan intuisi nilai HU dan CT windowing.

Jalankan:

```bash
python run_all_bab09.py
```

Output:
- `output/output_09_01_ct_sintetis.png`
- `output/output_09_02_windowing_ct.png`

Bagian measurement menggunakan contoh numerik marker 2 × 2 cm, 6.400 pixel marker,
dan 7.068 pixel mask. Nilai 4,4175 cm² adalah hasil contoh perhitungan terkontrol,
bukan hasil segmentasi pasien.
