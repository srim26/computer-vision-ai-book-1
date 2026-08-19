# Bab 6 — Transfer Learning dan Vision Transformer

Praktik aktif Compact Edition berfokus pada **Transfer Learning dengan MobileNetV2**
menggunakan subset Oxford-IIIT Pet: `Abyssinian` dan `american_bulldog`, masing-masing
200 citra. Split deterministik per kelas: 140 train, 30 validation, 30 test.

Vision Transformer dibahas secara konseptual dalam naskah Compact dan tidak
memerlukan model pretrained tambahan untuk clean-run Bab 6.

## Menjalankan

```bash
python run_all_bab06.py
```

Jika download otomatis gagal karena SSL/jaringan, unduh `images.tar.gz` dari halaman
resmi Oxford-IIIT Pet dan letakkan di `bab06/data/images.tar.gz`, lalu jalankan ulang.

Output utama berasal dari training nyata, bukan angka ilustratif.
