# Risk-Aware Regional Tax Forecasting Using Prophet and Conformal Prediction
Repositori resmi penelitian hibah internal UKI Toraja (2026).

## Pipeline
Train (2020–2023) → Kalibrasi Split Conformal (residual 2024) → 
Interval 90% → Agregasi Monte Carlo (B=10.000) → Skenario fiskal 2026 (P10/P50/P90).

## Struktur Repositori
(lihat tree di atas)

## Instalasi & Reproduksi
pip install -r requirements.txt
python src/01_data_preparation.py  (urutan 01 → 05)

## Catatan Data
Data riil BPKPD Tana Toraja bersifat confidential. Repo menyediakan data dummy
ber-skema identik untuk reproduksibilitas. Permintaan data riil untuk keperluan
akademik dapat diajukan ke penulis/BPKPD.

## Hasil Utama
MASE 0,649 (vs SARIMA 0,908; SNaive 0,859); cakupan empiris 91,7% (nominal 90%).

## Sitasi
(lihat CITATION.cff / BibTeX paper)

## Lisensi & Kontak
MIT — Muhammad Sofwan Adha (sovant.muhammad@gmail.com)
