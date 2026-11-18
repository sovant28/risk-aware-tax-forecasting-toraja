"""features.py — Data loading & splitting (Langkah 2 / Fase 1).

Mengikuti PROJECT_CONTEXT.md §3 (split TERKUNCI) dan §4 (regressor model utama).
`peak_season_flag` DILARANG masuk regressor.

DEVIASI A-009 (disetujui peneliti): `StandardScaler` eksternal DIHAPUS. Prophet
men-standardize regressor kontinu secara INTERNAL via `add_regressor(standardize='auto')`
(holiday_flag & rambu_solo_flag binary -> tidak distandardize), memakai statistik
Train; `prior_scale=2.98` berlaku pada koefisien terstandardize internal itu.

DEVIASI A-010 (disetujui peneliti): `log1p` DIHAPUS dari pipeline utama. Prophet
& SARIMA kini dimodelkan di RUANG ASLI (pajak_hotel, Rupiah). Alasan: log1p pada
serie dengan 9 nol eksak pandemi menciptakan ramp artifisial 0->18.6 yang, ketika
tren linear diekstrapolasi 24 bulan lalu di-invers expm1, meledak ke ~1e12
(bukti: dekomposisi tren ruang-log overshoot 6-9 unit). Kolom `y_log` tetap
disediakan HANYA untuk ablation terdokumentasi (SARIMA-log1p, MASE 206), bukan
untuk pipeline utama.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "data", "master_dataset.xlsx")


def load_full(path=DEFAULT_DATA):
    """Muat master dataset, urutkan kronologis. Ruang asli (pajak_hotel).

    Menambahkan kolom bantu: y_log (log1p, HANYA untuk ablation terdokumentasi).
    """
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["y_log"] = np.log1p(df["pajak_hotel"])
    return df


def load_and_split(path=DEFAULT_DATA):
    """Split TERKUNCI (§3): Train 2020-2023, Calib 2024, Test 2025."""
    df = load_full(path)
    train = df[df["year"] <= 2023].copy()   # 48 baris
    calib = df[df["year"] == 2024].copy()   # 12 baris
    test = df[df["year"] == 2025].copy()    # 12 baris
    return train, calib, test


def get_regressors():
    # peak_season_flag DILARANG (§4). 4 regressor model utama.
    return ["jumlah_wisatawan", "tingkat_hunian_hotel", "holiday_flag", "rambu_solo_flag"]
