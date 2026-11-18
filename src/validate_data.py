#!/usr/bin/env python3
"""
validate_data.py — Checklist validasi data (§8 PROJECT_CONTEXT.md + Langkah 1).

Hard-fail (STOP + lapor, exit code 1) jika:
  1. Shape != (72, 9) ATAU nama kolom != §2.
  2. Ada NaN atau duplikat tanggal.
  3. Rentang tanggal != 2020-01 s.d. 2025-12 bulanan (freq MS).
  4. peak_season_flag per tahun != tepat {Jan, Nov, Des} (sum = 3).
  5. rambu_solo_flag sum per tahun != 6  -> indikasi versi LAMA.
  6. holiday_flag[2023-02] != 1.
  7. Pola rambu_solo_flag per tahun != seragam {Jan,Mei,Jun,Jul,Nov,Des}
     = {1,5,6,7,11,12}.  (Catatan: assertion "anomali 2023" pada draft awal
     DIRELAKS oleh peneliti 2026-10-13 — lihat ASSUMPTIONS.md A-001. File final
     menyeragamkan 2023 ke pola standar; tidak ada lagi anomali yang diwajibkan.)

Report-only (cetak & catat, tidak menggagalkan):
  - Jumlah nol pada pajak_hotel (ekspektasi peneliti: 9).
  - Range tingkat_hunian_hotel (harus persen ~0-40) & jumlah_wisatawan >= 0.
  - Tabel flag penuh -> outputs/tables/validation_flags.csv

Deterministik; seed global 42 di-set untuk kepatuhan protokol (tidak ada RNG).
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

# --- Langkah 0.4: seed global 42 di setiap entry point -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Referensi protokol (§2) -------------------------------------------------
EXPECTED_COLUMNS = [
    "date", "year", "month", "jumlah_wisatawan", "tingkat_hunian_hotel",
    "pajak_hotel", "holiday_flag", "rambu_solo_flag", "peak_season_flag",
]
EXPECTED_SHAPE = (72, 9)
START, END = "2020-01-01", "2025-12-01"
PEAK_MONTHS = {1, 11, 12}                 # {Jan, Nov, Des}
RAMBU_SOLO_PER_YEAR = 6                   # versi FINAL
EXPECTED_ZEROS_PAJAK = 9                  # report-only (ekspektasi peneliti)

DEFAULT_INPUT = os.path.join("data", "master_dataset.xlsx")
FALLBACK_INPUT = "master_dataset.xlsx"    # lokasi aktual saat ini (root)
FLAGS_OUT = os.path.join("outputs", "tables", "validation_flags.csv")


def _resolve_input(cli_input: str | None) -> str:
    """Pilih path master secara transparan (tidak memilih versi data)."""
    if cli_input:
        return cli_input
    if os.path.exists(DEFAULT_INPUT):
        return DEFAULT_INPUT
    if os.path.exists(FALLBACK_INPUT):
        print(f"[WARN] '{DEFAULT_INPUT}' tidak ditemukan; memakai '{FALLBACK_INPUT}' "
              f"di root (lihat ASSUMPTIONS.md A-003).")
        return FALLBACK_INPUT
    raise FileNotFoundError(
        f"Master tidak ditemukan di '{DEFAULT_INPUT}' maupun '{FALLBACK_INPUT}'.")


def _cell(df: pd.DataFrame, year: int, month: int, col: str):
    row = df[(df["year"] == year) & (df["month"] == month)]
    if row.empty:
        return None
    return row[col].iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validasi master_dataset.xlsx (§8).")
    ap.add_argument("--input", default=None,
                    help=f"Path master (default: {DEFAULT_INPUT}, fallback: {FALLBACK_INPUT}).")
    args = ap.parse_args()

    print("=" * 72)
    print("VALIDASI DATA — pajak_hotel Tana Toraja")
    print("=" * 72)
    print(f"python   : {sys.version.split()[0]}")
    print(f"pandas   : {pd.__version__}")
    print(f"numpy    : {np.__version__}")
    import openpyxl
    print(f"openpyxl : {openpyxl.__version__}")
    print(f"seed     : {SEED}")

    path = _resolve_input(args.input)
    print(f"input    : {path}")

    xl = pd.ExcelFile(path)
    print(f"sheets   : {xl.sheet_names}")
    df = pd.read_excel(path, sheet_name=0)

    hard: list[str] = []      # pelanggaran hard-fail
    notes: list[str] = []     # report-only

    # --- 1. Shape & kolom ---------------------------------------------------
    if tuple(df.shape) != EXPECTED_SHAPE:
        hard.append(f"Shape {tuple(df.shape)} != {EXPECTED_SHAPE}.")
    if list(df.columns) != EXPECTED_COLUMNS:
        hard.append(f"Kolom tidak cocok §2.\n     didapat: {list(df.columns)}\n     harap  : {EXPECTED_COLUMNS}")

    # --- parse date ---------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- 2. NaN & duplikat tanggal -----------------------------------------
    nan_total = int(df.isna().sum().sum())
    if nan_total > 0:
        per_col = df.isna().sum()
        hard.append(f"Ditemukan {nan_total} NaN. Per kolom:\n{per_col[per_col>0].to_string()}")
    dup = int(df["date"].duplicated().sum())
    if dup > 0:
        hard.append(f"Ditemukan {dup} tanggal duplikat.")

    # --- 3. Rentang & frekuensi MS -----------------------------------------
    expected_idx = pd.date_range(START, END, freq="MS")
    dates_sorted = df["date"].dropna().sort_values().reset_index(drop=True)
    if len(dates_sorted) != len(expected_idx):
        hard.append(f"Jumlah tanggal {len(dates_sorted)} != {len(expected_idx)} (2020-01..2025-12, freq MS).")
    elif not dates_sorted.equals(pd.Series(expected_idx)):
        mismatch = [(str(a.date()), str(b.date())) for a, b in zip(dates_sorted, expected_idx) if a != b]
        hard.append(f"Deret tanggal tidak cocok freq MS. Contoh selisih (dapat,hapus) : {mismatch[:5]}")
    else:
        print(f"[OK]   Rentang {START}..{END} bulanan (freq MS), {len(expected_idx)} baris.")

    # --- 4. peak_season_flag per tahun == {Jan,Nov,Des}, sum 3 -------------
    if "peak_season_flag" in df and "year" in df:
        peak_bad = []
        for yr, g in df.groupby("year"):
            months = set(int(m) for m in g.loc[g["peak_season_flag"] == 1, "month"])
            s = int(g["peak_season_flag"].sum())
            if months != PEAK_MONTHS or s != 3:
                peak_bad.append(f"peak_season_flag {yr}: months={sorted(months)} sum={s} "
                                f"(harus {{1,11,12}} sum=3).")
        if peak_bad:
            hard.extend(peak_bad)
        else:
            print("[OK]   peak_season_flag == {Jan,Nov,Des} (sum=3) setiap tahun.")

    # --- 5. rambu_solo_flag sum per tahun == 6 -----------------------------
    if "rambu_solo_flag" in df and "year" in df:
        sums = df.groupby("year")["rambu_solo_flag"].sum()
        bad = {int(y): int(v) for y, v in sums.items() if int(v) != RAMBU_SOLO_PER_YEAR}
        if bad:
            hard.append(f"rambu_solo_flag sum/tahun != 6 pada {bad} -> indikasi versi LAMA. STOP.")
        else:
            print(f"[OK]   rambu_solo_flag sum/tahun == 6 (versi FINAL): {dict(sums.astype(int))}")

    # --- 6. holiday_flag[2023-02] == 1 -------------------------------------
    hf = _cell(df, 2023, 2, "holiday_flag") if "holiday_flag" in df else None
    if hf is None:
        hard.append("Baris 2023-02 tidak ditemukan untuk cek holiday_flag.")
    elif int(hf) != 1:
        hard.append(f"holiday_flag[2023-02] = {int(hf)} != 1.")
    else:
        print("[OK]   holiday_flag[2023-02] == 1.")

    # --- 7. Pola rambu_solo SERAGAM (anomali 2023 DIRELAKS per peneliti) ----
    # Keputusan peneliti 2026-10-13 (ASSUMPTIONS.md A-001): anomali 2023
    # (rambu_solo[2023-05]=0, [2023-08]=1) hanyalah observasi draft awal, bukan
    # syarat mutlak. File final menyeragamkan pola menjadi {1,5,6,7,11,12} untuk
    # SEMUA tahun (termasuk 2023). Assertion lama DIRELAKS -> pola seragam = PASS.
    RAMBU_PATTERN = {1, 5, 6, 7, 11, 12}
    if "rambu_solo_flag" in df and "year" in df:
        rambu_bad = []
        for yr, g in df.groupby("year"):
            months = set(int(m) for m in g.loc[g["rambu_solo_flag"] == 1, "month"])
            if months != RAMBU_PATTERN:
                rambu_bad.append(f"rambu_solo_flag {yr}: months={sorted(months)} "
                                 f"!= pola seragam {sorted(RAMBU_PATTERN)}.")
        if rambu_bad:
            hard.extend(rambu_bad)
        else:
            print("[OK]   rambu_solo_flag seragam {1,5,6,7,11,12} semua tahun "
                  "(anomali 2023 DIRELAKS per peneliti, A-001).")

    # === REPORT-ONLY =======================================================
    print("-" * 72)
    print("REPORT-ONLY (tidak menggagalkan):")

    # nol pada pajak_hotel
    if "pajak_hotel" in df:
        z = df[df["pajak_hotel"] == 0]
        notes.append(f"Jumlah nol pajak_hotel = {len(z)} (ekspektasi peneliti {EXPECTED_ZEROS_PAJAK}).")
        if len(z) != EXPECTED_ZEROS_PAJAK:
            notes.append("  [WARN] jumlah nol != ekspektasi 9 — catat, jangan imputasi.")
        if not z.empty:
            zlist = ", ".join(f"{int(r.year)}-{int(r.month):02d}" for r in z.itertuples())
            notes.append(f"  Bulan nol: {zlist}")

    # range hunian & wisatawan
    if "tingkat_hunian_hotel" in df:
        lo, hi = float(df["tingkat_hunian_hotel"].min()), float(df["tingkat_hunian_hotel"].max())
        notes.append(f"tingkat_hunian_hotel min/max = {lo:.2f} / {hi:.2f}.")
        if hi <= 1.0:
            notes.append("  [WARN] max <= 1 -> tampak seperti RASIO, bukan persen. Protokol: persen.")
        elif hi > 100.0:
            notes.append("  [WARN] max > 100 -> di luar skala persen.")
        else:
            notes.append("  [OK] skala persen plausibel (~0-40 sesuai §2).")
    if "jumlah_wisatawan" in df:
        wmin = int(df["jumlah_wisatawan"].min())
        nz = int((df["jumlah_wisatawan"] == 0).sum())
        notes.append(f"jumlah_wisatawan min = {wmin} (harus >= 0); jumlah bulan bernilai 0 = {nz}.")
        if wmin < 0:
            notes.append("  [WARN] ada jumlah_wisatawan negatif.")

    for n in notes:
        print("  " + n)

    # === Tabel flag penuh -> CSV ===========================================
    os.makedirs(os.path.dirname(FLAGS_OUT), exist_ok=True)
    flag_cols = ["date", "year", "month", "holiday_flag", "rambu_solo_flag", "peak_season_flag"]
    flag_cols = [c for c in flag_cols if c in df.columns]
    flags_df = df[flag_cols].copy()
    flags_df["date"] = flags_df["date"].dt.strftime("%Y-%m-%d")
    flags_df.to_csv(FLAGS_OUT, index=False)
    print("-" * 72)
    print(f"Tabel flag penuh disimpan -> {FLAGS_OUT} ({len(flags_df)} baris)")
    print("\nTabel flag (year x month), format H/R/P = holiday/rambu/peak:")
    piv = df.copy()
    piv["date"] = piv["date"].dt.strftime("%Y-%m")
    print(piv[["year", "month", "holiday_flag", "rambu_solo_flag", "peak_season_flag"]]
          .to_string(index=False))

    # === RINGKASAN =========================================================
    print("=" * 72)
    if hard:
        print(f"HASIL: FAIL — {len(hard)} pelanggaran HARD-FAIL:")
        for i, h in enumerate(hard, 1):
            print(f"  [{i}] {h}")
        print("=" * 72)
        print(">>> STOP. JANGAN lanjut ke modeling (Langkah 2). Lapor peneliti. <<<")
        return 1
    print("HASIL: PASS — semua hard-assertion terpenuhi.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
