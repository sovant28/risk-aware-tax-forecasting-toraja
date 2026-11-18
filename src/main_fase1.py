"""main_fase1.py — Orchestrator Fase 1 (Replikasi Model Dasar, RUANG ASLI / A-010).

Jalankan dari root repo (SETELAH konfigurasi di-freeze):
    .venv/bin/python src/main_fase1.py

Menghasilkan:
  - outputs/tables/table1_point_metrics.csv  (Prophet-BO, SARIMA, SeasonalNaive; Test 2025)
  - outputs/intermediate/preds_fase1.csv      (prediksi Calib 2024 & Test 2025 semua model;
                                               dibutuhkan Fase 2 untuk kalibrasi Conformal)
Determinisme: seed 42; 2 run harus identik.

GUARD: `FROZEN` harus True sebelum Test 2025 disentuh (aturan A-011: Test dievaluasi
tepat satu kali setelah konfigurasi difreeze; pemilihan growth TIDAK pakai data test).
Rolling-origin CV (src/cv_fase1.py) memicu aturan STOP peneliti: linear unggul atas
flat (MASE_mean 2.390 vs 3.371, ratio 0.709). Karena itu growth BELUM difreeze;
script menolak jalan sampai peneliti memutuskan (set GROWTH & FROZEN=True).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

# Pastikan flat-import (features/models/evaluate) jalan dari cwd mana pun.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import load_and_split, get_regressors          # noqa: E402
from models import run_prophet, run_sarima, run_snaive        # noqa: E402
from evaluate import calc_metrics, save_table1                # noqa: E402

# --- Konfigurasi growth Prophet (A-010/A-011) ---
GROWTH = "flat"    # FREEZE (keputusan tertulis peneliti 2026-10-13): konservatisme
                   # fiskal + Fold 2 CV (plateau) flat unggul telak. Lihat A-011.
FROZEN = True      # Konfigurasi difreeze -> Test 2025 boleh dievaluasi TEPAT SATU KALI.

np.random.seed(42)  # seed global (§4)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS_OUT = os.path.join(ROOT, "outputs", "intermediate", "preds_fase1.csv")

# Rekam warning konvergensi (Prophet/SARIMA) untuk checkpoint, tanpa spam log.
_caught = []


def _warn_show(message, category, filename, lineno, file=None, line=None):
    _caught.append(f"{category.__name__}: {message}")


warnings.showwarning = _warn_show
warnings.simplefilter("always")

import logging  # noqa: E402
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def main():
    if not FROZEN:
        print("STOP: konfigurasi growth BELUM difreeze (FROZEN=False).")
        print("CV (src/cv_fase1.py) memicu aturan STOP peneliti: linear MASE_mean 2.390 "
              "< flat 3.371 (ratio 0.709). Test 2025 tidak boleh disentuh sampai "
              "peneliti memutuskan growth (A-011).")
        print("Set GROWTH dan FROZEN=True di src/main_fase1.py setelah keputusan, lalu jalankan ulang.")
        sys.exit(2)

    train, calib, test = load_and_split()
    regs = get_regressors()

    y_train_orig = train["pajak_hotel"].values.astype(float)
    y_calib_orig = calib["pajak_hotel"].values.astype(float)
    y_test_orig = test["pajak_hotel"].values.astype(float)

    # 1. Prophet — RUANG ASLI (A-010), growth=GROWTH. Dict: yhat/lower/upper (native CI 90%).
    p_calib, p_test = run_prophet(train, calib, test, regs, growth=GROWTH)

    # 2. SARIMA — RUANG ASLI (A-010), order terkunci (1,1,1)(1,1,1)12.
    s_calib, s_test = run_sarima(y_train_orig, len(calib), len(test))

    # 3. SeasonalNaive — ruang asli, sp=12 (tidak berubah).
    sn_calib, sn_test = run_snaive(y_train_orig, len(calib), len(test))

    # Metrik titik pada Test 2025 (evaluasi utama, §3). Point prediction = p_test['yhat'].
    results = {
        "Prophet-BO": calc_metrics(y_test_orig, p_test["yhat"], y_train_orig),
        "SARIMA": calc_metrics(y_test_orig, s_test, y_train_orig),
        "SeasonalNaive": calc_metrics(y_test_orig, sn_test, y_train_orig),
    }
    df = save_table1(results)

    # Simpan prediksi Calib & Test + native CI Prophet (Fase 2: kalibrasi Conformal;
    # kolom yhat_lower_native/upper_native dibutuhkan C3 & M0 per Q2 peneliti).
    os.makedirs(os.path.dirname(PREDS_OUT), exist_ok=True)
    pred_rows = []
    for split, dset, ytrue, pc, ps, pn in [
        ("calib", calib, y_calib_orig, p_calib, s_calib, sn_calib),
        ("test", test, y_test_orig, p_test, s_test, sn_test),
    ]:
        for i, (_, row) in enumerate(dset.iterrows()):
            pred_rows.append({
                "date": row["date"], "split": split, "actual": float(ytrue[i]),
                "pred_prophet": float(pc["yhat"][i]),
                "pred_sarima": float(ps[i]),
                "pred_snaive": float(pn[i]),
                "yhat_lower_native": float(pc["lower"][i]),
                "yhat_upper_native": float(pc["upper"][i]),
            })
    preds = pd.DataFrame(pred_rows)
    preds.to_csv(PREDS_OUT, index=False)
    print(f"Saved predictions: {PREDS_OUT}  (n={len(preds)}: calib+test, +native CI)")

    # --- Ringkasan konvergensi untuk checkpoint ---
    print("\n=== WARNINGS TERTANGKAP ===")
    if not _caught:
        print("(tidak ada warning)")
    else:
        seen = []
        for w in _caught:
            key = w.split(":")[0]
            if key not in seen:
                seen.append(key)
                print("-", w[:200])
    return df


if __name__ == "__main__":
    main()
