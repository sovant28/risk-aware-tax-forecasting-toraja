"""cv_fase1.py — Rolling-origin CV dalam TRAIN untuk sanity-check growth Prophet.

Protokol peneliti (resolusi A-010):
  Fold 1: train 2020-01..2021-12 -> horizon 12 -> eval 2022.
  Fold 2: train 2020-01..2022-12 -> horizon 12 -> eval 2023.
  growth ∈ {linear, flat}; HP lain TERKUNCI; 4 regressor TERKUNCI; RUANG ASLI.
  MAE & MASE per fold (scale MASE = MAE seasonal-naive m=12 dalam train fold itu).

ATURAN KEPUTUSAN (peneliti):
  - flat >= linear (flat MASE <= linear MASE) -> freeze 'flat'.
  - linear menang TELAK -> STOP & lapor (jangan auto-pindah).
CV ini TIDAK menyentuh Test 2025 (hindari test-set tuning). Hasil = baris ablation paper.

Jalankan dari root repo:  .venv/bin/python src/cv_fase1.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import load_full, get_regressors          # noqa: E402
from models import fit_prophet, predict_prophet         # noqa: E402

np.random.seed(42)

_caught = []


def _warn_show(message, category, filename, lineno, file=None, line=None):
    _caught.append(f"{category.__name__}: {message}")


warnings.showwarning = _warn_show
warnings.simplefilter("always")

import logging  # noqa: E402
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tables", "cv_growth_fase1.csv")


def mase_scale(y_train):
    y = np.asarray(y_train, dtype=float)
    return float(np.mean(np.abs(y[12:] - y[:-12])))


def run_fold(df, train_max_year, eval_year, growth, regs):
    tr = df[df["year"] <= train_max_year].copy()
    ev = df[df["year"] == eval_year].copy()
    m = fit_prophet(tr, regs, growth=growth)
    pred = predict_prophet(m, ev, regs)
    y_true = ev["pajak_hotel"].values.astype(float)
    mae = float(np.mean(np.abs(y_true - pred)))
    scale = mase_scale(tr["pajak_hotel"].values)
    mase = mae / scale if scale > 1e-8 else np.nan
    return {
        "fold": f"train<={train_max_year}->eval{eval_year}",
        "growth": growth,
        "n_train": len(tr),
        "n_eval": len(ev),
        "MAE": mae,
        "MASE_scale": scale,
        "MASE": mase,
        "pred_max": float(pred.max()),
    }


def main():
    df = load_full()
    regs = get_regressors()
    folds = [(2021, 2022), (2022, 2023)]
    rows = []
    for tmy, ey in folds:
        for g in ("linear", "flat"):
            rows.append(run_fold(df, tmy, ey, g, regs))
    cv = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv.to_csv(OUT, index=False)

    # Ringkasan per growth (rata-rata MASE antar fold)
    summary = cv.groupby("growth")["MASE"].mean().rename("MASE_mean").reset_index()
    summary = summary.sort_values("MASE_mean")

    print("=== CV rolling-origin (dalam Train, ruang asli) ===")
    try:
        print(cv.to_markdown(index=False))
    except Exception:
        print(cv.to_string(index=False))
    print("\n=== Ringkasan MASE rata-rata per growth ===")
    try:
        print(summary.to_markdown(index=False))
    except Exception:
        print(summary.to_string(index=False))

    flat = float(summary.loc[summary["growth"] == "flat", "MASE_mean"].iloc[0])
    lin = float(summary.loc[summary["growth"] == "linear", "MASE_mean"].iloc[0])
    print(f"\nflat MASE_mean={flat:.3f} | linear MASE_mean={lin:.3f}")
    if flat <= lin:
        print("KEPUTUSAN: flat >= linear (MASE flat <= linear) -> FREEZE growth='flat'.")
    else:
        ratio = lin / flat if flat > 0 else 0
        print(f"KEPUTUSAN: linear lebih baik (ratio linear/flat={ratio:.3f}).")
        if ratio < 0.75:
            print("  -> linear menang TELAK. STOP & lapor peneliti (jangan auto-pindah).")
        else:
            print("  -> linear unggul tipis. Default a-priori 'flat' DIPERTAHANKAN; laporkan ke peneliti.")

    print(f"\nSaved: {OUT}")
    print("\n=== WARNINGS TERTANGKAP (dedup by category) ===")
    seen = []
    for w in _caught:
        k = w.split(":")[0]
        if k not in seen:
            seen.append(k)
            print("-", w[:180])
    if not _caught:
        print("(tidak ada)")
    return cv, summary


if __name__ == "__main__":
    main()
