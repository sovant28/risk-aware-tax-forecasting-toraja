"""main_fase5.py — Fase 5: Conditional Forecasting 2026 (simulasi KUA-PPAS).

Memakai MODEL BEKU Prophet (fit Train 2020-2023, growth='flat', HP terkunci) dari
pipeline Fase 1/2. Model TIDAK di-fit ulang pada data baru: direproduksi deterministik
dari Train-only lalu DIVERIFIKASI identik dengan preds_fase1.csv (calib 2024 & test 2025).

Per skenario (pesimis/dasar/optimis) dari data/asumsi_sektoral_2026.csv:
  - P50 = prediksi titik Prophet (ruang asli) pada regressor skenario.
  - P10/P90 = interval Split Conformal C1 (alpha=0.10): q = kuantil conformal |residual
    Calib 2024| (finite-sample, n=12 -> jenuh = max|resid|); P10=clip(P50-q,0), P90=P50+q.
  - Agregasi tahunan Monte Carlo (B=10000, seed 42): sampling residual calib WITH
    REPLACEMENT per bulan (independen antar-bulan = limitasi, A-014), bulan=clip(P50+r,0),
    jumlah 12 bulan -> persentil 10/50/90 total tahunan. (seperti Fase 3)
  - peak_season_flag DILARANG sebagai regressor (§4); hanya 4 regressor terkunci.

Output: table5_proyeksi_2026.csv, table6_ringkasan_proyeksi_2026.csv, fig5_kua_ppas_2026.png.
Jalankan dari root repo:  .venv/bin/python src/main_fase5.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features import load_full, get_regressors      # noqa: E402
from models import fit_prophet, predict_prophet      # noqa: E402
from conformal import conformal_quantile             # noqa: E402

np.random.seed(42)
warnings.filterwarnings("ignore")
import logging  # noqa: E402
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = os.path.join(ROOT, "data", "asumsi_sektoral_2026.csv")
PREDS1 = os.path.join(ROOT, "outputs", "intermediate", "preds_fase1.csv")
TABLE5 = os.path.join(ROOT, "outputs", "tables", "table5_proyeksi_2026.csv")
TABLE6 = os.path.join(ROOT, "outputs", "tables", "table6_ringkasan_proyeksi_2026.csv")
FIGDIR = os.path.join(ROOT, "outputs", "figures")
FIG5 = os.path.join(FIGDIR, "fig5_kua_ppas_2026.png")

ALPHA = 0.10
B = 10000
SCENARIOS = ["pesimis", "dasar", "optimis"]


def load_frozen_model_and_verify(df, regs):
    """Reproduksi model beku (Train-only, growth=flat) & VERIFIKASI == preds_fase1."""
    train = df[df["year"] <= 2023].copy()
    m = fit_prophet(train, regs, growth="flat", interval_width=0.90)

    calib = df[df["year"] == 2024].copy()
    test = df[df["year"] == 2025].copy()
    calib_pred = predict_prophet(m, calib, regs)
    test_pred = predict_prophet(m, test, regs)

    p1 = pd.read_csv(PREDS1)
    p1["date"] = pd.to_datetime(p1["date"])
    ref_calib = p1[p1["split"] == "calib"].sort_values("date")["pred_prophet"].values
    ref_test = p1[p1["split"] == "test"].sort_values("date")["pred_prophet"].values

    ok_c = np.allclose(calib_pred, ref_calib, rtol=1e-9, atol=1e-3)
    ok_t = np.allclose(test_pred, ref_test, rtol=1e-9, atol=1e-3)
    print("=== VERIFIKASI model beku (bukan refit pada data baru) ===")
    print(f"calib 2024 reproduksi == preds_fase1 : {ok_c} (max diff "
          f"{np.max(np.abs(calib_pred - ref_calib)):.3e})")
    print(f"test  2025 reproduksi == preds_fase1 : {ok_t} (max diff "
          f"{np.max(np.abs(test_pred - ref_test)):.3e})")
    if not (ok_c and ok_t):
        raise RuntimeError("Model reproduksi TIDAK identik dengan Fase 1/2 -> STOP (A-019).")
    print("[OK] Model beku terverifikasi IDENTIK dengan pipeline Fase 1/2.")

    # residual calib 2024 SIGNED dari preds beku (sumber conformal C1)
    calib_resid = (p1[p1["split"] == "calib"].sort_values("date")["actual"].values
                   - ref_calib)
    return m, calib_resid


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = load_full()
    regs = get_regressors()

    model, calib_resid = load_frozen_model_and_verify(df, regs)
    abs_resid = np.abs(calib_resid)
    q, k, n = conformal_quantile(abs_resid, ALPHA)
    print(f"\n=== Kuantil conformal C1 (|resid Calib 2024|, n={n}) ===")
    print(f"alpha={ALPHA} -> k={k}, q={q:.4e} (=max|resid|, jenuh; A-013)")

    # Muat skenario 2026
    sc = pd.read_csv(SCEN)
    sc["date"] = pd.to_datetime(sc["date"])
    # cek holiday_flag Juni 2026 (catatan peneliti: harus 1)
    june = sc[sc["month"] == 6]
    print("\n=== Cek holiday_flag Juni 2026 (harus 1: Pancasila + Tahun Baru Islam) ===")
    print(june[["scenario", "date", "holiday_flag", "rambu_solo_flag"]].to_string(index=False))
    june_ok = bool((june["holiday_flag"] == 1).all())
    print(f"Juni holiday_flag==1 semua skenario: {june_ok}")

    t5_rows = []
    t6_rows = []
    scen_data = {}
    for scen in SCENARIOS:
        s = sc[sc["scenario"] == scen].sort_values("month").reset_index(drop=True)
        future = s[["date"] + regs].copy()  # hanya 4 regressor terkunci (peak_season DILARANG)
        p50 = predict_prophet(model, future, regs)          # prediksi titik (ruang asli, clip>=0)
        p10 = np.clip(p50 - q, 0, None)                      # batas bawah conformal C1
        p90 = p50 + q                                        # batas atas conformal C1

        # Agregasi tahunan Monte Carlo (seperti Fase 3)
        np.random.seed(42)
        idx = np.random.randint(0, n, size=(B, len(p50)))
        months = np.clip(p50[None, :] + calib_resid[idx], 0, None)
        totals = months.sum(axis=1)
        a10, a50, a90 = np.percentile(totals, [10, 50, 90])

        scen_data[scen] = {"months": s["month"].values, "p10": p10, "p50": p50, "p90": p90,
                           "a10": a10, "a50": a50, "a90": a90,
                           "dates": s["date"].values}
        for i in range(len(s)):
            t5_rows.append({
                "scenario": scen, "month": int(s["month"].values[i]),
                "date": s["date"].values[i],
                "pred_P10": float(p10[i]), "pred_P50": float(p50[i]), "pred_P90": float(p90[i]),
                "annual_P10": float(a10), "annual_P50": float(a50), "annual_P90": float(a90),
            })
        t6_rows.append({"scenario": scen, "annual_P10": float(a10),
                        "annual_P50": float(a50), "annual_P90": float(a90),
                        "sum_monthly_P50": float(p50.sum())})
        print(f"\n[{scen}] annual P10={a10:.4e} P50={a50:.4e} P90={a90:.4e} | "
              f"sum monthly P50={p50.sum():.4e}")

    t5 = pd.DataFrame(t5_rows)[["scenario", "month", "date", "pred_P10", "pred_P50",
                                "pred_P90", "annual_P10", "annual_P50", "annual_P90"]]
    t5.to_csv(TABLE5, index=False)
    print(f"\nSaved: {TABLE5} (n={len(t5)})")

    t6 = pd.DataFrame(t6_rows)[["scenario", "annual_P10", "annual_P50", "annual_P90",
                                "sum_monthly_P50"]]
    t6.to_csv(TABLE6, index=False)
    print(f"Saved: {TABLE6}")
    try:
        print("\n=== table6_ringkasan_proyeksi_2026.csv ===")
        print(t6.to_markdown(index=False))
    except Exception:
        print(t6.to_string(index=False))

    # --- fig5: fan chart 3 skenario ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    colors = {"pesimis": "tab:red", "dasar": "tab:blue", "optimis": "tab:green"}
    month_labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    for ax, scen in zip(axes, SCENARIOS):
        d = scen_data[scen]
        x = np.arange(12)
        c = colors[scen]
        ax.fill_between(x, d["p10"], d["p90"], color=c, alpha=0.20, label="P10–P90 (conformal C1)")
        ax.plot(x, d["p50"], "o-", color=c, lw=1.8, ms=4, label="P50 (titik Prophet)")
        ax.set_title(f"Skenario {scen.upper()}\nAnnual P50 = Rp {d['a50']/1e6:.1f} juta", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_xlabel("Bulan (2026)"); ax.grid(alpha=0.3)
        # tandai Juni (2 tanggal merah)
        ax.axvline(5, color="gray", ls=":", lw=1)
        ax.legend(loc="upper left", fontsize=7)
    axes[0].set_ylabel("Pajak Hotel (Rp)")
    fig.suptitle("Fase 5 — Fan Chart Proyeksi Pajak Hotel 2026 (KUA-PPAS, conditional forecasting)\n"
                 "Model beku Prophet (Train 2020-2023, flat) + kalibrasi Split Conformal C1 (residual Calib 2024)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG5, dpi=150); plt.close(fig)
    print(f"Saved: {FIG5}")

    # --- Kriteria penerimaan Fase 5 ---
    print("\n=== KRITERIA PENERIMAAN FASE 5 ===")
    # 1. no error (sampai sini = OK)
    print("1. Inference tanpa error: OK")
    # 2. P10<P50<P90 tiap bulan & skenario
    strict = bool(((t5["pred_P10"] < t5["pred_P50"]) & (t5["pred_P50"] < t5["pred_P90"])).all())
    print(f"2. P10<P50<P90 (semua bulan & skenario): {'OK' if strict else 'FAIL'}")
    # 3. annual dasar in [4e8, 5.5e8]
    dasar_a50 = float(t6.loc[t6["scenario"] == "dasar", "annual_P50"].iloc[0])
    in_range = 4.0e8 <= dasar_a50 <= 5.5e8
    print(f"3. Annual P50 dasar = {dasar_a50:.4e} in [4e8,5.5e8]: {'OK' if in_range else 'DI LUAR'}")
    # 4. June > May & July (P50) semua skenario
    print("4. Juni P50 > Mei & Juli (semua skenario):")
    june_ok_all = True
    for scen in SCENARIOS:
        d = scen_data[scen]
        may, jun, jul = d["p50"][4], d["p50"][5], d["p50"][6]
        ok = (jun > may) and (jun > jul)
        june_ok_all = june_ok_all and ok
        print(f"   {scen:8s}: Mei={may:.3e} Jun={jun:.3e} Jul={jul:.3e} -> "
              f"Jun>Mei:{jun>may} Jun>Jul:{jun>jul} {'OK' if ok else 'FAIL'}")
    print(f"   => Juni>Mei/Juli semua skenario: {'OK' if june_ok_all else 'TIDAK (lapor jujur)'}")
    return t5, t6, scen_data


if __name__ == "__main__":
    main()
