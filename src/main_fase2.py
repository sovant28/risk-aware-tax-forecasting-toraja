"""main_fase2.py — Orchestrator Fase 2 (Conformal Prediction + ACI).

Baris table2 (kalibrator):
  M0  Prophet native CI 90% (tanpa kalibrasi) — strawman.
  C1  Split Conformal, residual absolut, alpha=0.10 — utama.
  C2  ACI (Gibbs & Candes 2021), residual absolut, gamma=0.01, alpha0=0.10 — adaptif.
  C3  Conformalized Native Interval (q dari |resid|/half-width native pada Calib).
  C4  Split Conformal residual absolut pada SARIMA (model-agnostic).
  C5  Split Conformal residual absolut pada SeasonalNaive (model-agnostic).

Input : outputs/intermediate/preds_fase1.csv (point preds + native CI Prophet, calib+test).
Output: outputs/tables/table2_probabilistic_metrics.csv
        outputs/intermediate/preds_fase2.csv (lower/upper per kalibrator per bulan Test)
        outputs/figures/fig1_intervals_2025.png (actual vs pred vs band C1 & C2)
        outputs/figures/fig2_aci_alpha_path.png (alpha_t + penanda miss)
Evaluasi HANYA pada Test 2025, SATU KALI. Determinisme: seed 42.
DILARANG fit ulang model dasar pada Calib/Test (residual Calib dari model fit-Train-only).

Jalankan dari root repo:  .venv/bin/python src/main_fase2.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from conformal import (  # noqa: E402
    CRPS_TAUS, conformal_quantile, split_conformal_interval, aci_intervals,
    c3_native_intervals, qf_symmetric_conformal, qf_c3_native, qf_native_piecewise,
)
from evaluate_prob import prob_metrics  # noqa: E402

np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS_IN = os.path.join(ROOT, "outputs", "intermediate", "preds_fase1.csv")
TABLE2 = os.path.join(ROOT, "outputs", "tables", "table2_probabilistic_metrics.csv")
PREDS2 = os.path.join(ROOT, "outputs", "intermediate", "preds_fase2.csv")
FIGDIR = os.path.join(ROOT, "outputs", "figures")
FIG1 = os.path.join(FIGDIR, "fig1_intervals_2025.png")
FIG2 = os.path.join(FIGDIR, "fig2_aci_alpha_path.png")

ALPHA = 0.10
GAMMA = 0.01


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = pd.read_csv(PREDS_IN)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    calib = df[df["split"] == "calib"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    # --- residual Calib (model fit-Train-only; TIDAK ada refit) ---
    r_prophet_c = (calib["actual"] - calib["pred_prophet"]).values
    r_sarima_c = (calib["actual"] - calib["pred_sarima"]).values
    r_snaive_c = (calib["actual"] - calib["pred_snaive"]).values
    a_rp, a_rs, a_rn = np.abs(r_prophet_c), np.abs(r_sarima_c), np.abs(r_snaive_c)

    # --- half-width native CI (Prophet) ---
    hw_c = ((calib["yhat_upper_native"] - calib["yhat_lower_native"]) / 2.0).values
    hw_t = ((test["yhat_upper_native"] - test["yhat_lower_native"]) / 2.0).values
    n_zero_hw = int((hw_c <= 0).sum() + (hw_t <= 0).sum())
    hw_floor = 1.0  # guard pembagi nol (A-013); native hw diharapkan selalu >0
    hw_c_safe = np.where(hw_c > 0, hw_c, hw_floor)
    scores_c3_c = a_rp / hw_c_safe  # skor ternormalisasi C3 pada Calib

    y_t = test["actual"].values
    pp_t = test["pred_prophet"].values
    ps_t = test["pred_sarima"].values
    pn_t = test["pred_snaive"].values

    # --- Diagnostik saturasi finite-sample (untuk laporan/A-013) ---
    q10, k10, n = conformal_quantile(a_rp, 0.10)
    q05, k05, _ = conformal_quantile(a_rp, 0.05)
    print("=== Diagnostik kuantil conformal (|resid Prophet| Calib, n=%d) ===" % n)
    print(f"alpha=0.10 -> k={k10}, q={q10:.4e}")
    print(f"alpha=0.05 -> k={k05}, q={q05:.4e}  "
          f"({'JENUH: sama dgn alpha=0.10' if abs(q05 - q10) < 1e-9 else 'beda'})")
    print(f"max|resid Prophet calib|={a_rp.max():.4e} (didominasi calib 2024-02, A-012)")

    # --- Bangun interval Test per kalibrator ---
    # M0: native CI 90%
    m0_lo = np.clip(test["yhat_lower_native"].values, 0, None)
    m0_up = test["yhat_upper_native"].values
    # C1: split conformal Prophet
    c1_lo, c1_up, c1_info = split_conformal_interval(pp_t, a_rp, ALPHA)
    # C2: ACI Prophet
    c2_lo, c2_up, c2_alpha, c2_miss, c2_q = aci_intervals(pp_t, y_t, a_rp, ALPHA, GAMMA)
    # C3: conformalized native
    c3_lo, c3_up, c3_info = c3_native_intervals(pp_t, hw_t, scores_c3_c, ALPHA)
    # C4: split conformal SARIMA
    c4_lo, c4_up, c4_info = split_conformal_interval(ps_t, a_rs, ALPHA)
    # C5: split conformal SNaive
    c5_lo, c5_up, c5_info = split_conformal_interval(pn_t, a_rn, ALPHA)

    print("\n=== ACI alpha path (C2) ===")
    print("alpha_used:", np.round(c2_alpha, 4))
    print("miss      :", c2_miss.tolist())
    print(f"q_used unik: {np.unique(np.round(c2_q, 6))}  "
          f"({'lebar TETAP (saturasi)' if len(np.unique(np.round(c2_q, 6))) == 1 else 'lebar BERUBAH (adaptif)'})")
    print(f"alpha range: [{c2_alpha.min():.4f}, {c2_alpha.max():.4f}]")

    # --- Quantile functions untuk CRPS ---
    qf_m0 = np.vstack([qf_native_piecewise(pp_t[i], m0_lo[i], m0_up[i], CRPS_TAUS)
                       for i in range(len(y_t))])
    qf_c1 = np.vstack([qf_symmetric_conformal(pp_t[i], a_rp, CRPS_TAUS) for i in range(len(y_t))])
    qf_c2 = qf_c1  # ACI memakai distribusi prediktif conformal yang sama (adapta level, bukan bentuk)
    qf_c3 = np.vstack([qf_c3_native(pp_t[i], hw_t[i], scores_c3_c, CRPS_TAUS) for i in range(len(y_t))])
    qf_c4 = np.vstack([qf_symmetric_conformal(ps_t[i], a_rs, CRPS_TAUS) for i in range(len(y_t))])
    qf_c5 = np.vstack([qf_symmetric_conformal(pn_t[i], a_rn, CRPS_TAUS) for i in range(len(y_t))])

    # --- Metrik probabilistik ---
    rows = {}
    rows["M0_ProphetNative"] = prob_metrics(m0_lo, m0_up, y_t, qf_m0, CRPS_TAUS, alpha=ALPHA)
    rows["C1_SplitConf"] = prob_metrics(c1_lo, c1_up, y_t, qf_c1, CRPS_TAUS, alpha=ALPHA)
    rows["C2_ACI"] = prob_metrics(c2_lo, c2_up, y_t, qf_c2, CRPS_TAUS, alpha=c2_alpha)
    rows["C3_ConfNative"] = prob_metrics(c3_lo, c3_up, y_t, qf_c3, CRPS_TAUS, alpha=ALPHA)
    rows["C4_SplitConf_SARIMA"] = prob_metrics(c4_lo, c4_up, y_t, qf_c4, CRPS_TAUS, alpha=ALPHA)
    rows["C5_SplitConf_SNaive"] = prob_metrics(c5_lo, c5_up, y_t, qf_c5, CRPS_TAUS, alpha=ALPHA)

    desc = {
        "M0_ProphetNative": "Prophet native CI 90% (tanpa kalibrasi; strawman)",
        "C1_SplitConf": "Split Conformal |resid| Prophet, alpha=0.10 (utama)",
        "C2_ACI": "ACI gamma=0.01 alpha0=0.10 Prophet (adaptif)",
        "C3_ConfNative": "Conformalized Native Interval (|resid|/hw native)",
        "C4_SplitConf_SARIMA": "Split Conformal |resid| SARIMA (model-agnostic)",
        "C5_SplitConf_SNaive": "Split Conformal |resid| SNaive (model-agnostic)",
    }

    t2 = pd.DataFrame(rows).T
    # RelWidth = MeanWidth / mean(|actual| Test 2025) — diagnostik sharpness-vs-coverage (Q2 Fase2).
    mean_abs_actual = float(np.mean(np.abs(y_t)))
    t2["RelWidth"] = t2["MeanWidth"] / mean_abs_actual
    roles = {
        "M0_ProphetNative": "reference", "C1_SplitConf": "main", "C2_ACI": "main",
        "C3_ConfNative": "ablation", "C4_SplitConf_SARIMA": "ablation",
        "C5_SplitConf_SNaive": "ablation",
    }
    t2.insert(0, "role", [roles[i] for i in t2.index])
    t2.insert(1, "description", [desc[i] for i in t2.index])
    t2 = t2[["role", "description", "Coverage90", "MeanWidth", "RelWidth", "Winkler90", "CRPS"]]
    t2.index.name = "Calibrator"
    t2.to_csv(TABLE2)
    print("\n=== table2_probabilistic_metrics.csv ===")
    try:
        print(t2.to_markdown())
    except Exception:
        print(t2.to_string())
    print(f"Saved: {TABLE2}")
    print(f"[footnote] RelWidth = MeanWidth / mean(|actual| Test 2025); "
          f"mean(|actual|)={mean_abs_actual:.4e}. C3/C4/C5 = ablation (A-015); "
          f"kalibrator utama Fase 3 hanya C1 & C2 (+ M0 referensi).")

    # --- preds_fase2.csv (lower/upper per kalibrator per bulan Test) ---
    p2 = pd.DataFrame({
        "date": test["date"].values, "actual": y_t, "pred_prophet": pp_t,
        "M0_lower": m0_lo, "M0_upper": m0_up,
        "C1_lower": c1_lo, "C1_upper": c1_up,
        "C2_lower": c2_lo, "C2_upper": c2_up, "C2_alpha": c2_alpha, "C2_miss": c2_miss,
        "C3_lower": c3_lo, "C3_upper": c3_up,
        "C4_lower": c4_lo, "C4_upper": c4_up,
        "C5_lower": c5_lo, "C5_upper": c5_up,
    })
    p2.to_csv(PREDS2, index=False)
    print(f"Saved: {PREDS2}  (n={len(p2)})")

    # --- fig1: actual vs pred vs band C1 & C2 ---
    x = np.arange(len(y_t))
    labels = [d.strftime("%b") for d in pd.to_datetime(test["date"])]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x, c1_lo, c1_up, color="tab:blue", alpha=0.20, label="C1 Split Conformal (90%)")
    ax.fill_between(x, c2_lo, c2_up, color="tab:red", alpha=0.20, label="C2 ACI (90%, adaptif)")
    ax.plot(x, pp_t, "o-", color="tab:green", lw=1.5, ms=4, label="Prediksi Prophet (flat)")
    ax.plot(x, y_t, "k^--", lw=1.5, ms=6, label="Actual 2025")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Bulan (2025)"); ax.set_ylabel("Pajak Hotel (Rp)")
    ax.set_title("Fase 2 — Interval Prediktif Test 2025: C1 (Split Conformal) vs C2 (ACI)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG1, dpi=150); plt.close(fig)
    print(f"Saved: {FIG1}")

    # --- fig2: ACI alpha path + miss markers + overlay batas saturasi (Q1 Fase2) ---
    SAT_BOUND = 2.0 / 13.0  # alpha=0.1538: ceil(13(1-alpha)) turun 12->11 (n=12)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhspan(0.0, SAT_BOUND, color="orange", alpha=0.08)
    ax.plot(x, c2_alpha, "o-", color="tab:red", lw=1.8, ms=6, label=r"$\alpha_t$ (ACI)")
    ax.axhline(ALPHA, color="gray", ls=":", lw=1, label=r"target $\alpha_0=0.10$")
    ax.axhline(SAT_BOUND, color="tab:orange", ls="--", lw=1.5,
               label=r"batas saturasi $\alpha=0.154$ (k:12$\to$11)")
    miss_idx = np.where(c2_miss == 1)[0]
    if len(miss_idx):
        ax.scatter(miss_idx, c2_alpha[miss_idx], s=180, facecolors="none",
                   edgecolors="black", lw=2, zorder=5, label="bulan MISS")
        for mi in miss_idx:
            ax.annotate(labels[mi], (mi, c2_alpha[mi]), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8)
    ax.set_ylim(0.09, 0.17)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Bulan (2025)"); ax.set_ylabel(r"$\alpha_t$")
    ax.set_title("Fase 2 — Lintasan $\\alpha_t$ ACI (gamma=0.01) + penanda miss + batas saturasi\n"
                 r"Sepanjang $\alpha_t<0.154$ (pita oranye), order-statistic k=12 -> lebar interval "
                 "TERPIN ke max|resid| (ACI adaptasi $\\alpha$, bukan lebar; n=12)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG2, dpi=150); plt.close(fig)
    print(f"Saved: {FIG2}")

    # --- Kriteria penerimaan Fase 2 ---
    print("\n=== KRITERIA PENERIMAAN FASE 2 ===")
    cov_c1 = rows["C1_SplitConf"]["Coverage90"]; cov_c2 = rows["C2_ACI"]["Coverage90"]
    w_m0 = rows["M0_ProphetNative"]["Winkler90"]
    w_c1 = rows["C1_SplitConf"]["Winkler90"]; w_c2 = rows["C2_ACI"]["Winkler90"]
    wid_c3 = rows["C3_ConfNative"]["MeanWidth"]; wid_c1 = rows["C1_SplitConf"]["MeanWidth"]
    print(f"1. Coverage90(C1)={cov_c1:.3f} >=0.90 ? {'OK' if cov_c1 >= 0.90 else 'FAIL'} | "
          f"Coverage90(C2)={cov_c2:.3f} >=0.90 ? {'OK' if cov_c2 >= 0.90 else 'FAIL'}")
    better = min(w_c1, w_c2) < w_m0
    print(f"2. Winkler90: M0={w_m0:.4e}, C1={w_c1:.4e}, C2={w_c2:.4e} -> "
          f"min(C1,C2)<M0 ? {'OK' if better else 'FAIL'}")
    print(f"3. MeanWidth: C3={wid_c3:.4e} < C1={wid_c1:.4e} ? "
          f"{'OK' if wid_c3 < wid_c1 else 'TIDAK (lapor jujur)'}")
    print(f"4. zero-hw guard terpakai: {n_zero_hw} (hw native diharapkan >0)")
    return t2


if __name__ == "__main__":
    main()
