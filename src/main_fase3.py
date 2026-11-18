"""main_fase3.py — Fase 3: Robustness Rezim (rolling-origin conformal) + Skenario Fiskal.

BLOK A — table3_regime_robustness.csv:
  3 origin, tiap origin REFIT Prophet (ruang asli, growth='flat', HP terkunci, 4 regressor)
  HANYA pada jendela train; kalibrasi C1 & C2 pada tahun calib; evaluasi tahun test:
    O1: train<=2021 -> calib 2022 -> test 2023 (R2 pemulihan)
    O2: train<=2022 -> calib 2023 -> test 2024 (R3 boom politik/anomali okupansi)
    O3: train<=2023 -> calib 2024 -> test 2025 (R4 normalisasi; = config Fase 1/2)
  Baris: origin/rezim x kalibrator {M0,C1,C2} x {Coverage90,MeanWidth,Winkler90,RelWidth}.
  R1 (pandemi) TIDAK diuji sebagai test (tak ada data pra-2020) -> narasi saja.
  Jika fit Prophet gagal (mis. O1 rentang nol panjang) -> baris dilaporkan GAGAL jujur.

BLOK B — table4_scenarios_2025.csv + fig3 (rekonstruksi skenario berbasis bukti):
  Split utama (train<=2023, calib 2024). Dari distribusi prediktif conformal C1
  (pred + residual calib signed, clip>=0): skenario bulanan P10/P50/P90.
  Agregasi tahunan Monte Carlo: B=10000, seed 42; tiap draw sampling residual calib
  WITH REPLACEMENT per bulan (independen antar-bulan = limitasi, A-014), jumlah 12 bulan
  (clip bulanan >=0); persentil 10/50/90 dari distribusi total tahunan.
  Skenario 2026 DITUNDA (butuh regressor & flag 2026 yang belum tersedia, A-016).

Output: table3, table4, preds_fase3.csv, fig3. Evaluasi test per origin SATU kali.
Jalankan dari root repo:  .venv/bin/python src/main_fase3.py
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

from features import load_full, get_regressors          # noqa: E402
from models import fit_prophet, predict_prophet_ci       # noqa: E402
from conformal import split_conformal_interval, aci_intervals  # noqa: E402
from evaluate_prob import coverage, mean_width, winkler  # noqa: E402

np.random.seed(42)
warnings.filterwarnings("ignore")
import logging  # noqa: E402
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE3 = os.path.join(ROOT, "outputs", "tables", "table3_regime_robustness.csv")
TABLE4 = os.path.join(ROOT, "outputs", "tables", "table4_scenarios_2025.csv")
PREDS3 = os.path.join(ROOT, "outputs", "intermediate", "preds_fase3.csv")
FIGDIR = os.path.join(ROOT, "outputs", "figures")
FIG3 = os.path.join(FIGDIR, "fig3_fiscal_scenarios.png")
FIG4 = os.path.join(FIGDIR, "fig4_native_vs_conformal_o1.png")

ALPHA = 0.10
GAMMA = 0.01
B = 10000
ORIGINS = [
    ("O1", "R2_pemulihan", 2021, 2022, 2023),
    ("O2", "R3_boom_politik", 2022, 2023, 2024),
    ("O3", "R4_normalisasi", 2023, 2024, 2025),
]


def eval_origin(df, regs, train_max_year, calib_year, test_year):
    """Refit Prophet train-only, kalibrasi C1/C2, evaluasi test. Return (metrics, detail) atau (error, None)."""
    train = df[df["year"] <= train_max_year].copy()
    calib = df[df["year"] == calib_year].copy()
    test = df[df["year"] == test_year].copy()
    try:
        m = fit_prophet(train, regs, growth="flat", interval_width=0.90)
        c_yhat, c_lo, c_up = predict_prophet_ci(m, calib, regs)
        t_yhat, t_lo, t_up = predict_prophet_ci(m, test, regs)
    except Exception as e:  # fit/predict GAGAL -> lapor jujur, tanpa fallback
        return {"error": f"{type(e).__name__}: {e}"}, None

    resid_calib = calib["pajak_hotel"].values.astype(float) - c_yhat   # signed
    abs_rc = np.abs(resid_calib)
    y_t = test["pajak_hotel"].values.astype(float)

    m0_lo, m0_up = np.clip(t_lo, 0, None), t_up
    c1_lo, c1_up, c1_info = split_conformal_interval(t_yhat, abs_rc, ALPHA)
    c2_lo, c2_up, c2_alpha, c2_miss, c2_q = aci_intervals(t_yhat, y_t, abs_rc, ALPHA, GAMMA)

    mean_abs = float(np.mean(np.abs(y_t)))
    point_mae = float(np.mean(np.abs(y_t - t_yhat)))  # MAE point Prophet sebelum kalibrasi (Q1)
    metrics = {}
    for name, (lo, up, al) in {
        "M0": (m0_lo, m0_up, ALPHA),
        "C1": (c1_lo, c1_up, ALPHA),
        "C2": (c2_lo, c2_up, c2_alpha),
    }.items():
        mw = mean_width(lo, up)
        metrics[name] = {
            "Coverage90": coverage(lo, up, y_t), "MeanWidth": mw,
            "Winkler90": winkler(lo, up, y_t, al), "RelWidth": mw / mean_abs,
        }
    detail = {
        "dates": test["date"].values, "y_t": y_t, "pred": t_yhat,
        "m0": (m0_lo, m0_up), "c1": (c1_lo, c1_up),
        "c2": (c2_lo, c2_up, c2_alpha, c2_miss),
        "resid_calib": resid_calib, "c1_info": c1_info, "point_mae": point_mae,
    }
    return metrics, detail


def blok_a(df, regs):
    rows = []
    details = {}
    pooled = {"C1": [], "C2": []}
    for oid, regime, tmy, cy, ty in ORIGINS:
        metrics, detail = eval_origin(df, regs, tmy, cy, ty)
        if detail is None:
            print(f"[{oid} {regime}] FIT GAGAL: {metrics['error']} -> baris ditandai FAILED (jujur, tanpa fallback)")
            for cal in ("M0", "C1", "C2"):
                rows.append({"origin": oid, "regime": regime, "calibrator": cal,
                             "status": "FIT_FAILED", "Coverage90": np.nan, "MeanWidth": np.nan,
                             "Winkler90": np.nan, "RelWidth": np.nan, "PointMAE": np.nan})
            continue
        details[oid] = detail
        for cal in ("M0", "C1", "C2"):
            r = {"origin": oid, "regime": regime, "calibrator": cal, "status": "OK"}
            r.update(metrics[cal])
            r["PointMAE"] = detail["point_mae"]  # sama utk M0/C1/C2 (error titik base model)
            rows.append(r)
        # pooled coverage (C1, C2)
        for cal, key in [("C1", "c1"), ("C2", "c2")]:
            lo, up = detail[key][0], detail[key][1]
            for i in range(len(detail["y_t"])):
                pooled[cal].append((lo[i], up[i], detail["y_t"][i]))
        print(f"[{oid} {regime}] train<={tmy} calib{cy} test{ty} -> "
              f"C1 cov={metrics['C1']['Coverage90']:.3f} width={metrics['C1']['MeanWidth']:.3e} | "
              f"C2 cov={metrics['C2']['Coverage90']:.3f} | M0 cov={metrics['M0']['Coverage90']:.3f}")

    t3 = pd.DataFrame(rows)[["origin", "regime", "calibrator", "status",
                             "Coverage90", "MeanWidth", "Winkler90", "RelWidth", "PointMAE"]]
    t3.to_csv(TABLE3, index=False)
    print(f"\nSaved: {TABLE3}")
    try:
        print(t3.to_markdown(index=False))
    except Exception:
        print(t3.to_string(index=False))

    # Pooled coverage 36 bulan
    print("\n=== POOLED COVERAGE (gabungan test O1-O3) ===")
    pooled_res = {}
    for cal in ("C1", "C2"):
        arr = np.array(pooled[cal])  # (N,3): lo,up,y
        cov = float(np.mean((arr[:, 2] >= arr[:, 0]) & (arr[:, 2] <= arr[:, 1])))
        pooled_res[cal] = cov
        print(f"{cal}: n={len(arr)} pooled coverage90={cov:.4f} (>=0.85 ? {'OK' if cov>=0.85 else 'FAIL'})")
    # flag rezim C1 coverage < 0.75
    print("\n=== FLAG rezim C1 coverage < 0.75 ===")
    flagged = []
    for r in rows:
        if r["calibrator"] == "C1" and r["status"] == "OK" and r["Coverage90"] < 0.75:
            flagged.append(f"{r['origin']}/{r['regime']}={r['Coverage90']:.3f}")
    print("FLAG:" if flagged else "(tidak ada rezim C1 < 0.75)", flagged if flagged else "")
    return t3, details, pooled_res, flagged


def blok_b(details):
    """Skenario fiskal 2025 dari distribusi prediktif conformal C1 (split utama = O3)."""
    d = details["O3"]              # train<=2023, calib 2024, test 2025
    resid = d["resid_calib"]       # 12 residual calib SIGNED
    pred = d["pred"]               # 12 prediksi Prophet test 2025
    y_t = d["y_t"]                 # 12 actual test 2025
    dates = pd.to_datetime(d["dates"])
    n = len(resid)

    # Skenario bulanan: distribusi prediktif conformal = clip(pred_i + resid_j, 0), j=1..n
    monthly = []
    for i in range(len(pred)):
        dist = np.clip(pred[i] + resid, 0, None)
        p10, p50, p90 = np.percentile(dist, [10, 50, 90])
        monthly.append({"date": dates[i].strftime("%Y-%m"), "actual": float(y_t[i]),
                        "P10": float(p10), "P50": float(p50), "P90": float(p90)})

    # Agregasi tahunan Monte Carlo (B draw, residual with replacement, independen antar-bulan)
    np.random.seed(42)  # re-seed tepat sebelum MC untuk determinisme
    idx = np.random.randint(0, n, size=(B, len(pred)))      # (B,12) sampling with replacement
    months = np.clip(pred[None, :] + resid[idx], 0, None)   # clip bulanan >=0
    totals = months.sum(axis=1)                             # total tahunan per draw
    a10, a50, a90 = np.percentile(totals, [10, 50, 90])
    actual_annual = float(y_t.sum())

    t4 = pd.DataFrame(monthly)
    annual_row = pd.DataFrame([{"date": "ANNUAL_2025", "actual": actual_annual,
                                "P10": float(a10), "P50": float(a50), "P90": float(a90)}])
    t4 = pd.concat([t4, annual_row], ignore_index=True)
    t4.to_csv(TABLE4, index=False)
    print(f"\nSaved: {TABLE4}")
    try:
        print(t4.to_markdown(index=False))
    except Exception:
        print(t4.to_string(index=False))

    # --- Kriteria table4 ---
    print("\n=== KRITERIA table4 ===")
    mon = t4[t4["date"] != "ANNUAL_2025"]
    strict_mon = bool(((mon["P10"] < mon["P50"]) & (mon["P50"] < mon["P90"])).all())
    strict_ann = bool(a10 < a50 < a90)
    ratio = a50 / actual_annual
    in_band = bool(a10 <= actual_annual <= a90)
    print(f"P10<P50<P90 bulanan (strict): {'OK' if strict_mon else 'FAIL'}")
    print(f"P10<P50<P90 tahunan (strict): {'OK' if strict_ann else 'FAIL'}  (P10={a10:.4e} P50={a50:.4e} P90={a90:.4e})")
    print(f"annual P50/actual = {ratio:.4f} in [0.75,1.25] ? {'OK' if 0.75<=ratio<=1.25 else 'FAIL'}")
    print(f"actual annual 2025 = {actual_annual:.4e} (≈ Rp {actual_annual/1e6:.1f} juta)")
    print(f"actual annual dalam [P10,P90] ? {'YA' if in_band else 'TIDAK'}")

    # --- fig3 ---
    x = np.arange(len(pred))
    labels = [d.strftime("%b") for d in dates]
    p10 = mon["P10"].values; p50 = mon["P50"].values; p90 = mon["P90"].values
    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.fill_between(x, p10, p90, color="tab:blue", alpha=0.20, label="Pita P10–P90 (conformal C1)")
    ax.plot(x, p50, "o-", color="tab:blue", lw=1.5, ms=5, label="P50 (dasar)")
    ax.plot(x, p10, "--", color="tab:red", lw=1.2, label="P10 (pesimis/aman)")
    ax.plot(x, p90, "--", color="tab:green", lw=1.2, label="P90 (optimis)")
    ax.plot(x, y_t, "k^-", lw=1.5, ms=6, label="Actual 2025")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Bulan (2025)"); ax.set_ylabel("Pajak Hotel (Rp)")
    ax.set_title("Fase 3 — Skenario Fiskal 2025 (rekonstruksi berbasis bukti, conformal C1)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    # inset: distribusi total tahunan MC
    axin = ax.inset_axes([0.55, 0.55, 0.42, 0.38])
    axin.hist(totals, bins=50, color="tab:purple", alpha=0.6)
    for val, col, lab in [(a10, "tab:red", "P10"), (a50, "tab:blue", "P50"),
                          (a90, "tab:green", "P90"), (actual_annual, "black", "actual")]:
        axin.axvline(val, color=col, lw=1.5, ls="--" if lab == "actual" else "-", label=f"{lab}")
    axin.set_title(f"Total tahunan MC (B={B})", fontsize=8)
    axin.tick_params(labelsize=6)
    axin.legend(fontsize=6, loc="upper right")
    axin.set_xlabel("Rp", fontsize=7)

    fig.tight_layout(); fig.savefig(FIG3, dpi=150); plt.close(fig)
    print(f"Saved: {FIG3}")
    return t4, {"a10": a10, "a50": a50, "a90": a90, "actual_annual": actual_annual,
                "strict_mon": strict_mon, "strict_ann": strict_ann, "ratio": ratio, "in_band": in_band}


def save_preds_fase3(details):
    rows = []
    reg_map = {oid: reg for oid, reg, *_ in ORIGINS}
    for oid in ("O1", "O2", "O3"):
        if oid not in details:
            continue
        d = details[oid]
        c2_alpha, c2_miss = d["c2"][2], d["c2"][3]
        for i in range(len(d["y_t"])):
            rows.append({
                "origin": oid, "regime": reg_map[oid], "date": d["dates"][i],
                "actual": float(d["y_t"][i]), "pred_prophet": float(d["pred"][i]),
                "M0_lower": float(d["m0"][0][i]), "M0_upper": float(d["m0"][1][i]),
                "C1_lower": float(d["c1"][0][i]), "C1_upper": float(d["c1"][1][i]),
                "C2_lower": float(d["c2"][0][i]), "C2_upper": float(d["c2"][1][i]),
                "C2_alpha": float(c2_alpha[i]), "C2_miss": int(c2_miss[i]),
            })
    p3 = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(PREDS3), exist_ok=True)
    p3.to_csv(PREDS3, index=False)
    print(f"\nSaved: {PREDS3}  (n={len(p3)})")
    return p3


def make_fig4(details):
    """Hero image (Q2): O1 (train<=2021 -> test 2023, rezim R2 pemulihan).

    Memvisualisasikan H1/H2: CI native Prophet (M0) KOLAPS jadi garis tipis
    (width ~1e-4, coverage 0.00) sementara CI Conformal (C1) membungkus actual
    dengan aman (coverage 1.00). Bukti visual kalibrasi conformal esensial.
    """
    if "O1" not in details:
        print("[fig4] O1 tidak tersedia (fit gagal) -> fig4 dilewati.")
        return
    d = details["O1"]
    dates = pd.to_datetime(d["dates"])
    labels = [x.strftime("%b") for x in dates]
    x = np.arange(len(labels))
    y_t = d["y_t"]; pred = d["pred"]
    m0_lo, m0_up = d["m0"]; c1_lo, c1_up = d["c1"]
    m0_w = float(np.mean(m0_up - m0_lo)); c1_w = float(np.mean(c1_up - c1_lo))
    m0_cov = float(np.mean((y_t >= m0_lo) & (y_t <= m0_up)))
    c1_cov = float(np.mean((y_t >= c1_lo) & (y_t <= c1_up)))

    fig, ax = plt.subplots(figsize=(12, 7))
    # C1 conformal band (lebar, membungkus actual)
    ax.fill_between(x, c1_lo, c1_up, color="tab:green", alpha=0.18,
                    label=f"C1 Split Conformal 90%  (cov={c1_cov:.2f}, width≈{c1_w:.2e})")
    ax.plot(x, c1_lo, color="tab:green", lw=1.0, ls="-", alpha=0.6)
    ax.plot(x, c1_up, color="tab:green", lw=1.0, ls="-", alpha=0.6)
    # M0 native CI (degenerasi: width ~1e-4 -> garis tipis)
    ax.fill_between(x, m0_lo, m0_up, color="tab:red", alpha=0.9,
                    label=f"M0 Prophet native CI 90%  (cov={m0_cov:.2f}, width≈{m0_w:.1e} — DEGENERASI)")
    ax.plot(x, m0_lo, color="darkred", lw=1.2, ls="--")
    ax.plot(x, m0_up, color="darkred", lw=1.2, ls="--")
    # Point forecast & actual
    ax.plot(x, pred, "o-", color="tab:blue", lw=1.5, ms=5, label="Prediksi titik Prophet (flat)")
    ax.plot(x, y_t, "k^-", lw=1.8, ms=7, label="Actual 2023 (R2 pemulihan)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Bulan (2023 — Test O1)"); ax.set_ylabel("Pajak Hotel (Rp)")
    ax.set_title("Fase 3 — HERO IMAGE: CI Native Kolaps vs Conformal Menyelamatkan (Origin O1)\n"
                 "Train≤2021 (nol pandemi) → Prophet native CI degenerasi (width≈1e-4, actual lolos);\n"
                 "Split Conformal mengkalibrasi ulang dari residual Calib 2022 → membungkus actual (cov 1.00)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG4, dpi=150); plt.close(fig)
    print(f"Saved: {FIG4}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = load_full()
    regs = get_regressors()
    print("=== BLOK A: rolling-origin conformal (table3) ===")
    t3, details, pooled_res, flagged = blok_a(df, regs)
    save_preds_fase3(details)
    make_fig4(details)
    print("\n=== BLOK B: skenario fiskal 2025 (table4 + fig3) ===")
    t4, scen = blok_b(details)
    print("\n=== RINGKASAN KRITERIA FASE 3 ===")
    print(f"Pooled C1={pooled_res.get('C1'):.4f} C2={pooled_res.get('C2'):.4f} (>=0.85)")
    print(f"Flag rezim C1<0.75: {flagged if flagged else 'tidak ada'}")
    print(f"table4 strict monotonic: monthly={scen['strict_mon']} annual={scen['strict_ann']}; "
          f"P50/actual={scen['ratio']:.4f}; actual in [P10,P90]={scen['in_band']}")
    return t3, t4


if __name__ == "__main__":
    main()
