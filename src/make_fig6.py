"""make_fig6.py — fig6_holiday_effect.png: bukti visual Policy Brief.

Narasi: "Banyak tanggal merah ≠ pendapatan naik." Puncak fiskal sebenarnya Juli
(libur sekolah) & Desember (akhir tahun / Rambu Solo'), bukan Juni (2 red dates).

Panel A (bukti HISTORIS data): pajak_hotel aktual Juni vs Juli per tahun Train (2020-2023).
Panel B (bukti MEKANISTIK model): dekomposisi Prophet (dasar 2026) Juni vs Juli —
  komponen musiman `yearly` (Juli > Juni) dan efek `holiday_flag` NEGATIF (Juni ditekan).

Model beku (Train 2020-2023, growth=flat) direproduksi; TIDAK ada fit ulang pada data baru.
Jalankan dari root repo:  .venv/bin/python src/make_fig6.py
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

from features import load_full, get_regressors  # noqa: E402
from models import fit_prophet                   # noqa: E402

np.random.seed(42)
warnings.filterwarnings("ignore")
import logging  # noqa: E402
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = os.path.join(ROOT, "data", "asumsi_sektoral_2026.csv")
FIGDIR = os.path.join(ROOT, "outputs", "figures")
FIG6 = os.path.join(FIGDIR, "fig6_holiday_effect.png")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = load_full()
    regs = get_regressors()
    train = df[df["year"] <= 2023].copy()

    # ---- Panel A: historis Juni vs Juli ----
    years = [2020, 2021, 2022, 2023]
    jun = [float(df[(df.year == y) & (df.month == 6)]["pajak_hotel"].values[0]) for y in years]
    jul = [float(df[(df.year == y) & (df.month == 7)]["pajak_hotel"].values[0]) for y in years]

    # ---- Panel B: dekomposisi Prophet Juni vs Juli 2026 (dasar) ----
    m = fit_prophet(train, regs, growth="flat", interval_width=0.90)
    sc = pd.read_csv(SCEN); sc["date"] = pd.to_datetime(sc["date"])
    s = sc[sc.scenario == "dasar"].sort_values("month").reset_index(drop=True)
    fut = s[["date"] + regs].rename(columns={"date": "ds"})
    fc = m.predict(fut)
    comp_cols = ["yearly", "holiday_flag", "jumlah_wisatawan", "tingkat_hunian_hotel", "rambu_solo_flag"]
    comp_cols = [c for c in comp_cols if c in fc.columns]
    jun_row = fc[fc["ds"].dt.month == 6].iloc[0]
    jul_row = fc[fc["ds"].dt.month == 7].iloc[0]
    trend_val = float(jun_row["trend"])

    # ---- Plot ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A
    xpos = np.arange(len(years)); w = 0.38
    axA.bar(xpos - w/2, np.array(jun)/1e6, w, label="Juni (2 tanggal merah)", color="tab:orange")
    axA.bar(xpos + w/2, np.array(jul)/1e6, w, label="Juli (puncak libur sekolah)", color="tab:blue")
    for i, y in enumerate(years):
        axA.annotate(f"{jun[i]/1e6:.1f}", (xpos[i]-w/2, jun[i]/1e6), ha="center", va="bottom", fontsize=8)
        axA.annotate(f"{jul[i]/1e6:.1f}", (xpos[i]+w/2, jul[i]/1e6), ha="center", va="bottom", fontsize=8)
    axA.set_xticks(xpos); axA.set_xticklabels([str(y) for y in years])
    axA.set_xlabel("Tahun (data historis Train)")
    axA.set_ylabel("Pajak Hotel (Rp juta)")
    axA.set_title("Panel A — Bukti HISTORIS: Juli ≥ Juni\n(2020 = nol pandemi; satuan juta Rp)", fontsize=11)
    axA.legend(fontsize=9); axA.grid(alpha=0.3, axis="y")

    # Panel B
    labels = {"yearly": "Musiman\n(yearly)", "holiday_flag": "holiday_flag\n(red dates)",
              "jumlah_wisatawan": "jumlah_\nwisatawan", "tingkat_hunian_hotel": "hunian_\nhotel",
              "rambu_solo_flag": "rambu_\nsolo"}
    comp_labels = [labels.get(c, c) for c in comp_cols]
    jun_vals = [float(jun_row[c])/1e6 for c in comp_cols]
    jul_vals = [float(jul_row[c])/1e6 for c in comp_cols]
    xpos2 = np.arange(len(comp_cols))
    axB.bar(xpos2 - w/2, jun_vals, w, label="Juni 2026", color="tab:orange")
    axB.bar(xpos2 + w/2, jul_vals, w, label="Juli 2026", color="tab:blue")
    axB.axhline(0, color="black", lw=0.8)
    for i in range(len(comp_cols)):
        axB.annotate(f"{jun_vals[i]:.1f}", (xpos2[i]-w/2, jun_vals[i]),
                     ha="center", va="bottom" if jun_vals[i] >= 0 else "top", fontsize=7)
        axB.annotate(f"{jul_vals[i]:.1f}", (xpos2[i]+w/2, jul_vals[i]),
                     ha="center", va="bottom" if jul_vals[i] >= 0 else "top", fontsize=7)
    axB.set_xticks(xpos2); axB.set_xticklabels(comp_labels, fontsize=8)
    axB.set_ylabel("Kontribusi komponen (Rp juta)")
    axB.set_title(f"Panel B — Dekomposisi Prophet (dasar 2026): mengapa Juli > Juni\n"
                  f"(trend flat konstan = Rp {trend_val/1e6:.1f} jt; holiday_flag JUNI negatif)", fontsize=11)
    axB.legend(fontsize=9); axB.grid(alpha=0.3, axis="y")

    fig.suptitle("fig6 — 'Banyak tanggal merah ≠ pendapatan naik': bukti untuk Policy Brief\n"
                 "Puncak fiskal: JULI (libur sekolah) & DESEMBER (akhir tahun/Rambu Solo'), BUKAN Juni",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG6, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {FIG6}")

    # Ringkasan angka untuk laporan
    print("\n=== Ringkasan bukti ===")
    print("Panel A (historis Juni vs Juli, Rp juta):")
    for y, a, b in zip(years, jun, jul):
        print(f"  {y}: Jun={a/1e6:6.1f}  Jul={b/1e6:6.1f}  (Jul-Jun={(b-a)/1e6:+6.1f})")
    print(f"Panel B (dekomposisi 2026 dasar, Rp juta): holiday_flag Juni={float(jun_row['holiday_flag'])/1e6:+.1f}, "
          f"Juli={float(jul_row['holiday_flag'])/1e6:+.1f}; yearly Juni={float(jun_row['yearly'])/1e6:+.1f}, "
          f"Juli={float(jul_row['yearly'])/1e6:+.1f}")
    return fig


if __name__ == "__main__":
    main()
