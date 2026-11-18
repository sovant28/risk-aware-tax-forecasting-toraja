"""make_ablation.py — Tabel ablation formal transformasi ruang (Q3 peneliti).

Baris: {Prophet log1p+linear, SARIMA log1p, Prophet raw+flat, SARIMA raw, SNaive}
Kolom: {MAE, RMSE, sMAPE, MASE} + provenance (commit git sumber angka).

PENTING (A-011): angka log1p DIAMBIL dari RIWAYAT GIT (run yang sudah menyentuh Test
sebelum freeze), BUKAN re-run baru. Ini menjaga "Test 2025 dievaluasi tepat satu kali
untuk konfigurasi beku". Angka raw+flat = table1 FINAL (commit b2ffb30).

Jalankan:  .venv/bin/python src/make_ablation.py
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tables", "table_ablation_logspace.csv")

# Angka persis dari `git show <commit>:outputs/tables/table1_point_metrics.csv`
ROWS = [
    # Model, transform, growth, MAE, RMSE, sMAPE, MASE, source_commit
    ("Prophet-BO", "log1p", "linear",
     714986273288.2723, 1576591703452.06, 185.17917484054684, 22837.092826505632,
     "a2b139f"),
    ("SARIMA", "log1p", "-",
     6462577767.854888, 11889643410.733358, 155.93515323928574, 206.4186319329642,
     "a2b139f"),
    ("Prophet-BO", "raw", "flat",
     20329456.388057176, 29149496.197676815, 54.1692500166322, 0.6493350991359162,
     "b2ffb30"),
    ("SARIMA", "raw", "-",
     28428941.771122973, 38775863.615027085, 58.12427096338402, 0.9080375476309273,
     "b2ffb30"),
    ("SeasonalNaive", "raw", "-",
     26898303.916666668, 37011117.203692816, 71.57513862011434, 0.859148051326026,
     "b2ffb30"),
]

COLS = ["Model", "transform", "growth", "MAE", "RMSE", "sMAPE", "MASE", "source_commit"]


def main():
    df = pd.DataFrame(ROWS, columns=COLS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved: {OUT}")
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
