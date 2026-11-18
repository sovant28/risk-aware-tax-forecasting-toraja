"""evaluate.py — Metrik titik & MASE terkunci (Langkah 2 / Fase 1).

Definisi (§4):
  - MAE, RMSE, sMAPE (guard pembagi nol), MASE (m=12).
  - MASE penyebut = MAE seasonal naive pada TRAIN (ruang asli), BUKAN in-sample:
    scale = mean(|y_t - y_{t-12}|) untuk t pada Train.
Semua metrik dihitung di ruang ASLI (Rupiah), setelah invers expm1 + clip>=0.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TABLE1 = os.path.join(ROOT, "outputs", "tables", "table1_point_metrics.csv")
EPS = 1e-8


def inverse_and_clip(y_log):
    return np.clip(np.expm1(y_log), 0, None)


def calc_metrics(y_true, y_pred, y_train_orig):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train_orig = np.asarray(y_train_orig, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    smape = np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + EPS)
    ) * 100

    # MASE: scale = MAE seasonal-naive (m=12) pada Train, ruang asli.
    naive_errors = np.abs(y_train_orig[12:] - y_train_orig[:-12])
    scale = np.mean(naive_errors)
    if scale < EPS:
        scale = EPS
    mase = mae / scale

    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "MASE": mase}


def save_table1(results_dict, path=DEFAULT_TABLE1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results_dict).T
    df.index.name = "Model"
    df.to_csv(path)
    print(f"Saved to {path}")
    try:
        print(df.to_markdown())
    except Exception:
        print(df.to_string())
    return df
