"""evaluate_prob.py — Metrik probabilistik Fase 2.

Per baris kalibrator (evaluasi HANYA pada Test 2025, satu kali):
  - Coverage90 : fraksi actual dalam [lower, upper] (nominal 90%).
  - MeanWidth  : mean(upper - lower).
  - Winkler90  : skor Winkler pada level alpha (default 0.10); lebih kecil = lebih baik.
                 W = (u-l) + (2/alpha)(l-y)1{y<l} + (2/alpha)(y-u)1{y>u}.
                 Untuk ACI (C2) alpha per titik (alpha_t) dipakai agar adil.
  - CRPS       : 2 * integral pinball loss atas grid tau=0.05..0.95 (trapezoid).
                 CRPS = 2 * trapz(rho_tau(y, q_tau), tau). Aproksimasi (ekor <0.05 &
                 >0.95 diabaikan) sesuai blueprint; formula identik antar metode ->
                 perbandingan valid.
"""
import numpy as np

# numpy 2.x mengganti np.trapz -> np.trapezoid; pakai alias aman.
_TRAPZ = getattr(np, "trapezoid", np.trapz)


def coverage(lower, upper, y):
    lower = np.asarray(lower, float); upper = np.asarray(upper, float); y = np.asarray(y, float)
    return float(np.mean((y >= lower) & (y <= upper)))


def mean_width(lower, upper):
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def winkler(lower, upper, y, alpha):
    """alpha: skalar (C1/C3/C4/C5/M0 = 0.10) atau array per-titik (C2 = alpha_t)."""
    lower = np.asarray(lower, float); upper = np.asarray(upper, float); y = np.asarray(y, float)
    alpha = np.asarray(alpha, float) if np.ndim(alpha) else np.full(len(y), float(alpha))
    width = upper - lower
    pen_lo = (2.0 / alpha) * np.clip(lower - y, 0, None)   # y < lower
    pen_up = (2.0 / alpha) * np.clip(y - upper, 0, None)   # y > upper
    return float(np.mean(width + pen_lo + pen_up))


def pinball(y, q, tau):
    y = np.asarray(y, float); q = np.asarray(q, float)
    diff = y - q
    return float(np.mean(np.where(diff >= 0, tau * diff, (tau - 1) * diff)))


def crps_from_qf(y, qf_matrix, taus):
    """qf_matrix: (n_test, n_tau) kuantil prediktif. CRPS = 2*trapz(pinball_tau, tau) per titik, lalu rata-rata.

    pinball per titik per tau dihitung, diintegrasi (trapezoid) atas tau, dikali 2.
    """
    y = np.asarray(y, float)
    qf_matrix = np.asarray(qf_matrix, float)
    taus = np.asarray(taus, float)
    n = len(y)
    crps_vals = np.empty(n)
    for i in range(n):
        rho = np.empty(len(taus))
        for j, t in enumerate(taus):
            d = y[i] - qf_matrix[i, j]
            rho[j] = t * d if d >= 0 else (t - 1) * d
        crps_vals[i] = 2.0 * _TRAPZ(rho, taus)
    return float(np.mean(crps_vals)), crps_vals


def prob_metrics(lower, upper, y, qf_matrix, taus, alpha=0.10):
    crps, _ = crps_from_qf(y, qf_matrix, taus)
    return {
        "Coverage90": coverage(lower, upper, y),
        "MeanWidth": mean_width(lower, upper),
        "Winkler90": winkler(lower, upper, y, alpha),
        "CRPS": crps,
    }
