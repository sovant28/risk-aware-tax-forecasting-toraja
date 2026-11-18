"""conformal.py — Kalibrator conformal (Fase 2).

Ketentuan TERKUNCI (blueprint peneliti):
  - Kuantil conformal = order statistic ke-ceil((n+1)(1-alpha)) dari |residual| Calib,
    finite-sample corrected, di-cap ke n. Dengan n=12: alpha=0.10 -> k=12 (=max);
    alpha=0.05 -> k=13 di-cap 12 (=max) => JENUH (butuh n>=19 agar alpha=0.05 beda).
    Ini limitasi small-sample; TIDAK pakai interpolated quantile untuk klaim validity.
  - Interpolated quantile HANYA untuk grid CRPS (tau=0.05..0.95).
  - ACI (Gibbs & Candes 2021): alpha_{t+1}=clip(alpha_t + gamma*(alpha0 - 1{miss_t}), 0.01, 0.99).
  - Jackknife+ DITURUNKAN dari scope (A-013): LOO deret waktu -> kebocoran temporal intra-Calib.
  - DILARANG fit ulang model dasar pada Calib/Test; residual Calib dari model fit-Train-only.
"""
import numpy as np

CRPS_TAUS = np.round(np.arange(0.05, 0.951, 0.05), 10)  # 0.05..0.95 (19 titik)


def conformal_quantile(scores_abs, alpha):
    """Kuantil conformal finite-sample. Return (q, k, n).

    k = ceil((n+1)(1-alpha)), di-cap ke [1, n]. q = |score| terurut ke-k.
    Bila k mencapai n (saturasi), q = max|score| -> interval konservatif.
    """
    s = np.sort(np.asarray(scores_abs, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1]), k, n


def split_conformal_interval(pred, scores_calib_abs, alpha):
    """Split Conformal residual absolut (C1/C4/C5). Interval simetris pred +/- q."""
    q, k, n = conformal_quantile(scores_calib_abs, alpha)
    pred = np.asarray(pred, dtype=float)
    lower = np.clip(pred - q, 0, None)
    upper = pred + q
    return lower, upper, {"q": q, "k": k, "n": n}


def aci_intervals(pred, y, scores_calib_abs, alpha0=0.10, gamma=0.01, lo=0.01, hi=0.99):
    """ACI (C2). Interval per titik Test memakai alpha_t adaptif; alpha0 = target.

    Urutan kronologis. alpha_t dipakai untuk interval titik t (sebelum observasi y_t),
    lalu di-update: alpha_{t+1}=clip(alpha_t + gamma*(alpha0 - 1{miss_t}), lo, hi).
    Return lower, upper, alpha_used (per titik), miss (1/0), q_used (per titik).
    """
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    T = len(pred)
    alpha = float(alpha0)
    lower = np.empty(T); upper = np.empty(T)
    alpha_used = np.empty(T); miss = np.empty(T, dtype=int); q_used = np.empty(T)
    for t in range(T):
        q, k, n = conformal_quantile(scores_calib_abs, alpha)
        l = max(pred[t] - q, 0.0); u = pred[t] + q
        lower[t] = l; upper[t] = u; alpha_used[t] = alpha; q_used[t] = q
        m = 0 if (l <= y[t] <= u) else 1
        miss[t] = m
        alpha = float(np.clip(alpha + gamma * (alpha0 - m), lo, hi))
    return lower, upper, alpha_used, miss, q_used


def c3_native_intervals(pred, hw_test, scores_calib_norm, alpha):
    """C3 Conformalized Native Interval.

    scores_calib_norm_j = |resid_j| / hw_calib_j  (hw = half-width CI native Prophet, >0).
    q = conformal_quantile(scores, alpha). Interval test_i = pred_i +/- q * hw_test_i.
    Bebas pembagi-nol (hw native selalu >0) -> menangani heteroskedastisitas ruang asli.
    """
    q, k, n = conformal_quantile(scores_calib_norm, alpha)
    pred = np.asarray(pred, dtype=float); hw = np.asarray(hw_test, dtype=float)
    lower = np.clip(pred - q * hw, 0, None)
    upper = pred + q * hw
    return lower, upper, {"q": q, "k": k, "n": n}


# --- Predictive quantile functions (HANYA untuk CRPS; interpolated) ---

def qf_symmetric_conformal(pred_i, scores_calib_abs, taus):
    """Quantile function simetris di sekitar pred dari distribusi |score| calib.

    tau>0.5: pred + interp_quantile(|s|, 2tau-1)
    tau<0.5: pred - interp_quantile(|s|, 1-2tau)
    tau=0.5: pred (median).
    """
    s = np.asarray(scores_calib_abs, dtype=float)
    out = np.empty(len(taus))
    for i, t in enumerate(taus):
        if abs(t - 0.5) < 1e-12:
            out[i] = pred_i
        elif t > 0.5:
            out[i] = pred_i + np.quantile(s, 2 * t - 1)
        else:
            out[i] = pred_i - np.quantile(s, 1 - 2 * t)
    return out


def qf_c3_native(pred_i, hw_i, scores_calib_norm, taus):
    """C3 quantile function: simetris, skor ternormalisasi diskalakan hw_i (test)."""
    s = np.asarray(scores_calib_norm, dtype=float)
    out = np.empty(len(taus))
    for i, t in enumerate(taus):
        if abs(t - 0.5) < 1e-12:
            out[i] = pred_i
        elif t > 0.5:
            out[i] = pred_i + np.quantile(s, 2 * t - 1) * hw_i
        else:
            out[i] = pred_i - np.quantile(s, 1 - 2 * t) * hw_i
    return out


def qf_native_piecewise(pred_i, lower_i, upper_i, taus):
    """M0 quantile function (APROKSIMASI): interpolasi linear-piecewise dari
    3 anchor native CI: (0.05, lower), (0.5, pred), (0.95, upper).
    Didokumentasikan sebagai aproksimasi (M0 = strawman; metrik keputusan M0
    memakai native CI 90% eksak, bukan aproksimasi ini)."""
    out = np.empty(len(taus))
    for i, t in enumerate(taus):
        if t <= 0.5:
            frac = (t - 0.05) / (0.5 - 0.05)
            out[i] = lower_i + (pred_i - lower_i) * frac
        else:
            frac = (t - 0.5) / (0.95 - 0.5)
            out[i] = pred_i + (upper_i - pred_i) * frac
    return out
