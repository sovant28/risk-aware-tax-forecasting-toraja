"""models.py — Training & forecasting model dasar (Langkah 2 / Fase 1).

DEVIASI A-010 (disetujui peneliti): pipeline utama di RUANG ASLI (pajak_hotel,
Rupiah), BUKAN log1p. `log1p` dihapus karena artefak ramp struktural (9 nol eksak
pandemi) yang membuat ekstrapolasi tren linear meledak setelah invers expm1.

Hyperparameter TERKUNCI (§4):
  - Prophet: changepoint_prior_scale=0.8736, seasonality_prior_scale=11.67,
    regressor_prior_scale=2.98 (via add_regressor). growth dipilih via rolling-origin
    CV dalam Train (default a-priori 'flat' = konservatisme fiskal; lihat A-010).
    Regressor kontinu di-standardize INTERNAL oleh Prophet (standardize='auto').
  - SARIMA (1,1,1)(1,1,1)12 — univariate, ruang asli (A-010), tanpa regressor.
  - Seasonal Naive sp=12 — ruang asli (non-parametrik), tidak berubah.
DILARANG fit ulang pada Calibration/Test: SARIMA/SNaive meramal calib+test
sekaligus dari Train (satu horizon kontinu). Prediksi di-clip >= 0.
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sktime.forecasting.naive import NaiveForecaster

TARGET = "pajak_hotel"  # ruang asli (A-010)


def fit_prophet(train_df, regressors, growth="flat", interval_width=0.90):
    """Fit Prophet di ruang asli. growth: 'flat' (default, A-010/A-011) atau 'linear'.

    interval_width=0.90 -> native CI 90% (yhat_lower/upper) untuk M0 & C3 (Fase 2).
    Tidak memengaruhi point prediction maupun fit (hanya laporan CI bawaan).
    """
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.8736,
        seasonality_prior_scale=11.67,
        growth=growth,
        interval_width=interval_width,
    )
    for reg in regressors:
        # prior_scale=2.98 (regressor_prior_scale §4); standardize='auto' internal.
        m.add_regressor(reg, prior_scale=2.98)
    d = train_df[["date", TARGET] + regressors].rename(
        columns={"date": "ds", TARGET: "y"})
    m.fit(d)
    return m


def predict_prophet(m, df, regressors):
    """Point prediction ruang asli, clip >= 0. (dipakai CV / Fase 1 point metrics)."""
    fu = df[["date"] + regressors].rename(columns={"date": "ds"})
    return np.clip(m.predict(fu)["yhat"].values, 0, None)


def predict_prophet_ci(m, df, regressors):
    """Point + native CI (lower/upper). lower di-clip >= 0 (pajak tak mungkin negatif).

    REPRODUKSIBILITAS: Prophet menghitung interval ketidakpastian (yhat_lower/upper)
    via SAMPLING Monte-Carlo tren masa depan -> bergantung state RNG global. Point
    forecast (yhat, MAP) deterministik. Agar native CI reproduktif & konsisten antar
    script (Fase 1/2/3, O3 == Fase 2), seed di-reset tepat sebelum predict. Side-effect:
    RNG global di-reset ke 42 (Blok B Fase 3 re-seed sendiri sebelum Monte Carlo).
    """
    import numpy as _np
    _np.random.seed(42)
    fu = df[["date"] + regressors].rename(columns={"date": "ds"})
    fc = m.predict(fu)
    yhat = np.clip(fc["yhat"].values, 0, None)
    lower = np.clip(fc["yhat_lower"].values, 0, None)
    upper = np.clip(fc["yhat_upper"].values, 0, None)
    return yhat, lower, upper


def run_prophet(train, calib, test, regressors, growth="flat"):
    """Fit Train-only (DILARANG refit di Calib/Test). Kembalikan dict calib & test:
    {'yhat','lower','upper'} di ruang asli. lower/upper = native CI 90% (A-013/Fase 2).
    """
    m = fit_prophet(train, regressors, growth=growth, interval_width=0.90)
    c_yhat, c_lo, c_up = predict_prophet_ci(m, calib, regressors)
    t_yhat, t_lo, t_up = predict_prophet_ci(m, test, regressors)
    calib_out = {"yhat": c_yhat, "lower": c_lo, "upper": c_up}
    test_out = {"yhat": t_yhat, "lower": t_lo, "upper": t_up}
    return calib_out, test_out


def run_sarima(train_y, calib_len, test_len):
    """SARIMA (1,1,1)(1,1,1)12 di RUANG ASLI (A-010). Forecast kontinu calib+test."""
    model = SARIMAX(np.asarray(train_y, dtype=float),
                    order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    res = model.fit(disp=False)
    full = np.asarray(res.forecast(steps=calib_len + test_len))
    return np.clip(full[:calib_len], 0, None), np.clip(full[calib_len:], 0, None)


def run_snaive(train_y_orig, calib_len, test_len):
    """Seasonal Naive ruang asli (non-parametric). y sebagai pd.Series (sktime 1.x)."""
    y = pd.Series(np.asarray(train_y_orig, dtype=float))
    forecaster = NaiveForecaster(strategy="last", sp=12)
    forecaster.fit(y)
    full = np.asarray(forecaster.predict(fh=np.arange(1, calib_len + test_len + 1))).ravel()
    return np.clip(full[:calib_len], 0, None), np.clip(full[calib_len:], 0, None)
