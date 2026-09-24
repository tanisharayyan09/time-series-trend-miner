"""Temporal data mining: trend decomposition + forecasting.

Time-series mining project by Nishanthini M.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


# ---------- 1. DATA ----------
def generate_sample_data(kind: str = "sales", n: int = 365, seed: int = 42) -> pd.DataFrame:
    """Synthetic time series for demo purposes (no internet needed)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
    t = np.arange(n)

    if kind == "sales":
        trend = 200 + 0.6 * t
        weekly = 40 * np.sin(2 * np.pi * t / 7)
        noise = rng.normal(0, 15, n)
        value = trend + weekly + noise
    elif kind == "weather":
        trend = 28 + 0.002 * t
        yearly = 6 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
        noise = rng.normal(0, 1.2, n)
        value = trend + yearly + noise
    elif kind == "stock":
        drift, vol = 0.15, 2.5
        steps = rng.normal(drift, vol, n)
        value = 500 + np.cumsum(steps)
    else:
        raise ValueError("kind must be 'sales', 'weather' or 'stock'")

    return pd.DataFrame({"date": dates, "value": np.round(value, 2)})


def load_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Clean a user-supplied dataframe into a sorted (date, value) series."""
    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(out) < 10:
        raise ValueError("Need at least 10 valid rows of date/value data.")
    return out


# ---------- 2. TREND / SEASONALITY ----------
def decompose(df: pd.DataFrame, period: int = 7):
    """Split the series into trend, seasonal and residual components."""
    s = df.set_index("date")["value"].asfreq("D").interpolate()
    period = max(2, min(period, len(s) // 2))
    result = seasonal_decompose(s, model="additive", period=period, extrapolate_trend="freq")
    return pd.DataFrame({
        "date": s.index,
        "observed": result.observed.values,
        "trend": result.trend.values,
        "seasonal": result.seasonal.values,
        "residual": result.resid.values,
    })


# ---------- 3. FORECASTING ----------
def _best_arima_order(series: pd.Series):
    """Small grid search over a few (p,d,q) combos, pick lowest AIC."""
    candidates = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1), (2, 1, 2)]
    best_order, best_aic, best_fit = (1, 1, 1), np.inf, None
    for order in candidates:
        try:
            fit = ARIMA(series, order=order).fit()
            if fit.aic < best_aic:
                best_order, best_aic, best_fit = order, fit.aic, fit
        except Exception:
            continue
    if best_fit is None:
        best_fit = ARIMA(series, order=(1, 1, 1)).fit()
    return best_order, best_fit


def forecast_arima(df: pd.DataFrame, periods: int = 30):
    """Fit ARIMA and forecast `periods` steps ahead with a confidence band."""
    s = df.set_index("date")["value"].asfreq("D").interpolate()
    order, fit = _best_arima_order(s)
    pred = fit.get_forecast(steps=periods)
    idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    ci = pred.conf_int(alpha=0.2)
    out = pd.DataFrame({
        "date": idx,
        "forecast": pred.predicted_mean.values,
        "lower": ci.iloc[:, 0].values,
        "upper": ci.iloc[:, 1].values,
    })
    return out, order


def forecast_prophet(df: pd.DataFrame, periods: int = 30):
    """Fit Prophet (if installed) and forecast `periods` steps ahead."""
    if not HAS_PROPHET:
        raise ImportError("prophet is not installed. Run: pip install prophet")
    m = Prophet()
    m.fit(df.rename(columns={"date": "ds", "value": "y"}))
    future = m.make_future_dataframe(periods=periods)
    fc = m.predict(future)
    tail = fc.tail(periods)
    return pd.DataFrame({
        "date": tail["ds"].values,
        "forecast": tail["yhat"].values,
        "lower": tail["yhat_lower"].values,
        "upper": tail["yhat_upper"].values,
    })
