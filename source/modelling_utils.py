from math import inf
from typing import Callable, Sequence, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import torch


def plot_oos_multi(models, ylabel="Equity premium"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, (r2, y_true, y_pred, dates) in models.items():
        ax.plot(dates, y_pred, label=f"{name} (R²={r2:.3f})")
    # plot true values from the first
    first_dates = list(models.values())[0][3]
    first_true = list(models.values())[0][1]
    ax.plot(first_dates, first_true, color="black", linewidth=1.5, label="True")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()

def baseline_forecast(
    y_true: np.ndarray,
    mode: str = "mean",
) -> np.ndarray:
    """
    Build a baseline forecast sequence for y_true.

    Parameters
    ----------
    y_true : 1D array of OOS realizations (after masking NaNs)
    mode   : "mean" | "rw" | "rw_drift"

    Returns
    -------
    baseline : 1D array, same length as y_true
               (may contain NaN in the first element for RW-type baselines)
    """
    y_true = np.asarray(y_true, float)

    if mode == "mean":
        # expanding mean: same as your original evaluate_oos
        return np.array([y_true[:i].mean() for i in range(1, len(y_true)+1)])

    elif mode == "rw":
        # https://agorism.dev/book/finance/time-series/James%20Douglas%20Hamilton%20-%20Time%20Series%20Analysis%20%281994%2C%20Princeton%20University%20Press%29%20-%20libgen.lc.pdf
        # random walk: forecast y_t by y_{t-1}
        b = np.empty_like(y_true)
        b[:] = np.nan
        b[0] = y_true[0]
        if len(y_true) > 1:
            b[1:] = y_true[:-1]
        return b

    elif mode == "rw_drift":
        # random walk with expanding drift in the OOS sample:
        # y_t^RW = y_{t-1} + μ_{t-1}, where μ_{t-1} is mean of Δy up to t-1
        b = np.empty_like(y_true)
        b[:] = np.nan
        if len(y_true) > 1:
            # differences Δy_t = y_t - y_{t-1}
            diffs = np.diff(y_true)
            # expanding mean of diffs
            drift = np.array([diffs[:i].mean() for i in range(1, len(diffs)+1)])
            # baseline from t=1 onward: y_{t-1} + μ_{t-1}
            b[1:] = y_true[:-1] + drift
        return b

    else:
        raise ValueError(f"Unknown baseline mode: {mode}")
def evaluate_oos(y_true, y_pred, model_name="Model", device="cpu", quiet=False, mode: str = "mean",):
    """
    Compute MSE, RMSE and out-of-sample R² (Campbell–Thompson style)
    using the expanding mean as the benchmark forecast.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]

    if len(y_true) == 0:
        if not quiet:
            print(f"[{model_name}] No valid predictions (all NaN).")
        return np.nan

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    mean_forecast = baseline_forecast(y_true, mode=mode)
    denom = np.sum((y_true - mean_forecast) ** 2)
    r2_oos = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else np.nan

    if not quiet:
        print(f"[{model_name}] Device={device} | Valid months={len(y_true)} | "
              f"MSE={mse:.6f} | RMSE={rmse:.6f} | R²_OS={r2_oos:.4f}")

    return r2_oos


def plot_oos(
    y_true,
    y_pred,
    dates=None,
    title="Out-of-sample forecast",
    ylabel="Equity premium",
    save_path=None,
    show=True,
    mode = 'mean'
):
    """
    Plot true values, model predictions, and expanding-mean benchmark.
    Assumes 1-step-ahead series (use horizon=1 slice if you did multi-step).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]

    #csum = np.cumsum(y_true)
    #mean_forecast = csum / np.arange(1, len(y_true) + 1)
    mean_forecast = baseline_forecast(y_true,mode)
    if dates is not None:
        x = pd.to_datetime(pd.Index(dates))[m]
    else:
        x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_true, label="True")
    ax.plot(x, y_pred, label="Prediction")
    ax.plot(x, mean_forecast, label="Expanding mean", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date" if dates is not None else "OOS step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df.index is a DatetimeIndex and sorted."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def align_monthly(series: pd.DataFrame, freq: str = "MS", col: str | None = None) -> pd.DataFrame:
    """
    Align a single-column DataFrame to monthly frequency with ffill.
    freq: "M" (month-end) or "MS" (month-start).
    """
    z = series.copy()
    if col is None:
        if z.shape[1] != 1:
            raise ValueError("align_monthly expects a single-column DataFrame or pass col.")
        col = z.columns[0]
    z.index = z.index.to_period(freq).to_timestamp(freq)
    z = z[~z.index.duplicated(keep="last")].sort_index().asfreq(freq)
    z[col] = z[col].ffill()
    return z


def ct_truncate(x: float | np.ndarray) -> float | np.ndarray:
    """Campbell–Thompson truncation at 0."""
    return np.maximum(x, 0.0)


def expand_start_with_min_history(
    index: pd.DatetimeIndex,
    start_oos: str | pd.Timestamp,
    min_history_months: int,
) -> pd.Timestamp:
    """
    Ensure at least min_history_months of past data before first OOS point.
    """
    start_oos = pd.Timestamp(start_oos)
    if start_oos < index.min():
        start_oos = index.min()

    pos0 = index.get_indexer([start_oos], method="backfill")[0]
    while pos0 < min_history_months and pos0 < len(index):
        pos0 += 1

    if pos0 >= len(index):
        raise ValueError("No valid OOS start date with sufficient history.")
    return index[pos0]


# ================================================================
# 2. GENERIC EXPANDING OOS DRIVERS
# ================================================================

def expanding_oos_tabular(
    data: pd.DataFrame,
    target_col: str = "equity_premium",
    feature_cols: Sequence[str] | None = None,
    start_oos: str = "1965-01-01",
    start_date: str = "1927-01-01",
    min_train: int = 120,
    min_history_months: int | None = None,
    ct_cutoff: bool = False,
    quiet: bool = False,
    model_name: str = "Model",
    model_fit_predict_fn: Callable[[pd.DataFrame, pd.Series], float] | None = None,
    mode = "mean"
) -> Tuple[float, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Generic expanding-window OOS driver for tabular models (1-step ahead).
    """
    if model_fit_predict_fn is None:
        raise ValueError("You must supply model_fit_predict_fn.")

    df = ensure_datetime_index(data)
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in data.")

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

    if min_history_months is not None:
        start_ts = expand_start_with_min_history(df.index, start_oos, min_history_months)
    else:
        start_ts = pd.Timestamp(start_oos)

    loop_dates = df.index[df.index >= start_ts]

    preds, trues, oos_dates = [], [], []

    for date_t in loop_dates:
        pos = df.index.get_loc(date_t)
        est = df.iloc[:pos].copy()        # strictly past
        row_t = df.iloc[pos]

        if est[target_col].notna().sum() < min_train:
            continue

        y_true = float(row_t[target_col])
        if np.isnan(y_true):
            continue

        y_hat = model_fit_predict_fn(est, row_t)
        if y_hat is None or np.isnan(y_hat):
            continue

        if ct_cutoff:
            y_hat = float(ct_truncate(y_hat))

        preds.append(float(y_hat))
        trues.append(y_true)
        oos_dates.append(date_t)

    if not preds:
        raise RuntimeError(f"[{model_name}] No valid predictions produced.")

    trues = np.asarray(trues, float)
    preds = np.asarray(preds, float)
    r2 = evaluate_oos(trues, preds, model_name=model_name, device="cpu", quiet=quiet, mode = mode)
    return r2, trues, preds, pd.DatetimeIndex(oos_dates)


def expanding_oos_univariate(
    y: pd.Series,
    start_oos: str = "1965-01-01",
    prediction_length: int = 1,
    min_history_months: int = 240,
    ct_cutoff: bool = False,
    quiet: bool = False,
    model_name: str = "TS-Model",
    forecast_multi_step_fn: Callable[[pd.Series, pd.Timestamp, int], np.ndarray] | None = None,
    mode = "mean",
) -> Tuple[
    Dict[int, float],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    Dict[int, pd.DatetimeIndex],
]:
    """
    Generic expanding-window univariate OOS driver with arbitrary prediction_length.

    forecast_multi_step_fn(y_hist, origin_date, prediction_length) -> array of shape (H,)

    Returns dicts keyed by horizon h = 1..H:
      - r2[h]       : scalar R²_OS for horizon h
      - trues[h]    : np.array of true values at horizon h
      - preds[h]    : np.array of predictions at horizon h
      - dates[h]    : DatetimeIndex of evaluation dates for horizon h
    """
    if forecast_multi_step_fn is None:
        raise ValueError("forecast_multi_step_fn must be provided.")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be >= 1")

    y = y.astype("float32").dropna()
    if y.empty:
        raise ValueError("Target series is empty after cleaning.")

    # enforce min history before first origin
    start_ts = expand_start_with_min_history(
        y.index, start_oos, min_history_months=min_history_months
    )

    # also need enough future data to evaluate all horizons
    last_valid_origin = y.index[-prediction_length]  # index for t such that t+H-1 exists
    test_idx = y.index[(y.index >= start_ts) & (y.index <= last_valid_origin)]

    if not quiet:
        print(f"[{model_name}] prediction_length={prediction_length}, "
              f"origins={len(test_idx)}, first_origin={test_idx[0].date()}, "
              f"last_origin={test_idx[-1].date()}")

    preds = {h: [] for h in range(1, prediction_length+1)}
    trues = {h: [] for h in range(1, prediction_length+1)}
    dates = {h: [] for h in range(1, prediction_length+1)}

    for date_t in test_idx:
        pos = y.index.get_loc(date_t)
        if isinstance(pos, slice):
            pos = pos.start

        y_hist = y.iloc[:pos]
        if y_hist.isna().any():
            continue

        y_hat_vec = forecast_multi_step_fn(y_hist, date_t, prediction_length)
        y_hat_vec = np.asarray(y_hat_vec, float).reshape(-1)
        if len(y_hat_vec) < prediction_length:
            continue

        for h in range(1, prediction_length + 1):
            target_pos = pos + (h - 1)
            y_true = float(y.iloc[target_pos])
            y_hat = float(y_hat_vec[h-1])
            if np.isnan(y_true) or np.isnan(y_hat):
                continue
            if ct_cutoff:
                y_hat = float(ct_truncate(y_hat))
            trues[h].append(y_true)
            preds[h].append(y_hat)
            dates[h].append(y.index[target_pos])

    # convert to arrays/index & compute R² per horizon
    r2 = {}
    for h in range(1, prediction_length+1):
        trues[h] = np.asarray(trues[h], float)
        preds[h] = np.asarray(preds[h], float)
        dates[h] = pd.DatetimeIndex(dates[h])

        if len(trues[h]) == 0:
            r2[h] = np.nan
            if not quiet:
                print(f"[{model_name}] horizon h={h}: no valid predictions.")
        else:
            r2[h] = evaluate_oos(
                trues[h],
                preds[h],
                model_name=f"{model_name} (h={h})",
                device="cpu",
                quiet=quiet,
                mode = mode
            )

    return r2, trues, preds, dates


