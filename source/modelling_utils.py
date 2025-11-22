from math import inf
from typing import Callable, Sequence, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import torch
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
from sklearn.metrics import r2_score

def calculate_block_bootstrap_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
    block_size: int = 12,      # 12 months block size to preserve seasonality
    n_bootstraps: int = 1000
) -> dict:
    """
    Performs block bootstrapping on the *errors* of the model vs baseline
    to estimate the distribution of the R2 metric.
    """
    n_samples = len(y_true)
    r2_bootstraps = []
    
    # 1. Calculate Error Vectors (Residuals squared)
    # We bootstrap these errors to preserve the correlation structure
    err_model = (y_true - y_pred) ** 2
    err_baseline = (y_true - baseline_pred) ** 2

    for _ in range(n_bootstraps):
        # 2. Generate Block Indices
        indices = []
        while len(indices) < n_samples:
            start_idx = np.random.randint(0, n_samples - block_size + 1)
            # Select a contiguous block
            indices.extend(range(start_idx, start_idx + block_size))
        
        # Trim to original length
        indices = np.array(indices[:n_samples])

        # 3. Calculate R2 for this specific bootstrap sample
        # Sum of squared errors for the sample
        sse_model = np.sum(err_model[indices])
        sse_baseline = np.sum(err_baseline[indices])
        
        # Avoid division by zero
        if sse_baseline > 1e-9:
            r2_sample = 1 - (sse_model / sse_baseline)
        else:
            r2_sample = np.nan
            
        r2_bootstraps.append(r2_sample)

    # 4. Aggregate Statistics
    r2_bootstraps = np.array([x for x in r2_bootstraps if not np.isnan(x)])
    
    if len(r2_bootstraps) == 0:
        return {"std": np.nan, "lower": np.nan, "upper": np.nan}

    return {
        "mean": np.mean(r2_bootstraps),
        "std": np.std(r2_bootstraps),
        "lower": np.percentile(r2_bootstraps, 2.5),   # 95% Confidence Interval
        "upper": np.percentile(r2_bootstraps, 97.5)
    }



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
    


from collections import Counter
import numpy as np

def baseline_classification(y_true, mode: str = "majority"):
    """
    Time-series-safe baselines for classification.

    Parameters
    ----------
    y_true : 1D array-like of OOS labels in chronological order
    mode   : "majority" | "persistence"

    Returns
    -------
    baseline : 1D array of baseline predictions, same length as y_true
    """

    y_true = np.asarray(y_true)
    n = len(y_true)
    baseline = np.empty_like(y_true, dtype=y_true.dtype)

    if mode == "majority":
        # expanding majority using ONLY past labels:
        # for t=0 we can't compute a past majority, so just use y_true[0]
        for i in range(n):
            if i == 0:
                baseline[i] = y_true[0]
            else:
                counts = Counter(y_true[:i])   # past only, y_true[0..i-1]
                baseline[i] = counts.most_common(1)[0][0]
        return baseline

    elif mode == "persistence":
        # persistence: y_hat_t = y_{t-1}, with y_hat_0 = y_true[0] by convention
        baseline[0] = y_true[0]
        if n > 1:
            baseline[1:] = y_true[:-1]
        return baseline

    else:
        raise ValueError(f"Unknown baseline mode: {mode}")



def evaluate_oos(
    y_true, 
    y_pred, 
    model_name="Model", 
    device="cpu", 
    quiet=False, 
    mode: str = "mean"
):
    """
    Compute MSE, RMSE and out-of-sample R² (Campbell–Thompson style)
    using the expanding mean as the benchmark forecast.
    
    Returns: (r2_oos_point_estimate, bootstrap_stats_dict)
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    # Filter NaNs
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]

    if len(y_true) == 0:
        if not quiet:
            print(f"[{model_name}] No valid predictions (all NaN).")
        return np.nan, {}

    # 1. Standard Point Estimates
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    # Generate the Baseline Prediction Vector
    # (Assuming baseline_forecast returns an array of shape y_true)
    mean_forecast = baseline_forecast(y_true, mode=mode)
    
    denom = np.sum((y_true - mean_forecast) ** 2)
    
    if denom > 0:
        r2_oos = float(1 - np.sum((y_true - y_pred) ** 2) / denom)
    else:
        r2_oos = np.nan

    # 2. Calculate Bootstrap Statistics (New Part)
    stats = calculate_block_bootstrap_stats(
        y_true, 
        y_pred, 
        mean_forecast, 
        block_size=12, 
        n_bootstraps=1000
    )

    if not quiet:
        print(f"[{model_name}] Valid={len(y_true)} | "
              f"MSE={mse:.6f} | "
              f"R²_OS={r2_oos:.4f} (±{stats['std']:.4f})")

    # Return both the point estimate and the stats dictionary
    return r2_oos, stats
# def evaluate_oos(y_true, y_pred, model_name="Model", device="cpu", quiet=False, mode: str = "mean",):
#     """
#     Compute MSE, RMSE and out-of-sample R² (Campbell–Thompson style)
#     using the expanding mean as the benchmark forecast.
#     """
#     y_true = np.asarray(y_true, float)
#     y_pred = np.asarray(y_pred, float)

#     m = ~np.isnan(y_true) & ~np.isnan(y_pred)
#     y_true, y_pred = y_true[m], y_pred[m]

#     if len(y_true) == 0:
#         if not quiet:
#             print(f"[{model_name}] No valid predictions (all NaN).")
#         return np.nan

#     mse = mean_squared_error(y_true, y_pred)
#     rmse = float(np.sqrt(mse))

#     mean_forecast = baseline_forecast(y_true, mode=mode)
#     denom = np.sum((y_true - mean_forecast) ** 2)
#     r2_oos = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else np.nan

#     if not quiet:
#         print(f"[{model_name}] Device={device} | Valid months={len(y_true)} | "
#               f"MSE={mse:.6f} | RMSE={rmse:.6f} | R²_OS={r2_oos:.4f}")

#     return r2_oos
def evaluate_oos_classification(
    y_true,
    y_pred,
    model_name: str = "Model",
    baseline_mode: str = "majority",
    quiet: bool = False,
):
    """
    OOS evaluation for Bull/Bear classification.

    Parameters
    ----------
    y_true        : 1D array of true labels (e.g. 0/1 or "Bear"/"Bull")
    y_pred        : 1D array of model predictions (same type as y_true)
    baseline_mode : "majority" or "persistence"
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # mask NaNs (works if labels are numeric or you used np.nan for missing)
    mask = ~np.isnan(y_true) if np.issubdtype(y_true.dtype, np.number) else np.ones_like(y_true, bool)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        if not quiet:
            print(f"[{model_name}] No valid predictions (all NaN).")
        return {
            "acc": np.nan,
            "bal_acc": np.nan,
            "acc_baseline": np.nan,
            "bal_acc_baseline": np.nan,
            "skill_acc": np.nan,
        }

    # model performance
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # baseline performance
    y_base = baseline_classification(y_true, mode=baseline_mode)
    acc_base = accuracy_score(y_true, y_base)
    bal_acc_base = balanced_accuracy_score(y_true, y_base)

    # skill score (analogue of R² relative to baseline accuracy)
    skill_acc = (acc - acc_base) / (1 - acc_base) if acc_base < 1.0 else np.nan

    if not quiet:
        print(
            f"[{model_name}] Valid obs={len(y_true)} | "
            f"Acc={acc:.4f} (baseline={acc_base:.4f}) | "
            f"BalAcc={bal_acc:.4f} (baseline={bal_acc_base:.4f}) | "
            f"Skill_Acc={skill_acc:.4f}"
        )

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "acc_baseline": acc_base,
        "bal_acc_baseline": bal_acc_base,
        "skill_acc": skill_acc,
    }


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
        print( date_t)
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
        print(date_t)
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


def expanding_oos_tabular_cls(
    data: pd.DataFrame,
    target_col: str = "state",
    start_oos: str = "2007-01-01",
    start_date: str = "2000-01-05",
    min_train: int = 120,
    min_history_months: int | None = None,
    quiet: bool = False,
    model_name: str = "Logit-lag-baseline",
    model_fit_predict_fn: Callable[[pd.DataFrame, pd.Series], float] | None = None,
    baseline_mode: str = "majority",   # for evaluate_oos_classification
):
    """
    Expanding-window 1-step-ahead OOS driver for CLASSIFICATION.
    Returns: (metrics_dict, y_true, y_pred, oos_dates)
    """
    
    if model_fit_predict_fn is None:
        raise ValueError("You must supply model_fit_predict_fn.")
    data.index = pd.to_datetime(data.timestamp, format = "%Y-%m-%d")
    df = data.copy()
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in data.")

    if min_history_months is not None:
        start_ts = expand_start_with_min_history(df.index, start_oos, min_history_months)
    else:
        start_ts = pd.Timestamp(start_oos)

    loop_dates = df.index[df.index >= start_ts]

    preds, trues, oos_dates = [], [], []

    for date_t in loop_dates:
        print( date_t)
        pos = df.index.get_loc(date_t)
        est = df.iloc[:pos].copy()      # strictly past
        row_t = df.iloc[pos]

        # require enough past non-missing target for a meaningful model
        if est[target_col].notna().sum() < min_train:
            continue

        y_true = row_t[target_col]
        if pd.isna(y_true):
            continue

        y_hat = model_fit_predict_fn(est, row_t)
        if y_hat is None or (isinstance(y_hat, float) and np.isnan(y_hat)):
            continue

        trues.append(int(y_true))
        preds.append(int(y_hat))
        oos_dates.append(date_t)

    if not preds:
        raise RuntimeError(f"[{model_name}] No valid predictions produced.")

    trues = np.asarray(trues, int)
    preds = np.asarray(preds, int)
    oos_dates = pd.DatetimeIndex(oos_dates)

    metrics = evaluate_oos_classification(
        trues,
        preds,
        model_name=model_name,
        baseline_mode=baseline_mode,
        quiet=quiet,
    )

    return metrics, trues, preds, oos_dates
