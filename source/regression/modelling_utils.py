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
import random, numpy as np, torch

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_global_seed(42)
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
        b = np.empty_like(y_true)
        b[0] = 0 # Kein Forecast für den allerersten Punkt möglich ohne Historie
        
        # Für t=1 bis Ende: Mean von 0 bis t-1
        # y_true[:i] geht von 0 bis i-1. Das ist korrekt für den Forecast an Stelle i.
        b[1:] = [y_true[:i].mean() for i in range(1, len(y_true))]
        return b
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
    y_bench,
    model_name="Model", 
    quiet=False
):
    """
    Compute MSE, RMSE and out-of-sample R² (Campbell–Thompson style)
    using the expanding mean as the benchmark forecast.
    
    Returns: (r2_oos_point_estimate, bootstrap_stats_dict)
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    # Filter NaNs
    m = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(y_bench)    
    
    y_true, y_pred, y_bench = y_true[m], y_pred[m], y_bench[m]

    if len(y_true) == 0:
        if not quiet:
            print(f"[{model_name}] No valid predictions (all NaN).")
        return np.nan, {}

    # 1. Standard Point Estimates
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    # Generate the Baseline Prediction Vector
    # (Assuming baseline_forecast returns an array of shape y_true)
    #mean_forecast = baseline_forecast(y_true, mode=mode)
    
    denom = np.sum((y_true - y_bench) ** 2)
    
    if denom > 0:
        r2_oos = float(1 - np.sum((y_true - y_pred) ** 2) / denom)
    else:
        r2_oos = np.nan

    # 2. Calculate Bootstrap Statistics (New Part)
    stats = calculate_block_bootstrap_stats(
        y_true, 
        y_pred, 
        y_bench, 
        block_size=12, 
        n_bootstraps=1000
    )

    if not quiet:
        print(f"[{model_name}] Valid={len(y_true)} | "
              f"MSE={mse:.6f} | "
              f"R²_OS={r2_oos:.4f} (±{stats['std']:.4f})")

    # Return both the point estimate and the stats dictionary
    return r2_oos, stats

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
            f"SS={skill_acc:.4f}"
        )

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "acc_baseline": acc_base,
        "bal_acc_baseline": bal_acc_base,
        "skill_acc": skill_acc,
    }


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

    preds,preds_before_ct, trues, oos_dates, HA, y_lowers, y_uppers = [], [],[], [], [], [], []
    truncated = 0
    for date_t in loop_dates:
        if not quiet:
            print(date_t)
        pos = df.index.get_loc(date_t)
        est = df.iloc[:pos].copy()        
        row_t = df.iloc[pos]

        if est[target_col].notna().sum() < min_train:
            continue

        y_true = float(row_t[target_col])
        if np.isnan(y_true):
            continue

        #y_hat = model_fit_predict_fn(est, row_t)
        results = model_fit_predict_fn(est, row_t)
        if isinstance(results, (list, tuple)):
            y_hat, y_lower, y_upper = results
        else:
            y_hat = results
            y_lower = None
            y_upper = None

        if y_hat is None or np.isnan(y_hat):
            continue
        preds_before_ct.append(float(y_hat))
        if ct_cutoff:
            if y_hat < 0:
                truncated += 1
            y_hat = float(ct_truncate(y_hat))


        preds.append(float(y_hat))
        trues.append(y_true)
        oos_dates.append(date_t)
        ha_t = float(est[target_col].dropna().mean()) 
        HA.append(ha_t)      
        y_lowers.append(y_lower)
        y_uppers.append(y_upper)

    if not preds:
        raise RuntimeError(f"[{model_name}] No valid predictions produced.")

    
    print(f"percentage of negative forecasts before truncation: {truncated/len(preds)*100:.2f}%")
        

     # Manually calculate R2 for verification
    trues = np.asarray(trues, float)
    preds = np.asarray(preds, float)        
    HA = np.asarray(HA, float)
    r2_oos = 1 - np.sum((trues - preds)**2) / np.sum((trues - HA)**2)
    print(f"Manually calculated R2: {r2_oos}")
    r2,stats = evaluate_oos(trues, preds, y_bench=HA, model_name=model_name, quiet=quiet)
    print(f"evaluate_oos calculated R2 CT: {r2}")
    print(f"Manually calculated Stats: {stats}")
    r2_wct, stats_wct = evaluate_oos(trues, preds_before_ct, y_bench=HA, model_name=model_name+" (WCT)", quiet=quiet)
    print(f"evaluate_oos calculated R2 WCT: {r2_wct}")
    print(f"Stats WCT: {stats_wct}")
    
    r2,stats = evaluate_oos(trues, preds, y_bench=HA, model_name=model_name, quiet=quiet)
    return r2, stats, trues, preds, pd.DatetimeIndex(oos_dates), y_lowers, y_uppers, HA