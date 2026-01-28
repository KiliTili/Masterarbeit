from math import inf
from typing import Callable, Sequence, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import os, sys
sys.path.insert(0, os.path.abspath('../'))

from source.regression.modelling_utils import evaluate_oos,ensure_datetime_index,align_monthly, expand_start_with_min_history
import torch
def expanding_oos_univariate_multistep(
    y: pd.Series,
    start_oos: str = "1965-01-01",
    prediction_length: int = 12,
    min_history_months: int = 240,
    ct_cutoff: bool = False,
    quiet: bool = False,
    model_name: str = "TS-Model",
    forecast_multi_step_fn: Callable[[pd.Series, pd.Timestamp, int], np.ndarray] | None = None,
) -> tuple[
    dict[int, float],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, pd.DatetimeIndex],
]:
    """
    Multi-step expanding-window OOS driver.

    forecast_multi_step_fn(y_hist, origin_date, H) -> array of shape (H,)

    Returns dicts keyed by horizon h=1..H:
        r2[h]    : scalar R²_OS for horizon h
        trues[h] : np.array of true values for horizon h
        preds[h] : np.array of predictions for horizon h
        dates[h] : DatetimeIndex of evaluation dates for horizon h
    """
    if forecast_multi_step_fn is None:
        raise ValueError("forecast_multi_step_fn must be provided.")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be >= 1")

    y = y.astype("float32").dropna()
    if y.empty:
        raise ValueError("Target series is empty after cleaning.")

    # ensure enough past data before first origin
    start_ts = expand_start_with_min_history(
        y.index, start_oos, min_history_months=min_history_months
    )

    # also need enough future data to evaluate all horizons
    last_valid_origin = y.index[-prediction_length]  # t such that t+H-1 exists
    test_idx = y.index[(y.index >= start_ts) & (y.index <= last_valid_origin)]

    if not quiet and len(test_idx) > 0:
        print(f"[{model_name}] H={prediction_length}, origins={len(test_idx)}, "
              f"first_origin={test_idx[0].date()}, last_origin={test_idx[-1].date()}")

    preds = {h: [] for h in range(1, prediction_length + 1)}
    trues = {h: [] for h in range(1, prediction_length + 1)}
    dates = {h: [] for h in range(1, prediction_length + 1)}

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
            y_hat = float(y_hat_vec[h - 1])
            if np.isnan(y_true) or np.isnan(y_hat):
                continue
            if ct_cutoff:
                y_hat = float(ct_truncate(y_hat))
            trues[h].append(y_true)
            preds[h].append(y_hat)
            dates[h].append(y.index[target_pos])

    # convert to arrays & compute R² per horizon
    r2 = {}
    for h in range(1, prediction_length + 1):
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
            )

    return r2, trues, preds, dates


from chronos import BaseChronosPipeline

def chronos_oos_multistep(
    data: pd.DataFrame,
    target_col: str = "equity_premium",
    start_oos: str = "1965-01-01",
    freq: str = "MS",
    prediction_length: int = 12,
    ct_cutoff: bool = False,
    quiet: bool = False,
):
    """
    Multi-step expanding-window OOS for Chronos-Bolt.

    Returns:
        r2_dict, trues_dict, preds_dict, dates_dict
    where keys are horizons h=1..prediction_length.
    """
    df = ensure_datetime_index(data)
    y = align_monthly(df[[target_col]], freq, col=target_col)[target_col]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    def forecast_multi_step(y_hist: pd.Series, date_t, H: int) -> np.ndarray:
        ctx = y_hist.to_numpy(dtype="float32")
        if len(ctx) < 24:
            return np.full(H, np.nan, dtype="float32")
        with torch.inference_mode():
            _, mean_pred = pipe.predict_quantiles(
                context=[torch.tensor(ctx, device=device)],
                prediction_length=H,
                quantile_levels=[0.5],
            )
        # mean_pred shape: (batch, H)
        return np.asarray(mean_pred[0, :H], dtype="float32")

    r2, trues, preds, dates = expanding_oos_univariate_multistep(
        y,
        start_oos=start_oos,
        prediction_length=prediction_length,
        min_history_months=240,  # 20y history before first origin
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name="Chronos-Bolt",
        forecast_multi_step_fn=forecast_multi_step,
    )
    return r2, trues, preds, dates
