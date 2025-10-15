from math import inf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from chronos import BaseChronosPipeline
import torch
import timesfm

def evaluate_oos(y_true, y_pred, model_name="Model", device="cpu", quiet=False):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) == 0:
        if not quiet:
            print(f"[{model_name}] No valid predictions (all NaN).")
        return np.nan
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mean_forecast = np.array([y_true[:i].mean() for i in range(1, len(y_true)+1)])
    r2_oos = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - mean_forecast)**2)
    if not quiet:
        print(f"[{model_name}] Device={device} | Valid months={len(y_true)} | "
              f"MSE={mse:.6f} | RMSE={rmse:.6f} | R²_OS={r2_oos:.4f}")
    return r2_oos

def linear_regression_oos(
    data,
    variables=['d/p'],
    start_oos='1965-01-01',
    device='cpu',
    quiet=False,
    lag=1,
    start_date='1927-01-01'
):
    """
    Expanding-window OLS with *lagged* predictors that matches the manual script:
    - uses shift(1) when lag=1
    - no global median imputation
    - drops NaNs in the estimation window
    """
    df = data.copy()

    # ensure DatetimeIndex and date filtering
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

    # create lagged features: 1..lag (NOT 0..lag-1)
    for L in range(1, lag + 1):
        for v in variables:
            df[f"{v}_lag{L}"] = df[v].shift(L)

    feature_cols = [f"{v}_lag1" for v in variables] if lag == 1 \
                   else [f"{v}_lag{L}" for v in variables for L in range(1, lag+1)]

    start_oos = pd.Timestamp(start_oos)
    predictions, actuals = [], []

    for date_t in df.index:
        if date_t < start_oos:
            continue

        # estimation window: up to but excluding date_t
        est = df.loc[:date_t].iloc[:-1]

        # drop NaNs exactly like your script effectively does (no imputation)
        est = est.dropna(subset=feature_cols + ['equity_premium'])
        if len(est) < 30:
            continue

        X_train = est[feature_cols].to_numpy()
        y_train = est['equity_premium'].to_numpy()

        # one-step-ahead features; skip if they are NaN (e.g., at the start)
        x_pred = df.loc[date_t, feature_cols].to_numpy(dtype=float).reshape(1, -1)
        if np.isnan(x_pred).any():
            continue

        model = LinearRegression().fit(X_train, y_train)
        pred = np.max(model.predict(x_pred)[0],0)

        predictions.append(pred)
        actuals.append(df.loc[date_t, 'equity_premium'])

    return evaluate_oos(actuals, predictions, model_name=f"OLS({','.join(variables)})", device=device, quiet=quiet)



# --- Driver to evaluate many monthly variables with your function ---


def rank_monthly_predictors(
    data,
    monthly_vars,
    start_date="1927-01-01",
    start_oos="1965-01-01",
    lag=1,
    quiet=True,
):
    """
    Calls your linear_regression_oos once per variable, collects OOS R²,
    and prints a worst->best ranking. Returns a DataFrame with results.
    """
    results = []
    for v in monthly_vars:
        try:
            r2 = linear_regression_oos(
                data,
                variables=[v],
                start_oos=start_oos,
                device="cpu",
                quiet=quiet,   # suppress per-variable prints
                lag=lag,
                start_date=start_date,
            )
        except Exception as e:
            # capture failures as NaN with an error message
            r2 = float("nan")
            print(f"[WARN] {v}: {e}")

        results.append({"variable": v, "r2_oos": r2})

    res_df = pd.DataFrame(results)

    # For sorting: treat NaN as -inf (worst)
    sort_key = res_df["r2_oos"].fillna(-inf)
    res_df = res_df.loc[sort_key.sort_values(ascending=True).index].reset_index(drop=True)

    # Pretty print worst -> best
    print("\nMonthly predictors ranked (worst → best) by OOS R²:")
    for i, row in res_df.iterrows():
        r2 = row["r2_oos"]
        r2_str = "NaN" if pd.isna(r2) else f"{r2:.4f}"
        print(f"{i+1:2d}. {row['variable']:>10s}   R²_OOS = {r2_str}")

    return res_df








def chronos_oos( data,
    start_oos='1965-01-01',
    quiet=False,
    lag=1,
    start_date='1927-01-01'):
    print("Starting Chronos OOS evaluation...")
    df = data.sort_index()[['equity_premium']].dropna().asfreq('MS')
    print(f"Data from {df.index[0]} to {df.index[-1]}, {len(df)} months total.")
    y = df['equity_premium']

    # --- Chronos model ---
    print("Loading Chronos model...")
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    print("Chronos model loaded.")
    # --- Forecast setup ---
    test_idx = y.index[y.index >= start_oos]
    preds, trues = [], []

    # --- One-step-ahead forecasting ---
    for date_t in test_idx:
        pos = y.index.get_loc(date_t)
        if pos < 24:  # need at least 2 years of data before t
            continue

        context_vals = y.iloc[:pos].to_numpy(dtype="float32")
        if np.isnan(context_vals).any() or len(context_vals) == 0:
            continue

        context = torch.tensor(context_vals)

        with torch.inference_mode():
            _, mean_pred = pipe.predict_quantiles(
                context=[context],
                prediction_length=1,
                quantile_levels=[0.5],
            )

        y_hat = np.max(float(mean_pred[0, 0]),0)
        y_true = float(y.iloc[pos])
        if np.isnan(y_hat) or np.isnan(y_true):
            continue

        preds.append(y_hat)
        trues.append(y_true)

    # --- Safety check before evaluation ---
    if len(preds) == 0:
        raise RuntimeError("No valid Chronos predictions were generated. Check that your data has enough history before 1965.")

    # --- Evaluate ---
    preds, trues = np.array(preds), np.array(trues)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mean_forecast = np.array([trues[:i].mean() for i in range(1, len(trues)+1)])
    r2_oos = 1 - np.sum((trues - preds)**2) / np.sum((trues - mean_forecast)**2)

    print(f"[Chronos-Bolt] OOS months: {len(preds)}  MSE={mse:.6f}  RMSE={rmse:.6f}  R²_OS={r2_oos:.4f}")




def timesfm_oos(
    data,
    start_oos="1965-01-01",
    min_context=120,          # require at least 10 years of history
    max_context=1024,         # TimesFM context length
    ct_cutoff=True,           # apply Campbell–Thompson cutoff at 0
    quiet=False,
):
    """
    Expanding-window one-step-ahead OOS evaluation using Google TimesFM (PyTorch port).
    - Monthly equity premium (column: 'equity_premium'), MS frequency
    - Uses last `max_context` months as input context
    - Optional Campbell–Thompson cutoff (clip negative forecasts at 0)
    - Returns OOS R² from evaluate_oos(...)
    """

    # -----------------------------
    # 1) Device detection
    # -----------------------------
    torch.set_float32_matmul_precision("high")
    if torch.backends.mps.is_available():
        device = "mps"     # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = "cuda"    # NVIDIA GPU
    else:
        device = "cpu"
    if not quiet:
        print(f"[TimesFM] Using device: {device}")

    # -----------------------------
    # 2) Data preparation
    # -----------------------------
    df = data.copy()
    # enforce monthly start-of-month frequency and clean
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index().asfreq("MS")
    df = df[["equity_premium"]].dropna()
    y = df["equity_premium"].astype("float32")

    if len(y) == 0:
        raise ValueError("No equity_premium data after cleaning.")

    # -----------------------------
    # 3) Load TimesFM model
    # -----------------------------
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    cfg = timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=1,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
    model.compile(cfg)

    # -----------------------------
    # 4) Forecast setup (expanding window)
    # -----------------------------
    start_oos = pd.Timestamp(start_oos)
    test_idx = y.index[y.index >= start_oos]
    preds, trues = [], []

    for date_t in test_idx:
        pos = y.index.get_loc(date_t)
        if pos < min_context:
            continue

        # expanding window: all data up to t (exclude t)
        context = y.iloc[:pos].to_numpy(dtype="float32")

        # truncate to last max_context
        if len(context) > cfg.max_context:
            context = context[-cfg.max_context:]

        # skip invalid contexts
        if np.isnan(context).any() or np.std(context) < 1e-6:
            continue

        # one-step ahead forecast
        with torch.inference_mode():
            point_fcst, _ = model.forecast(horizon=1, inputs=[context])

        y_hat = float(point_fcst[0, 0])
        if ct_cutoff:
            y_hat = max(y_hat, 0.0)  # Campbell–Thompson cutoff

        y_true = float(y.iloc[pos])
        if np.isnan(y_hat) or np.isnan(y_true):
            continue

        preds.append(y_hat)
        trues.append(y_true)

    # -----------------------------
    # 5) Evaluation (reuse your helper)
    # -----------------------------
    if len(preds) == 0:
        raise RuntimeError(
            "No valid TimesFM predictions were generated. "
            "Check MIN_CONTEXT / start_oos / data length."
        )

    # use your existing evaluate_oos for consistent printing/return
    return evaluate_oos(trues, preds, model_name="TimesFM", device=device, quiet=quiet)





    #missing: CT truncation, 20 years after the series begins (≥1946), they recompute any filter/coefficients expanding in time.