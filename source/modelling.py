from math import inf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from chronos import BaseChronosPipeline
import torch
import timesfm
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_oos(
    y_true,
    y_pred,
    dates=None,                 # optional: pandas.DatetimeIndex or list-like of same length
    title="Out-of-sample 1-step forecast",
    ylabel="Equity premium",
    save_path=None,             # optional: path to save the figure
    show=True,
):
    """
    Plots true values, 1-step predictions, and the expanding-mean baseline
    (the same mean used in evaluate_oos).

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    dates  : array-like of timestamps, optional
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # mask any NaNs in either array (should be rare after your evaluate function)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]

    # expanding-mean baseline (same as in evaluate_oos)
    # fast cumulative mean: mean_t = cumsum[:t] / t
    csum = np.cumsum(y_true)
    mean_forecast = csum / np.arange(1, len(y_true) + 1)

    # build x-axis
    if dates is not None:
        x = pd.to_datetime(pd.Index(dates))[m]
    else:
        x = np.arange(len(y_true))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_true, label="True")
    ax.plot(x, y_pred, label="Prediction")
    ax.plot(x, mean_forecast, label="Expanding mean", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date" if dates is not None else "OOS step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


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
    Expanding-window OLS with lagged predictors.
    Now returns (r2_oos, y_true, y_pred, dates) for plotting.
    """
    df = data.copy()

    # ensure DatetimeIndex and date filtering
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

    # create lagged features: 1..lag
    for L in range(1, lag + 1):
        for v in variables:
            df[f"{v}_lag{L}"] = df[v].shift(L)

    feature_cols = [f"{v}_lag1" for v in variables] if lag == 1 \
                   else [f"{v}_lag{L}" for v in variables for L in range(1, lag+1)]

    start_oos = pd.Timestamp(start_oos)
    predictions, actuals, dates = [], [], []

    for date_t in df.index:
        if date_t < start_oos:
            continue

        # estimation window: up to but excluding date_t
        est = df.loc[:date_t].iloc[:-1]

        # drop NaNs exactly like before
        est = est.dropna(subset=feature_cols + ['equity_premium'])
        if len(est) < 30:
            continue

        X_train = est[feature_cols].to_numpy()
        y_train = est['equity_premium'].to_numpy()

        # one-step-ahead features
        x_pred = df.loc[date_t, feature_cols].to_numpy(dtype=float).reshape(1, -1)
        if np.isnan(x_pred).any():
            continue

        model = LinearRegression().fit(X_train, y_train)
        pred = float(model.predict(x_pred)[0])
        pred = max(pred, 0.0)  # keep your truncation, if you want it (or remove)

        y_t = float(df.loc[date_t, 'equity_premium'])

        predictions.append(pred)
        actuals.append(y_t)
        dates.append(date_t)

    r2 = evaluate_oos(actuals, predictions,
                      model_name=f"OLS({','.join(variables)})",
                      device=device, quiet=quiet)

    # return the traces for plotting
    return r2, np.asarray(actuals, float), np.asarray(predictions, float), pd.DatetimeIndex(dates)



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
            r2,_,_,_ = linear_regression_oos(
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







def chronos_oos(
    data,
    start_oos="1965-01-01",
    quiet=False,
    ct_cutoff=True,          # Campbell–Thompson cutoff at 0
):
    # ---------- data ----------
    df = data.sort_index()[["equity_premium"]].dropna().asfreq("MS")
    y = df["equity_premium"].astype("float32")
    start_oos = pd.Timestamp(start_oos)
    test_idx = y.index[y.index >= start_oos]

    if not quiet:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Chronos] data {df.index[0].date()}→{df.index[-1].date()}  n={len(df)}  device={dev}")

    # ---------- model ----------
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # ---------- expanding OOS ----------
    preds, trues, dates = [], [], []
    for date_t in test_idx:
        pos = y.index.get_loc(date_t)
        if pos < 24:       # need some history
            continue

        ctx = y.iloc[:pos].to_numpy(dtype="float32")
        if np.isnan(ctx).any() or len(ctx) == 0:
            continue

        with torch.inference_mode():
            _, mean_pred = pipe.predict_quantiles(
                context=[torch.tensor(ctx)],
                prediction_length=1,
                quantile_levels=[0.5],
            )

        y_hat = float(mean_pred[0, 0])
        if ct_cutoff:
            y_hat = max(y_hat, 0.0)

        y_true = float(y.iloc[pos])
        if np.isnan(y_hat) or np.isnan(y_true):
            continue

        preds.append(y_hat)
        trues.append(y_true)
        dates.append(date_t)

    if len(preds) == 0:
        raise RuntimeError("No valid Chronos predictions. Ensure enough pre-1965 history.")

    # ---------- evaluate & return traces for plotting ----------
    r2 = evaluate_oos(trues, preds, model_name="Chronos-Bolt", device=("cuda" if torch.cuda.is_available() else "cpu"), quiet=quiet)
    return r2, np.asarray(trues, float), np.asarray(preds, float), pd.DatetimeIndex(dates)


def timesfm_oos(
    data,
    start_oos="1965-01-01",
    min_context=120,          # require at least 10 years of history
    max_context=512,         # TimesFM context length
    ct_cutoff=True,           # Campbell–Thompson cutoff at 0
    quiet=False,
):
    """
    Expanding-window one-step-ahead OOS using Google TimesFM (PyTorch port).
    Returns: (r2, y_true, y_pred, dates) for plotting.
    """

    # 1) Device note (TimesFM torch port takes NumPy; runs on CPU unless tensors are moved)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if not quiet:
        print(f"[TimesFM] Using device hint: {device}")

    # 2) Data prep
    df = data.sort_index()[["equity_premium"]].dropna().asfreq("MS")
    y = df["equity_premium"].astype("float32")
    if len(y) == 0:
        raise ValueError("No equity_premium data after cleaning.")

    # 3) Load TimesFM model
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    cfg = timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=128,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
    model.compile(cfg)

    # 4) Expanding OOS loop
    start_oos = pd.Timestamp(start_oos)
    test_idx = y.index[y.index >= start_oos]
    preds, trues, oos_dates = [], [], []

    for date_t in test_idx:
        pos = y.index.get_loc(date_t)
        if pos < min_context:
            print("skipping")
            continue

        context = y.iloc[:pos].to_numpy(dtype="float32")
        if len(context) > cfg.max_context:
            context = context[-cfg.max_context:]
        if np.isnan(context).any() or np.std(context) < 1e-6:
            continue

        with torch.inference_mode():
            point_fcst, _ = model.forecast(horizon=1, inputs=[context])

        y_hat = float(point_fcst[0, 0])
        if ct_cutoff:
            y_hat = max(y_hat, 0.0)

        y_true = float(y.iloc[pos])
        if np.isnan(y_hat) or np.isnan(y_true):
            continue

        preds.append(y_hat)
        trues.append(y_true)
        oos_dates.append(date_t)

    if len(preds) == 0:
        raise RuntimeError("No valid TimesFM predictions were generated. Check min_context/start_oos/data length.")

    r2 = evaluate_oos(trues, preds, model_name="TimesFM", device=device, quiet=quiet)
    return r2, np.asarray(trues, float), np.asarray(preds, float), pd.DatetimeIndex(oos_dates)


def tree_ensemble_oos(
    data,
    variables,
    start_oos="1965-01-01",
    start_date="1927-01-01",
    lag=1,
    min_train=120,
    ct_cutoff=True,
    quiet=False,
    model_params=None,
    drop_sparse=True,          # drop too-sparse features per window
    sparse_thresh=0.6,         # require >=60% non-missing in the training window
):
    import numpy as np
    import pandas as pd

    # pick model
    if model_params is None:
        model_params = {}
    try:
        from xgboost import XGBRegressor
        use_xgb = True
        default_params = dict(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="reg:squarederror", random_state=42,
        )
        default_params.update(model_params)
        def Model(): return XGBRegressor(**default_params)
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        use_xgb = False
        default_params = dict(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            subsample=0.8, random_state=42,
        )
        default_params.update(model_params)
        def Model(): return GradientBoostingRegressor(**default_params)

    # data prep
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()
    if "equity_premium" not in df.columns:
        raise ValueError("data must contain 'equity_premium' column.")

    # lag features
    for L in range(1, lag + 1):
        for v in variables:
            df[f"{v}_lag{L}"] = df[v].shift(L)
    feature_cols_all = [f"{v}_lag{L}" for v in variables for L in range(1, lag + 1)]
    start_oos = pd.Timestamp(start_oos)

    preds, trues, oos_dates = [], [], []
    loop_dates = df.index[df.index >= start_oos]

    for date_t in loop_dates:
        pos = df.index.get_loc(date_t)
        est = df.iloc[:pos].copy()  # strictly past

        # choose candidate features (optionally drop very sparse in the training window)
        feats = feature_cols_all
        if drop_sparse:
            avail = est[feats].notna().mean(axis=0)
            feats = [c for c in feats if avail.get(c, 0.0) >= sparse_thresh]
            if len(feats) == 0:
                continue

        # need at least min_train rows of y, regardless of NaNs in X
        y_est = est["equity_premium"]
        if y_est.notna().sum() < min_train:
            continue

        # compute training-window medians for features (past-only)
        med = est[feats].median(skipna=True)

        # impute features in training window with medians (no look-ahead)
        X_train = est[feats].fillna(med).to_numpy()
        y_train = y_est.to_numpy()

        # drop rows where y is NaN (X already imputed)
        m = ~np.isnan(y_train)
        X_train, y_train = X_train[m], y_train[m]
        if len(y_train) < min_train:
            continue

        # prepare x_pred: past-only ffill then fallback to training median
        x_pred_series = df.loc[date_t, feats].copy()
        past_vals = est[feats].iloc[-1:].ffill().iloc[0]      # ffill from past
        x_pred_series = x_pred_series.fillna(past_vals)
        x_pred_series = x_pred_series.fillna(med)              # fallback to training medians
        x_pred = x_pred_series.to_numpy(dtype=float).reshape(1, -1)
        if np.isnan(x_pred).any():
            continue

        # fit & predict
        model = Model()
        model.fit(X_train, y_train)
        y_hat = float(model.predict(x_pred)[0])
        if ct_cutoff:
            y_hat = max(y_hat, 0.0)

        y_true = float(df.loc[date_t, "equity_premium"])
        if np.isnan(y_true):
            continue

        preds.append(y_hat)
        trues.append(y_true)
        oos_dates.append(date_t)

    if len(preds) == 0:
        raise RuntimeError("No valid predictions; loosen sparsity threshold, reduce vars, or lower min_train.")

    name = "XGB" if use_xgb else "GBRT"
    r2 = evaluate_oos(trues, preds, model_name=f"{name}({','.join(variables)})", device="cpu", quiet=quiet)
    return r2, np.asarray(trues, float), np.asarray(preds, float), pd.DatetimeIndex(oos_dates)





def moirai2_retrain_each_step(
    data: pd.DataFrame,
    covariates=("d/p", "tms", "dfy"),
    start_oos="1965-01-01",
    ctx=240,
    device="cpu",
    ct_cutoff=False,
    quiet=False,
    model_name="Moirai 2 (reinstantiated each step)",
    FREQ_STR="M",  # month-end; avoids pandas <MonthBegin> issue in GluonTS 0.14.x
):
    """
    Expanding-window, one-step-ahead forecasting with Moirai-2.
    Returns: (r2_oos, y_true, y_pred, dates) for plotting.
    """
    if not quiet:
        print(f"[moirai2] Using freq='{FREQ_STR}' (month-end) | ctx={ctx}")

    # --- Data prep: coerce index to month-end ---
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period(FREQ_STR).to_timestamp(FREQ_STR)
    df = df.sort_index().asfreq(FREQ_STR)

    needed = ["equity_premium"] + list(covariates)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[needed].dropna()
    y = df["equity_premium"].astype("float32")
    covs = [df[c].astype("float32") for c in covariates]

    start_oos = pd.Timestamp(start_oos).to_period(FREQ_STR).to_timestamp(FREQ_STR)
    test_idx = y.index[y.index >= start_oos]

    preds, trues, oos_dates = [], [], []

    # Reuse module across steps
    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")

    # --- Helper: dataset up to (t-1) ---
    def make_entry(end_ts: pd.Timestamp):
        pos = y.index.get_loc(end_ts)
        if isinstance(pos, slice):
            pos = pos.start
        if pos <= 0:
            return None

        y_hist = y.values[:pos]
        if y_hist.size == 0:
            return None

        # keep last ctx points
        if len(y_hist) > ctx:
            y_hist = y_hist[-ctx:]
            start_idx = pos - len(y_hist)
        else:
            start_idx = 0

        entry = {
            "start": pd.Timestamp(y.index[start_idx]),
            "target": y_hist.astype("float32"),
        }

        if len(covs) > 0:
            mats = [c.values[:pos] for c in covs]
            mats = [m[-len(y_hist):] for m in mats]  # align to target len
            entry["past_feat_dynamic_real"] = np.vstack(mats).astype("float32")  # (num_feat, T)

        return ListDataset([entry], freq=FREQ_STR)

    # --- Monthly loop (re-instantiate forecaster each step) ---
    for date_t in test_idx:
        pos = y.index.get_loc(date_t)
        if pos < 60:   # require at least some warmup history
            continue

        ds_one = make_entry(date_t)
        if ds_one is None:
            continue

        model = Moirai2Forecast(
            module=module,
            prediction_length=1,
            context_length=ctx,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=len(covs),
        )
        predictor = model.create_predictor(batch_size=2)
        try:
            predictor = predictor.to(device)
        except Exception:
            pass

        f = next(predictor.predict(ds_one))
        y_hat = float(f.quantile(0.5)[0])
        if ct_cutoff:
            y_hat = max(y_hat, 0.0)

        y_true = float(y.iloc[pos])
        if not (np.isnan(y_hat) or np.isnan(y_true)):
            preds.append(y_hat)
            trues.append(y_true)
            oos_dates.append(date_t)

    # --- Evaluate & return traces for plotting ---
    trues = np.asarray(trues, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if preds.size == 0:
        raise RuntimeError("No valid Moirai-2 predictions; check data coverage / ctx / start_oos.")
    r2 = evaluate_oos(trues, preds, model_name=model_name, device=device, quiet=quiet)
    return r2, trues, preds, pd.DatetimeIndex(oos_dates)






def ols_oos_dp_lag(
    data: pd.DataFrame,
    start_oos="1965-01-01",
    lag=1,
    min_train=30,
    quiet=False,
):
    df = data.copy()

    # 1) Ensure datetime index & sort
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 2) Ensure equity_premium exists
    if "equity_premium" not in df.columns:
        raise ValueError("Column 'equity_premium' not found in data.")

    # 3) Create dp_lag if missing (try 'd/p' or 'dp' as source)
    if "dp_lag" not in df.columns:
        src = None
        if "d/p" in df.columns:
            src = "d/p"
        elif "dp" in df.columns:
            src = "dp"
        else:
            raise ValueError("Neither 'dp_lag' nor a source column ('d/p' or 'dp') exists.")
        df["dp_lag"] = df[src].shift(lag)

    start_oos = pd.Timestamp(start_oos)

    predictions, actuals = [], []

    # 4) Expanding-window OOS loop
    for date_t in df.index:
        if date_t < start_oos:
            continue

        # strictly past data up to t-1
        est = df.loc[:date_t].iloc[:-1].copy()

        # drop rows with NaNs in training features or target
        est = est.dropna(subset=["dp_lag", "equity_premium"])
        if len(est) < min_train:
            continue

        X_train = est[["dp_lag"]].to_numpy(dtype=float)
        y_train = est["equity_premium"].to_numpy(dtype=float)

        # feature for prediction at time t
        x_pred = df.loc[date_t, "dp_lag"]
        if pd.isna(x_pred):
            continue
        X_pred = np.array([[float(x_pred)]])

        # fit OLS and predict
        model = LinearRegression().fit(X_train, y_train)
        pred = float(model.predict(X_pred)[0])

        predictions.append(pred)
        actuals.append(float(df.loc[date_t, "equity_premium"]))

    # 5) To arrays
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)

    if predictions.size == 0:
        raise RuntimeError("No valid predictions produced. Check lags/data coverage.")

    # 6) Metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = float(np.sqrt(mse))
    mean_forecast = np.array([actuals[:i].mean() for i in range(1, len(actuals) + 1)])
    denom = np.sum((actuals - mean_forecast) ** 2)
    r2_oos = float(1 - np.sum((actuals - predictions) ** 2) / denom) if denom > 0 else np.nan

    if not quiet:
        print(f"[OLS Benchmark] Valid months={len(actuals)} | "
              f"MSE={mse:.6f} | RMSE={rmse:.6f} | R²_OS={r2_oos:.4f}")

    return r2_oos, mse, rmse, actuals, predictions


import numpy as np
import pandas as pd
import torch
from tsfm_public import FlowStateForPrediction

def flowstate_oos(
    data: pd.DataFrame,
    target_col="equity_premium",
    start_oos="1965-01-01",
    ctx=240,
    freq="M",
    scale_factor=0.25,
    quantile=0.5,
    ct_cutoff=False,
    quiet=False,
    model_name="FlowState (expanding, 1-step)",
    auto_move_start=True,
):
    # -------------------- data prep --------------------
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found.")

    s = df[[target_col]].copy()

    def align_monthly(series, f):
        z = series.copy()
        z.index = z.index.to_period(f).to_timestamp(f)
        z = z[~z.index.duplicated(keep="last")].sort_index().asfreq(f)
        z[target_col] = z[target_col].ffill()
        return z

    if freq in {"M", "MS"}:
        s = align_monthly(s, freq)
        if s[target_col].isna().all():
            alt = "M" if freq == "MS" else "MS"
            s = align_monthly(df[[target_col]], alt)
            if not quiet:
                print(f"[FlowState] Retried with freq='{alt}' because '{freq}' produced all-NaN.")
            freq = alt
    elif freq is not None:
        s = s.asfreq(freq)
        s[target_col] = s[target_col].ffill()

    y = s[target_col].astype("float32")
    if y.isna().all():
        raise ValueError("Target is all NaN after preprocessing/alignment.")

    start_oos = pd.Timestamp(start_oos)
    if freq in {"M", "MS"}:
        start_oos = start_oos.to_period(freq).to_timestamp(freq)

    if start_oos < y.index.min():
        start_oos = y.index.min()

    pos0 = y.index.get_indexer([start_oos], method="backfill")[0]
    while pos0 < ctx and auto_move_start and pos0 < len(y):
        pos0 += 1
    if pos0 >= len(y):
        raise ValueError("No valid start date with sufficient history found.")
    start_oos = y.index[pos0]
    test_idx = y.index[y.index >= start_oos]

    if not quiet:
        print(f"[FlowState] freq={freq} | rows={len(y)} | first={y.index.min().date()} | last={y.index.max().date()}")
        print(f"[FlowState] start_oos={start_oos.date()} | ctx={ctx} | tests={len(test_idx)}")

    # -------------------- model --------------------
    # Avoid MPS due to incompatibility; prefer CUDA else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = FlowStateForPrediction.from_pretrained("ibm-research/flowstate").to(device)

    preds, trues = [], []
    oos_dates = []   # <--- add this

    q_idx = None

    # -------------------- expanding OOS loop --------------------
    for date_t in test_idx:
        pos = y.index.get_loc(date_t)
        if isinstance(pos, slice):
            pos = pos.start
        if pos < ctx:
            continue

        ctx_vals = y.values[pos - ctx: pos]
        if np.isnan(ctx_vals).any():
            continue

        # ensure float32 and correct device
        ctx_tensor = torch.from_numpy(ctx_vals[:, None, None]).to(torch.float32).to(device)  # (ctx, 1, 1)

        with torch.inference_mode():
            out = predictor(ctx_tensor, scale_factor=scale_factor, prediction_length=1, batch_first=False)
        po = out.prediction_outputs  # (1, num_quantiles, 1, 1)

        if q_idx is None:
            if hasattr(out, "quantile_values"):
                qs = torch.tensor(out.quantile_values, device=po.device)
                q_idx = int(torch.argmin(torch.abs(qs - quantile)).item())
            else:
                q_idx = po.shape[1] // 2

        y_hat = float(po[0, q_idx, 0, 0].detach().cpu().numpy())
        if ct_cutoff:
            y_hat = max(y_hat, 0.0)

        y_true = float(y.iloc[pos])
        if np.isnan(y_hat) or np.isnan(y_true):
            continue

        preds.append(y_hat)
        trues.append(y_true)
        oos_dates.append(date_t)   # <--- add this


    trues = np.asarray(trues, dtype=float)
    preds = np.asarray(preds, dtype=float)

    if preds.size == 0:
        raise RuntimeError("No valid FlowState predictions. Check ctx/start_oos/freq alignment.")

    r2 = evaluate_oos(trues, preds, model_name=model_name, device=device, quiet=quiet)
    return r2, trues, preds,pd.DatetimeIndex(oos_dates)



    #missing: CT truncation, 20 years after the series begins (≥1946), they recompute any filter/coefficients expanding in time.