from math import inf
import os
import sys
from typing import Callable, Sequence, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import torch


sys.path.insert(0, os.path.abspath('../'))

from source.modelling_utils import ensure_datetime_index,align_monthly, expanding_oos_tabular, expanding_oos_univariate,plot_oos
import torch
# ================================================================
# 3. OLS & RANKING (1-step tabular)
# ================================================================

def make_lagged_features(df: pd.DataFrame, vars_, lag: int) -> pd.DataFrame:
    df = df.copy()
    for L in range(1, lag + 1):
        for v in vars_:
            df[f"{v}_lag{L}"] = df[v].shift(L)
    return df


def ols_oos(
    data: pd.DataFrame,
    variables=("d/p",),
    target_col="equity_premium",
    start_oos="1965-01-01",
    start_date="1927-01-01",
    lag=1,
    min_train=30,
    ct_cutoff: bool = False,
    quiet: bool = False,
    model_name: str | None = None,
    mode = "mean"
):
    """
    Expanding-window OLS with lagged predictors (1-step ahead).
    """
    if model_name is None:
        model_name = f"OLS({','.join(variables)})"

    df = ensure_datetime_index(data)
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()
    df = make_lagged_features(df, variables, lag)

    feature_cols = [f"{v}_lag{L}" for v in variables for L in range(1, lag+1)]

    def fit_predict(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        est = est.dropna(subset=feature_cols + [target_col])
        if len(est) < min_train:
            return None

        X_train = est[feature_cols].to_numpy(float)
        y_train = est[target_col].to_numpy(float)

        if row_t[feature_cols].isna().any():
            return None
        x_pred = row_t[feature_cols].to_numpy(float).reshape(1, -1)

        model = LinearRegression().fit(X_train, y_train)
        return float(model.predict(x_pred)[0])

    return expanding_oos_tabular(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
        start_oos=start_oos,
        start_date=start_date,
        min_train=min_train,
        min_history_months=None,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=fit_predict,
        mode = mode
    )


def rank_monthly_predictors(
    data: pd.DataFrame,
    monthly_vars,
    start_date="1927-01-01",
    start_oos="1965-01-01",
    lag=1,
    quiet=True,
    ct_cutoff=True,
):
    """
    1-step OLS predictor ranking (unchanged logic).
    """
    results = []
    for v in monthly_vars:
        try:
            r2, _, _, _ = ols_oos(
                data,
                variables=(v,),
                target_col="equity_premium",
                start_oos=start_oos,
                start_date=start_date,
                lag=lag,
                min_train=30,
                ct_cutoff=ct_cutoff,
                quiet=True,
                model_name=f"OLS({v})",
            )
        except Exception as e:
            r2 = float("nan")
            if not quiet:
                print(f"[WARN] {v}: {e}")
        results.append({"variable": v, "r2_oos": r2})

    res_df = pd.DataFrame(results)
    sort_key = res_df["r2_oos"].fillna(-inf)
    res_df = res_df.loc[sort_key.sort_values(ascending=True).index].reset_index(drop=True)

    print("\nMonthly predictors ranked (worst → best) by OOS R²:")
    for i, row in res_df.iterrows():
        r2 = row["r2_oos"]
        r2_str = "NaN" if pd.isna(r2) else f"{r2:.4f}"
        print(f"{i+1:2d}. {row['variable']:>10s}   R²_OOS = {r2_str}")

    return res_df


# ================================================================
# 4. TREE ENSEMBLE (XGB / GBRT, 1-step)
# ================================================================

def tree_ensemble_oos(
    data: pd.DataFrame,
    variables,
    target_col="equity_premium",
    start_oos="1965-01-01",
    start_date="1927-01-01",
    lag=1,
    min_train=120,
    ct_cutoff=True,
    drop_sparse=True,
    sparse_thresh=0.6,
    quiet=False,
    model_params=None,
    mode = "mean"
):
    """
    1-step tree ensemble OOS (same logic as before).
    """
    import numpy as np

    if model_params is None:
        model_params = {}

    # model selection
    try:
        from xgboost import XGBRegressor
        use_xgb = True
        default_params = dict(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="reg:squarederror", random_state=42,
        )
        default_params.update(model_params)

        def make_model():
            return XGBRegressor(**default_params)
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        use_xgb = False
        default_params = dict(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            subsample=0.8, random_state=42,
        )
        default_params.update(model_params)

        def make_model():
            return GradientBoostingRegressor(**default_params)

    df = ensure_datetime_index(data)
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

    # lag features
    df = make_lagged_features(df, variables, lag)


    all_feats = [f"{v}_lag{L}" for v in variables for L in range(1, lag + 1)]

    def fit_predict(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        feats = all_feats

        if drop_sparse:
            avail = est[feats].notna().mean(axis=0)
            feats = [c for c in feats if avail.get(c, 0.0) >= sparse_thresh]
            if not feats:
                return None

        y_est = est[target_col]
        if y_est.notna().sum() < min_train:
            return None

        med = est[feats].median(skipna=True)

        X_train = est[feats].fillna(med).to_numpy()
        y_train = y_est.to_numpy()
        m = ~np.isnan(y_train)
        X_train, y_train = X_train[m], y_train[m]
        if len(y_train) < min_train:
            return None

        past_vals = est[feats].iloc[-1:].ffill().iloc[0]
        x_pred_series = row_t[feats].fillna(past_vals).fillna(med)
        x_pred = x_pred_series.to_numpy(float).reshape(1, -1)
        if np.isnan(x_pred).any():
            return None

        model = make_model()
        model.fit(X_train, y_train)
        return float(model.predict(x_pred)[0])

    name = "XGB" if use_xgb else "GBRT"
    return expanding_oos_tabular(
        df,
        target_col=target_col,
        feature_cols=all_feats,
        start_oos=start_oos,
        start_date=start_date,
        min_train=min_train,
        min_history_months=None,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=f"{name}({','.join(variables)})",
        model_fit_predict_fn=fit_predict,
        mode = mode
    )


# ================================================================
# 5. DEEP TS MODELS – CHRONOS, TIMESFM, FLOWSTATE, MOIRAI2
# ================================================================

# 5.1 Chronos
from chronos import BaseChronosPipeline

def chronos_oos(
    data: pd.DataFrame,
    target_col="equity_premium",
    start_oos="1965-01-01",
    freq="MS",
    prediction_length: int = 1,
    ct_cutoff=True,
    quiet=False,
    mode = "mean"
):
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

    r2, trues, preds, dates = expanding_oos_univariate(
        y,
        start_oos=start_oos,
        prediction_length=prediction_length,
        min_history_months=240,   # 20 years
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name="Chronos-Bolt",
        forecast_multi_step_fn=forecast_multi_step,
        mode = mode
    )
    return r2, trues, preds, dates


# 5.2 TimesFM
import timesfm

def timesfm_oos(
    data: pd.DataFrame,
    target_col="equity_premium",
    start_oos="1965-01-01",
    freq="MS",
    prediction_length: int = 1,
    min_context=120,
    max_context=512,
    ct_cutoff=True,
    quiet=False,
    mode = "mean"
):
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if not quiet:
        print(f"[TimesFM] Using device hint: {device}")

    df = ensure_datetime_index(data)
    y = align_monthly(df[[target_col]], freq, col=target_col)[target_col].astype("float32")
    if len(y) == 0:
        raise ValueError("No target data after cleaning.")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    cfg = timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=max(prediction_length, 128),
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
    model.compile(cfg)

    def forecast_multi_step(y_hist: pd.Series, date_t, H: int) -> np.ndarray:
        context = y_hist.to_numpy(dtype="float32")
        if len(context) < min_context:
            return np.full(H, np.nan, dtype="float32")
        if len(context) > cfg.max_context:
            context = context[-cfg.max_context:]
        if np.isnan(context).any() or np.std(context) < 1e-6:
            return np.full(H, np.nan, dtype="float32")

        with torch.inference_mode():
            point_fcst, _ = model.forecast(horizon=H, inputs=[context])
        # shape: (1, H)
        return np.asarray(point_fcst[0, :H], dtype="float32")

    r2, trues, preds, dates = expanding_oos_univariate(
        y,
        start_oos=start_oos,
        prediction_length=prediction_length,
        min_history_months=0,  # rely on min_context
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name="TimesFM",
        forecast_multi_step_fn=forecast_multi_step,
        mode = mode
    )
    return r2, trues, preds, dates


# 5.3 FlowState
from tsfm_public import FlowStateForPrediction

def flowstate_oos(
    data: pd.DataFrame,
    target_col="equity_premium",
    start_oos="1965-01-01",
    ctx=240,
    freq="M",
    prediction_length: int = 1,
    scale_factor=0.25,
    quantile=0.5,
    ct_cutoff=False,
    quiet=False,
    model_name="FlowState (expanding)",
    mode = "mean"
):
    df = ensure_datetime_index(data)
    s = df[[target_col]].copy()

    def align_freq(series, f):
        z = series.copy()
        z.index = z.index.to_period(f).to_timestamp(f)
        z = z[~z.index.duplicated(keep="last")].sort_index().asfreq(f)
        z[target_col] = z[target_col].ffill()
        return z

    if freq in {"M", "MS"}:
        s = align_freq(s, freq)
        if s[target_col].isna().all():
            alt = "M" if freq == "MS" else "MS"
            s = align_freq(df[[target_col]], alt)
            if not quiet:
                print(f"[FlowState] Retried with freq='{alt}' because '{freq}' produced all-NaN.")
            freq = alt
    elif freq is not None:
        s = s.asfreq(freq)
        s[target_col] = s[target_col].ffill()

    y = s[target_col].astype("float32")
    if y.isna().all():
        raise ValueError("Target is all NaN after preprocessing/alignment.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = FlowStateForPrediction.from_pretrained("ibm-research/flowstate").to(device)

    q_idx = {"value": None}

    def forecast_multi_step(y_hist: pd.Series, date_t, H: int) -> np.ndarray:
        if len(y_hist) < ctx:
            return np.full(H, np.nan, dtype="float32")

        ctx_vals = y_hist.iloc[-ctx:].to_numpy(dtype="float32")
        if np.isnan(ctx_vals).any():
            return np.full(H, np.nan, dtype="float32")

        ctx_tensor = torch.from_numpy(ctx_vals[:, None, None]).to(torch.float32).to(device)
        with torch.inference_mode():
            out = predictor(ctx_tensor, scale_factor=scale_factor, prediction_length=H, batch_first=False)
        po = out.prediction_outputs  # expected shape: (1, num_quantiles, H, 1)

        if q_idx["value"] is None:
            if hasattr(out, "quantile_values"):
                qs = torch.tensor(out.quantile_values, device=po.device)
                q_idx["value"] = int(torch.argmin(torch.abs(qs - quantile)).item())
            else:
                q_idx["value"] = po.shape[1] // 2

        # (H,) vector
        vec = po[0, q_idx["value"], :H, 0].detach().cpu().numpy().astype("float32")
        return vec

    r2, trues, preds, dates = expanding_oos_univariate(
        y,
        start_oos=start_oos,
        prediction_length=prediction_length,
        min_history_months=0,  # rely on ctx
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        forecast_multi_step_fn=forecast_multi_step,
        mode = mode
    )
    return r2, trues, preds, dates


# 5.4 Moirai2
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

def moirai2_oos(
    data: pd.DataFrame,
    covariates=("d/p", "tms", "dfy"),
    start_oos="1965-01-01",
    ctx=240,
    prediction_length: int = 1,
    device="cpu",
    ct_cutoff=False,
    quiet=False,
    model_name="Moirai 2 (reinstantiated each step)",
    FREQ_STR="M",
    mode = "mean"
):
    if not quiet:
        print(f"[Moirai2] Using freq='{FREQ_STR}' (month-end) | ctx={ctx} | H={prediction_length}")

    df = ensure_datetime_index(data)
    df.index = df.index.to_period(FREQ_STR).to_timestamp(FREQ_STR)
    df = df.sort_index().asfreq(FREQ_STR)

    needed = ["equity_premium"] + list(covariates)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[needed].dropna()
    y = df["equity_premium"].astype("float32")
    cov_df = df[list(covariates)].astype("float32")

    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")

    def forecast_multi_step(y_hist: pd.Series, date_t, H: int) -> np.ndarray:
        pos = y.index.get_loc(date_t)
        if isinstance(pos, slice):
            pos = pos.start
        if pos <= 0 or pos < 60:
            return np.full(H, np.nan, dtype="float32")

        y_hist_full = y.values[:pos]
        if len(y_hist_full) == 0:
            return np.full(H, np.nan, dtype="float32")

        if len(y_hist_full) > ctx:
            y_seg = y_hist_full[-ctx:]
            start_idx = pos - len(y_seg)
        else:
            y_seg = y_hist_full
            start_idx = 0

        entry = {
            "start": pd.Timestamp(y.index[start_idx]),
            "target": y_seg.astype("float32"),
        }

        if len(covariates) > 0:
            mats = [cov_df[c].values[:pos] for c in covariates]
            mats = [m[-len(y_seg):] for m in mats]
            entry["past_feat_dynamic_real"] = np.vstack(mats).astype("float32")

        ds_one = ListDataset([entry], freq=FREQ_STR)

        model = Moirai2Forecast(
            module=module,
            prediction_length=H,
            context_length=ctx,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=len(covariates),
        )
        predictor = model.create_predictor(batch_size=2)
        try:
            predictor = predictor.to(device)
        except Exception:
            pass

        f = next(predictor.predict(ds_one))
        q_med = f.quantile(0.5)  # shape: (H,)
        vec = np.asarray(q_med[:H], dtype="float32")
        return vec

    r2, trues, preds, dates = expanding_oos_univariate(
        y,
        start_oos=start_oos,
        prediction_length=prediction_length,
        min_history_months=0,  # rely on pos>=60 + ctx
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        forecast_multi_step_fn=forecast_multi_step,
        mode = mode
    )
    return r2, trues, preds, dates


# ================================================================
# 6. TABPFN (tabular, 1-step) & TABPFN-TS (multi-step)
# ================================================================

def tabpfn_oos_fit_each_step(
    data: pd.DataFrame,
    variables=("d/p", "tms", "dfy"),
    target_col="equity_premium",
    start_oos="1965-01-01",
    start_date="1927-01-01",
    lag=1,
    min_train=120,
    ct_cutoff=False,
    quiet=False,
    model_name="TabPFN (fit each step)",
    model_params=None,
    mode="mean",
):
    """
    Tabular TabPFN: still 1-step (needs exogenous predictors).
    """
    import torch
    try:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except Exception as e:
        raise RuntimeError("TabPFN not installed. Please `pip install tabpfn`.") from e

    # ---- FIX: don't use N_ensemble_configurations here ----
    default_params: dict = {}
    
    df = ensure_datetime_index(data)
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in data.")
    for v in variables:
        if v not in df.columns:
            raise ValueError(f"Predictor '{v}' not found in data.")

    for L in range(1, lag + 1):
        for v in variables:
            df[f"{v}_lag{L}"] = df[v].shift(L)
    feature_cols = [f"{v}_lag{L}" for v in variables for L in range(1, lag + 1)]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit_predict(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        from tabpfn import TabPFNRegressor

        est_clean = est.dropna(subset=feature_cols + [target_col])
        if len(est_clean) < min_train:
            return None

        X_train = est_clean[feature_cols].to_numpy(float)
        y_train = est_clean[target_col].to_numpy(float)

        if row_t[feature_cols].isna().any():
            return None
        X_pred = row_t[feature_cols].to_numpy(float).reshape(1, -1)

        # only pass parameters that TabPFNRegressor actually supports
        if model_params == '2.5':
            model = TabPFNRegressor(device=device)
        else:
            model = TabPFNRegressor.create_default_for_version(ModelVersion.V2,device=device)
        model.fit(X_train, y_train)
        return float(model.predict(X_pred)[0])

    return expanding_oos_tabular(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
        start_oos=start_oos,
        start_date=start_date,
        min_train=min_train,
        min_history_months=None,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=fit_predict,
        mode=mode,
    )



def tabpfn_ts_oos_fit_each_step(
    data: pd.DataFrame,
    target_col: str = "equity_premium",
    start_oos: str = "1965-01-01",
    ctx: int = 240,
    freq: str = "M",
    prediction_length: int = 1,
    min_windows: int = 120,
    ct_cutoff: bool = False,
    quiet: bool = False,
    model_name: str = "TabPFN-TS (fit each step)",
    forecaster_repo: str = "tabpfn/tabpfn-ts",
    fit_kwargs: dict | None = None,
    mode = "mean"
):
    """
    Expanding-window, multi-step OOS using TabPFN-TS, retrained at each origin.

    Uses recursive predictions for horizons > 1.
    """
    import torch
    try:
        from tabpfn_ts import TabPFNForecaster
    except Exception as e:
        raise RuntimeError("tabpfn-ts not found. Install it first (`pip install tabpfn-ts`).") from e

    if fit_kwargs is None:
        fit_kwargs = {}

    df = ensure_datetime_index(data)

    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found.")

    def align_freq(series: pd.DataFrame, f: str) -> pd.DataFrame:
        z = series.copy()
        z.index = z.index.to_period(f).to_timestamp(f)
        z = z[~z.index.duplicated(keep="last")].sort_index().asfreq(f)
        z[target_col] = z[target_col].ffill()
        return z

    s = align_freq(df[[target_col]], freq if freq in {"M", "MS"} else "M")
    if s[target_col].isna().all() and freq in {"M", "MS"}:
        alt = "M" if freq == "MS" else "MS"
        s = align_freq(df[[target_col]], alt)
        if not quiet:
            print(f"[TabPFN-TS] Retried with freq='{alt}' because '{freq}' produced all-NaN.")
        freq = alt

    y = s[target_col].astype("float32")
    if y.isna().all():
        raise ValueError("Target is all NaN after preprocessing/alignment.")

    if not quiet:
        print(f"[TabPFN-TS] freq={freq} | rows={len(y)} | first={y.index.min().date()} | last={y.index.max().date()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_windows(arr: np.ndarray, end_pos: int, w: int):
        """
        Build contexts (N, w, 1) and targets (N,) from arr[:end_pos].
        contexts[i] = arr[i : i+w], target[i] = arr[i+w]
        """
        N = end_pos - w
        if N <= 0:
            return None, None
        X = np.lib.stride_tricks.sliding_window_view(arr[:end_pos], window_shape=w, axis=0)
        X = X.astype("float32")[..., None]  # (N, w, 1)
        y_next = arr[w:end_pos].astype("float32")  # length N
        return X, y_next

    def forecast_multi_step(y_hist: pd.Series, date_t, H: int) -> np.ndarray:
        pos = y.index.get_loc(date_t)
        if isinstance(pos, slice):
            pos = pos.start

        if pos < ctx + min_windows:
            return np.full(H, np.nan, dtype="float32")

        contexts, targets = build_windows(y.values, end_pos=pos, w=ctx)
        if contexts is None or len(contexts) < min_windows:
            return np.full(H, np.nan, dtype="float32")
        if np.isnan(contexts).any() or np.isnan(targets).any():
            return np.full(H, np.nan, dtype="float32")

        # we will recursively forecast using a copy of y_hist
        history = y_hist.copy()

        try:
            model = TabPFNForecaster.from_pretrained(forecaster_repo).to(device)
        except Exception:
            model = TabPFNForecaster().to(device)
        model.eval()

        if hasattr(model, "fit"):
            try:
                model.fit(
                    contexts=contexts,
                    targets=targets,
                    **fit_kwargs,
                )
            except Exception as e:
                if not quiet:
                    print(f"[TabPFN-TS] fit failed at {date_t.date()}: {e}")
                return np.full(H, np.nan, dtype="float32")

        preds = []
        for h in range(H):
            # last ctx window from current history
            if len(history) < ctx:
                return np.full(H, np.nan, dtype="float32")
            ctx_vals = history.iloc[-ctx:].to_numpy(dtype="float32")
            if np.isnan(ctx_vals).any():
                return np.full(H, np.nan, dtype="float32")

            ctx_last = ctx_vals[None, :, None]  # (1, ctx, 1)
            with torch.inference_mode():
                try:
                    out = model.predict(ctx_last)  # (1, 1)
                except Exception:
                    ctx_tensor = torch.tensor(ctx_last, dtype=torch.float32, device=device)
                    out = model.predict(ctx_tensor)
                    if isinstance(out, torch.Tensor):
                        out = out.detach().cpu().numpy()

            y_hat = float(np.asarray(out).reshape(-1)[0])
            preds.append(y_hat)
            # append prediction to history for next step
            history = pd.concat(
                [history, pd.Series([y_hat], index=[history.index[-1] + (history.index[1] - history.index[0])])]
            )

        return np.asarray(preds, dtype="float32")

    r2, trues, preds, dates = expanding_oos_univariate(
        y,
        start_oos=start_oos,
        prediction_length=prediction_length,
        min_history_months=0,  # rely on ctx + min_windows
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        forecast_multi_step_fn=forecast_multi_step,
        mode = mode
    )
    return r2, trues, preds, dates
