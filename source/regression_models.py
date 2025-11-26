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

from source.modelling_utils import ensure_datetime_index,align_monthly, expanding_oos_tabular
import torch
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
# ================================================================
# 1. OLS & RANKING (1-step tabular)
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
            r2,_, _, _, _ = ols_oos(
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
# 2. TREE ENSEMBLE (XGB / GBRT, 1-step)
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

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def autoarima_oos(
    data: pd.DataFrame,
    target_col: str = "equity_premium",
    start_oos: str = "1965-01-01",
    freq: str = "MS",                # user-facing freq, but we will align to MS internally
    prediction_length: int = 1,      # this implementation only supports 1-step
    min_history_months: int = 60,
    seasonal: bool | None = None,
    m: int | None = None,
    ct_cutoff: bool = True,
    quiet: bool = False,
    model_name: str = "AutoARIMA",
    auto_arima_kwargs: dict | None = None,
    order_search_every: int | None = None,    # re-search orders every k steps
    refit_each_step: bool = True,
    mode: str = "mean",
) -> Tuple[float, Dict, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Expanding-window 1-step-ahead OOS for pmdarima.auto_arima using
    the generic expanding_oos_tabular driver.

    IMPORTANT:
    - We force the target series to month-start ("MS") internally so that
      AutoARIMA works on a regular monthly grid without shifting your
      original month-start Convention.
    - This does NOT affect your linear regression / tree / TabPFN models.
    """
    # --- local imports, so NameError cannot happen ---
    import warnings
    from pmdarima import auto_arima, ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    if prediction_length != 1:
        raise ValueError("This autoarima_oos implementation supports only prediction_length = 1.")

    if auto_arima_kwargs is None:
        auto_arima_kwargs = {}

    # ------------------------------------------------------------------
    # 1. Prepare target series at MONTH-START frequency
    # ------------------------------------------------------------------
    df = ensure_datetime_index(data)

    # We FORCE month-start alignment here, regardless of `freq` passed in,
    # so that AutoARIMA sees the same month-start grid as the rest of your data.
    df_ms = align_monthly(df[[target_col]], freq=freq, col=target_col)
    y = df_ms[target_col].astype("float32")

    # ------------------------------------------------------------------
    # 2. Seasonal settings (still based on user freq, but everything is monthly)
    # ------------------------------------------------------------------
    freq_u = freq.upper()
    # If not specified, default to seasonal for monthly/quarterly
    if seasonal is None:
        seasonal = freq_u in {"M", "MS", "ME", "Q", "QS"}

    # Default seasonal period length
    if m is None:
        if freq_u in {"M", "MS", "ME"}:
            m = 12
        else:
            m = 1

    # Keep state of best orders across time
    state = {"order": None, "seasonal_order": None, "step": 0}

    # ------------------------------------------------------------------
    # 3. Internal: forecast one step given y_hist and date_t
    # ------------------------------------------------------------------
    def forecast_one_step(y_hist: pd.Series, date_t: pd.Timestamp) -> float | None:
        """
        y_hist: past values up to t-1 on a month-start grid
        date_t: current origin date t (month-start timestamp)
        """
        state["step"] += 1
        y_hist = y_hist.dropna()
        if len(y_hist) < min_history_months:
            return None

        # Decide if we need to (re)search the best (p,d,q)(P,D,Q,m)
        need_search = (
            state["order"] is None
            or state["seasonal_order"] is None
            or (order_search_every is not None
                and state["step"] % order_search_every == 1)
        )

        if need_search:
            if not quiet:
                print(f"[AutoARIMA] Searching best order at {date_t.date()}...")
            model = auto_arima(
                y_hist.values,
                seasonal=seasonal,
                m=m,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                **auto_arima_kwargs,
            )
            state["order"] = model.order
            state["seasonal_order"] = model.seasonal_order
            if not quiet:
                print(f"  best order={model.order}, seasonal_order={model.seasonal_order}")
        else:
            # IMPORTANT: do NOT pass seasonal/m here to avoid the FutureWarning
            model = ARIMA(
                order=state["order"],
                seasonal_order=state["seasonal_order"],
            )

        # Refit on current history if requested
        if refit_each_step:
            try:
                with warnings.catch_warnings():
                    if quiet:
                        warnings.simplefilter("ignore")
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model = model.fit(y_hist.values)
            except Exception as e:
                if not quiet:
                    print(f"[AutoARIMA] fit failed at {date_t}: {e}")
                return None

        # 1-step forecast
        try:
            with warnings.catch_warnings():
                if quiet:
                    warnings.simplefilter("ignore")
                fc = model.predict(n_periods=1)
        except Exception as e:
            if not quiet:
                print(f"[AutoARIMA] predict failed at {date_t}: {e}")
            return None

        return float(fc[0])

    # ------------------------------------------------------------------
    # 4. Wrap this into model_fit_predict_fn for expanding_oos_tabular
    # ------------------------------------------------------------------
    def model_fit_predict_fn(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        """
        est : past data (month-start-aligned DataFrame) up to t-1
        row_t : row at time t (we only need its timestamp)
        """
        date_t = row_t.name
        # Use the global aligned series 'y' (month-start) to build history
        # up to but not including date_t.
        y_hist = y.loc[:date_t].iloc[:-1]
        if y_hist.empty:
            return None
        return forecast_one_step(y_hist, date_t)

    # ------------------------------------------------------------------
    # 5. Call the generic expanding OOS driver
    # ------------------------------------------------------------------
    r2, stats, trues, preds, dates = expanding_oos_tabular(
        df_ms,                          # month-start target series
        target_col=target_col,
        feature_cols=[],                # no explicit tabular features; model uses y_hist internally
        start_oos=start_oos,
        start_date=df_ms.index.min().strftime("%Y-%m-%d"),
        min_train=min_history_months,
        min_history_months=min_history_months,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=model_fit_predict_fn,
        mode=mode,
    )

    return r2, stats, trues, preds, dates
# ================================================================
# 5. DEEP TS MODELS – CHRONOS
# ================================================================
from chronos import BaseChronosPipeline

def chronos_oos(
    data: pd.DataFrame,
    target_col="equity_premium",
    start_oos="1965-01-01",
    freq="MS",
    prediction_length: int = 1,
    ctx_min: int = 24,
    ct_cutoff=True,
    quiet=False,
    mode = "mean"
):
    """
    1-step-ahead Chronos-Bolt OOS evaluation using expanding_oos_tabular.

    For true multi-step forecasts, keep a separate function using expanding_oos_univariate.
    """
    if prediction_length != 1:
        raise ValueError("This chronos_oos version supports only prediction_length=1.")

    df = ensure_datetime_index(data)
    y_df = align_monthly(df[[target_col]], freq=freq, col=target_col)
    y = y_df[target_col].astype("float32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    def forecast_one_step(y_hist: pd.Series, date_t: pd.Timestamp) -> float | None:
        ctx = y_hist.to_numpy(dtype="float32")
        if len(ctx) < ctx_min:
            return None
        with torch.inference_mode():
            _, mean_pred = pipe.predict_quantiles(
                context=[torch.tensor(ctx, device=device)],
                prediction_length=1,
                quantile_levels=[0.5],
            )
        return float(mean_pred[0, 0])

    def model_fit_predict_fn(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        date_t = row_t.name
        y_hist = y.loc[:date_t].iloc[:-1]
        return forecast_one_step(y_hist, date_t)

    r2, stats, trues, preds, dates = expanding_oos_tabular(
        y_df,
        target_col=target_col,
        feature_cols=[],  # no exogenous predictors; Chronos sees the sequence in y_hist
        start_oos=start_oos,
        start_date=y_df.index.min().strftime("%Y-%m-%d"),
        min_train=ctx_min,
        min_history_months=ctx_min,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name="Chronos-Bolt (1-step)",
        model_fit_predict_fn=model_fit_predict_fn,
        mode=mode,
    )
    return r2, stats, trues, preds, dates
# 5.2 Moirai2
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
    """
    Same as your original moirai2_oos, but using expanding_oos_tabular
    instead of expanding_oos_univariate. Only supports 1-step ahead.
    """
    if prediction_length != 1:
        raise ValueError("This moirai2_oos(tabular) version supports only prediction_length=1.")

    if not quiet:
        print(f"[Moirai2] Using freq='{FREQ_STR}' (month-end) | ctx={ctx} | H=1")

    # ------- 1. Data prep (unchanged) -------
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

    # ------- 2. Load Moirai2 (unchanged) -------
    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")

    # ------- 3. Fit/predict wrapper for tabular OOS -------
    def model_fit_predict_fn(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        """
        Called by expanding_oos_tabular at each OOS date.
        est  = past data up to t-1
        row_t = row at time t (we use its timestamp)
        """
        date_t = row_t.name

        # position in global y index (same as your original code)
        pos = y.index.get_loc(date_t)
        if isinstance(pos, slice):
            pos = pos.start
        if pos <= 0 or pos < 60:
            return None

        y_hist_full = y.values[:pos]
        if len(y_hist_full) == 0:
            return None

        # context window
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
            prediction_length=1,      # 1-step only
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
        q_med = f.quantile(0.5)  # shape: (1,)
        fc = float(np.asarray(q_med[0], dtype="float32"))
        return fc

    # ------- 4. Use expanding_oos_tabular instead of univariate -------
    # expanding_oos_tabular returns: r2, stats, trues, preds, dates
    r2, stats, trues, preds, dates = expanding_oos_tabular(
        df,
        target_col="equity_premium",
        feature_cols=[],              # all logic is inside model_fit_predict_fn
        start_oos=start_oos,
        start_date=df.index.min().strftime("%Y-%m-%d"),
        min_train=1,                  # real history check is inside model_fit_predict_fn (pos<60)
        min_history_months=None,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=model_fit_predict_fn,
        mode=mode,
    )

    # keep old return style (no stats)
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


def add_ts_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, list]:
    """
    Replicates the feature engineering of tabpfn-time-series:
    1. RunningIndexFeature (Time progress)
    2. CalendarFeature (Month, Quarter)
    3. AutoSeasonalFeature (Proxy: Lag-12 for monthly data)
    """
    df = df.copy()
    new_cols = []

    # --- 1. RunningIndexFeature ---
    # Continuous float representation of time (normalized)
    # This tells the Transformer "where" we are in history.
    t_start = df.index[0].timestamp()
    df["feat_running_index"] = (df.index.map(pd.Timestamp.timestamp) - t_start) / 31536000.0
    new_cols.append("feat_running_index")

    # --- 2. CalendarFeature ---
    # Captures cyclical patterns (January effects, Quarter end effects)
    # We treat these as integers; TabPFN handles them well.
    df["feat_month"] = df.index.month
    df["feat_quarter"] = df.index.quarter
    new_cols.extend(["feat_month", "feat_quarter"])

    # --- 3. AutoSeasonalFeature ---
    # The strongest seasonality for monthly financial data is usually Year-over-Year.
    # We add the target value from exactly 1 year ago (Lag 12).
    # If you have higher frequency data, adjust '12'.
    seasonal_lag = 12
    col_seasonal = f"feat_seasonal_lag{seasonal_lag}"
    df[col_seasonal] = df[target_col].shift(seasonal_lag)
    new_cols.append(col_seasonal)

    return df, new_cols
import pandas as pd
import numpy as np
import torch
from scipy.signal import find_peaks
from tabpfn import TabPFNRegressor

# -------------------------------------------------------------------------
# 1. Calendar Features (Refined for Monthly/Daily Data)
# -------------------------------------------------------------------------
def get_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extracts cyclic calendar components relevant for low-frequency (daily/monthly) data.
    Removed: Second, Minute, Hour.
    Kept: Day of week, Day of month, Day of year, Week, Month.
    """
    df_cal = pd.DataFrame(index=dates)
    
    # 1. Linear Year Feature
    df_cal["year"] = dates.year.astype(float)
    
    # 2. Cyclic Features with defined periods (P)
    cycles = [
        # (name, period_length, extraction_func)
        ("day_of_week", 7, lambda x: x.dayofweek),     # 0=Mon, 6=Sun
        ("day_of_month", 30.5, lambda x: x.day),       # Paper uses 30.5 approximation
        ("day_of_year", 365, lambda x: x.dayofyear),
        ("week_of_year", 52, lambda x: x.isocalendar().week),
        ("month_of_year", 12, lambda x: x.month),
    ]
    
    for name, period, func in cycles:
        vals = func(dates).astype(float)
        # Transform to Sin/Cos pairs
        rads = 2 * np.pi * vals / period
        df_cal[f"{name}_sin"] = np.sin(rads)
        df_cal[f"{name}_cos"] = np.cos(rads)
        
    return df_cal

# -------------------------------------------------------------------------
# 2. Automatic Seasonal Features (FFT Based)
# -------------------------------------------------------------------------
def get_auto_seasonal_features(
    y_history: np.ndarray, 
    full_length: int, 
    k: int = 5
) -> pd.DataFrame:
    """
    Extracts top-k periodicities using FFT and creates sin/cos features.
    """
    N = len(y_history)
    # If history is too short, return empty features
    if N < 12: 
        return pd.DataFrame(index=range(full_length))

    # A. Preprocessing: Detrend (Linear)
    t = np.arange(N)
    A = np.vstack([t, np.ones(len(t))]).T
    # Least squares detrending
    m, c = np.linalg.lstsq(A, y_history, rcond=None)[0]
    y_detrend = y_history - (m * t + c)

    # B. Apply Hann Window
    hann_win = np.hanning(N) 
    y_windowed = y_detrend * hann_win

    # C. Zero-Pad to 2N (Paper suggests symmetric zero-padding)
    y_padded = np.pad(y_windowed, (0, N), 'constant')

    # D. FFT & Magnitudes
    fft_vals = np.fft.rfft(y_padded)
    magnitudes = np.abs(fft_vals)
    magnitudes[0] = 0 # Remove DC component - THIS WAS THE LINE WITH THE ERROR
    
    # E. Find Peaks & Select Top K
    peaks, _ = find_peaks(magnitudes)
    if len(peaks) == 0:
        return pd.DataFrame(index=range(full_length))
        
    # Sort peaks by magnitude descending and keep top k
    top_peaks = sorted(peaks, key=lambda x: magnitudes[x], reverse=True)[:k]
    
    # F. Create Features
    features = {}
    t_full = np.arange(full_length) # Time index for whole range
    
    for i, peak_idx in enumerate(top_peaks):
        # Frequency = index / length of FFT buffer (2N)
        f = peak_idx / len(y_padded) 
        
        rads = 2 * np.pi * f * t_full
        features[f"auto_seas_{i}_sin"] = np.sin(rads)
        features[f"auto_seas_{i}_cos"] = np.cos(rads)

    return pd.DataFrame(features)

# -------------------------------------------------------------------------
# 3. Main Driver 
# -------------------------------------------------------------------------
def tabpfn_ts_full_features(
    data: pd.DataFrame,
    variables=("d/p", "tms", "dfy"),
    target_col="equity_premium",
    start_oos="1965-01-01",
    start_date="1927-01-01",
    lag=1,
    k_seasonal=5, 
    min_train=120,
    ct_cutoff=True,
    quiet=False,
    model_params='2.5',
    mode="mean"
):
    # 1. Setup Data
    df = ensure_datetime_index(data).copy()
    df = df.sort_index()
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

    # 2. Pre-calculate Static Features 
    
    # A. Running Index
    # Normalized time index (approx years passed)
    t_start = df.index[0].timestamp()
    df["running_index"] = (df.index.map(pd.Timestamp.timestamp) - t_start) / 31536000.0
    
    # B. Calendar Features (using the refined function above)
    df_cal = get_calendar_features(df.index)
    df = pd.concat([df, df_cal], axis=1)
    
    cal_cols = df_cal.columns.tolist()
    idx_col = ["running_index"]
    
    # C. Lagged Covariates (Standard TS practice)
    macro_cols = []
    lag_range = range(1, lag + 1)
    for v in variables:
        for L in lag_range:
            col_name = f"{v}_lag{L}"
            df[col_name] = df[v].shift(L)
            macro_cols.append(col_name)

    # D. Lagged Target (Autoregression)
    ar_cols = []
    for L in lag_range:
        col_name = f"AR_lag{L}"
        df[col_name] = df[target_col].shift(L)
        ar_cols.append(col_name)

    # Combine static features to track
    static_features = idx_col + cal_cols + macro_cols + ar_cols
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------------
    # Fit/Predict Function
    # ---------------------------------------------------------------------
    def fit_predict(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        # 1. Clean Data
        est_clean = est.dropna(subset=[target_col] + static_features)
        
        if len(est_clean) < min_train:
            return None
        if row_t[static_features].isna().any():
            return None

        # 2. Compute Auto-Seasonal Features (Dynamic)
        # We compute these strictly from history to avoid look-ahead bias
        y_hist = est_clean[target_col].to_numpy(float)
        
        # We need features for N history points + 1 test point
        df_seas = get_auto_seasonal_features(y_hist, full_length=len(y_hist)+1, k=k_seasonal)
        
        # Split back into train/test
        X_seas_train = df_seas.iloc[:-1].to_numpy(float)
        X_seas_test = df_seas.iloc[-1:].to_numpy(float) 
        
        # 3. Construct Final X Matrices
        X_static_train = est_clean[static_features].to_numpy(float)
        X_train = np.hstack([X_static_train, X_seas_train])
        y_train = y_hist
        
        X_static_test = row_t[static_features].to_numpy(float).reshape(1, -1)
        X_pred = np.hstack([X_static_test, X_seas_test])

        # 4. TabPFN Inference
        if model_params == '2.5':
             model = TabPFNRegressor(device=device)
        else:
             model = TabPFNRegressor(device=device)

        model.fit(X_train, y_train)
        return float(model.predict(X_pred)[0])

    # ---------------------------------------------------------------------
    # Run OOS Loop
    # ---------------------------------------------------------------------
    return expanding_oos_tabular(
        df,
        target_col=target_col,
        feature_cols=static_features,
        start_oos=start_oos,
        start_date=start_date,
        min_train=min_train,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name="TabPFN-TS (Refined Features)",
        model_fit_predict_fn=fit_predict,
        mode=mode
    )

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        elif 'Date' in df.columns:
            df = df.set_index('Date')
        df.index = pd.to_datetime(df.index)
    return df


