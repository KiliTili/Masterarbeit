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

from source.regression.modelling_utils import ensure_datetime_index,align_monthly, expanding_oos_tabular
from source.plot_functions.plot_functions import plot_oos,plot_cum_dsse_with_bootstrap_band
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
    min_train=240,
    ct_cutoff: bool = False,
    quiet: bool = False,
    model_name: str | None = None,
    ci = 0.9,
    return_addtional_info: bool = False,
):
    """
    Expanding-window OLS with lagged predictors (1-step ahead).
    """
    if model_name is None:
        model_name = f"OLS({','.join(variables)})"

    df = ensure_datetime_index(data)
#    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()
    df = make_lagged_features(df, variables, lag)

    feature_cols = [f"{v}_lag{L}" for v in variables for L in range(1, lag+1)]
    from scipy.stats import norm            

    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        est = est.dropna(subset=feature_cols)
        valid_feature_cols = [c for c in feature_cols if est[c].notna().all()]
        if len(valid_feature_cols) == 0:
            raise ValueError("No valid features with complete data for OLS.")
        
        
        X_train = est[valid_feature_cols].to_numpy(float)
        y_train = est[target_col].to_numpy(float)

        x_pred = row_t[valid_feature_cols].to_numpy(float).reshape(1, -1)

        model = LinearRegression().fit(X_train, y_train)
        y_hat = float(model.predict(x_pred)[0])
        y_fitted = model.predict(X_train)                 # CHANGED
        resid = y_train - y_fitted                         # CHANGED
        sigma = np.sqrt(np.mean(resid ** 2))               # CHANGED

        z = norm.ppf(0.5 + ci / 2)                          # CHANGED
        y_lo = y_hat - z * sigma                            # CHANGED
        y_hi = y_hat + z * sigma                            # CHANGED

        return [y_hat, y_lo, y_hi]                          # CHANGED
            


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
        return_addtional_info = return_addtional_info,
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
        # use start data where variable is firstly not nan
        start_date_var = data.index[data[v].notna()].min()
        start_date_var = max(pd.Timestamp(start_date), start_date_var)
        start_oos_var = max(pd.Timestamp(start_oos), start_date_var + pd.DateOffset(years=10)) 
        enddate = data.index[data[v].notna()].max()
        
        r2,stats, _, _, _,_,_,_,additional = ols_oos(
            data[(data.index >= start_date_var) &(data.index <= enddate)],
            variables=(v,),
            target_col="equity_premium",
            start_oos=start_oos_var,
            start_date=start_date_var,
            lag=lag,
            min_train=30,
            ct_cutoff=ct_cutoff,
            quiet=True,
            model_name=f"OLS({v})",
            return_addtional_info=True
        )
    
        results.append({"variable": v, "r2_oos": r2, "stats": stats, "start_oos": start_oos_var, "start_date": start_date_var, "end_date": enddate, "additional_info": additional})

    res_df = pd.DataFrame(results)
    sort_key = res_df["r2_oos"].fillna(-inf)
    res_df = res_df.loc[sort_key.sort_values(ascending=True).index].reset_index(drop=True)

    print("\nMonthly predictors ranked (worst → best) by OOS R²:")
    for i, row in res_df.iterrows():
        r2 = row["r2_oos"]
        r2_str = "NaN" if pd.isna(r2) else f"{r2:.4f}"
        print(f"{i+1:2d}. {row['variable']:>10s}   R²_OOS = {r2_str} | Std: {row['stats']['std']:.4f} |data: {row['start_date'].date()} to {row['end_date'].date()} | OOS start: {row['start_oos'].date()} | R^2OOSCT: {row['additional_info']['r2_wct']:.4f} | R^2OOSCT Std: {row['additional_info']['stats_wct']['std']:.4f} ")

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
    #df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

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
    min_history_months: int = 24,
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
    df = df.sort_index()
    #df.index = df.index.to_period("M").to_timestamp("MS")

    y = df[target_col].astype(float)


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
            
            # --- SUPPRESS SKLEARN FUTUREWARNINGS HERE ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                model = auto_arima(
                    y_hist.values,
                    seasonal=seasonal,
                    m=m,
                    error_action="ignore",
                    suppress_warnings=True,
                    stepwise=True,
                    **auto_arima_kwargs,
                )
            # --------------------------------------------
            
            state["order"] = model.order
            state["seasonal_order"] = model.seasonal_order
            if not quiet:
                print(f"  best order={model.order}, seasonal_order={model.seasonal_order}")
        else:
            model = ARIMA(
                order=state["order"],
                seasonal_order=state["seasonal_order"],
            )

        # Refit on current history if requested
        if refit_each_step:
            try:
                with warnings.catch_warnings():
                    # We ignore both Convergence and FutureWarnings during fit
                    warnings.simplefilter("ignore")
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    model = model.fit(y_hist.values)
            except Exception as e:
                if not quiet:
                    print(f"[AutoARIMA] fit failed at {date_t}: {e}")
                return None

        # 1-step forecast
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
        except Exception as e:
            if not quiet:
                print(f"[AutoARIMA] predict failed at {date_t}: {e}")
            return None

        return [float(fc[0]), float(conf_int[0,0]), float(conf_int[0,1])]
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

    
    

    return expanding_oos_tabular(
        df,                          # month-start target series
        target_col=target_col,
        start_oos=start_oos,
        min_train=min_history_months,
        min_history_months=min_history_months,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=model_fit_predict_fn,
        mode=mode,
    )
# ================================================================
# 5. DEEP TS MODELS – CHRONOS
# ================================================================


def chronos_oos(
    data: pd.DataFrame,
    target_col="equity_premium",
    start_oos="1965-01-01",
    freq="MS",
    prediction_length: int = 1,
    ctx_min: int = 24,
    ct_cutoff=True,
    quiet=False,
    mode = "mean",
    model_id = "amazon/chronos-bolt-small",
    model_name="Chronos-Bolt (1-step)",
    ci = 0.9,
):
    """
    1-step-ahead Chronos-Bolt OOS evaluation using expanding_oos_tabular.

    For true multi-step forecasts, keep a separate function using expanding_oos_univariate.
    """
    from chronos import BaseChronosPipeline
    if prediction_length != 1:
        raise ValueError("This chronos_oos version supports only prediction_length=1.")

    df = ensure_datetime_index(data)
    y_df = df[[target_col]] #align_monthly(df[[target_col]], freq=freq, col=target_col)
    y = y_df[target_col].astype("float32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    def forecast_one_step(y_hist: pd.Series, date_t: pd.Timestamp) -> float | None:
        ctx = y_hist.to_numpy(dtype="float32")
        if len(ctx) < ctx_min:
            return None
        with torch.inference_mode():
            alpha = round((1 - ci) / 2, 2)
            upper = 1 - alpha
            quantiles, mean_pred = pipe.predict_quantiles(
                inputs=[torch.tensor(ctx, device=device)],
                prediction_length=1,
                quantile_levels=[alpha,0.5,upper],
            )
        if model_id == "amazon/chronos-2":
            return float(mean_pred[0][0]), quantiles
        return float(mean_pred[0, 0]),quantiles

    def model_fit_predict_fn(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        date_t = row_t.name
        y_hist = y.loc[:date_t].iloc[:-1]
        result = forecast_one_step(y_hist, date_t)
        return [result[0],result[1][0][0][0].item(), result[1][0][0][2].item()] 

    return expanding_oos_tabular(
        y_df,
        target_col=target_col,
        feature_cols=[],  # no exogenous predictors; Chronos sees the sequence in y_hist
        start_oos=start_oos,
        start_date=y_df.index.min().strftime("%Y-%m-%d"),
        min_train=ctx_min,
        min_history_months=ctx_min,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=model_fit_predict_fn,
        mode=mode,
    )
# 5.2 Moirai2

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
    ci = 0.9,
):
    """
    Same as your original moirai2_oos, but using expanding_oos_tabular
    instead of expanding_oos_univariate. Only supports 1-step ahead.
    """
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    if prediction_length != 1:
        raise ValueError("This moirai2_oos(tabular) version supports only prediction_length=1.")

    if not quiet:
        print(f"[Moirai2] Using freq='{FREQ_STR}' (month-end) | ctx={ctx} | H=1")

    # ------- 1. Data prep (unchanged) -------
    df = ensure_datetime_index(data)
    #df.index = df.index.to_period(FREQ_STR).to_timestamp(FREQ_STR)
    df = df.sort_index()#.asfreq(FREQ_STR)
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
        active_covs = []  #changed
        mats = []         #changed

        if len(covariates) > 0:
            seg_len = len(y_seg)
            cov_window = cov_df.iloc[pos - seg_len : pos]
            for c in covariates:
                s = cov_window[c]
                if s.notna().all():
                    active_covs.append(c)
                    mats.append(s.values.astype("float32").reshape(1, -1))
        if len(mats) > 0:
            entry["past_feat_dynamic_real"] = np.vstack(mats)
        ds_one = ListDataset([entry], freq=FREQ_STR)
        effective_ctx = min(ctx, len(y_hist_full))
        past_dim = len(mats)
        model = Moirai2Forecast(
            module=module,
            prediction_length=1,      # 1-step only
            context_length=effective_ctx,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=past_dim,
        )
        predictor = model.create_predictor(batch_size=2)
        try:
            predictor = predictor.to(device)
        except Exception:
            pass
        alpha = round((1 - ci) / 2, 2)
        upper = 1 - alpha
        f = next(predictor.predict(ds_one))
        q_med = f.quantile(0.5)  # shape: (1,)
        fc = float(np.asarray(q_med[0], dtype="float32"))
        fl = float(np.asarray(f.quantile(alpha), dtype="float32"))
        fu = float(np.asarray(f.quantile(upper), dtype="float32"))
        return [fc, fl, fu]

    # ------- 4. Use expanding_oos_tabular instead of univariate -------
    # expanding_oos_tabular returns: r2, stats, trues, preds, dates
    # keep old return style (no stats)

    return expanding_oos_tabular(
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
        model_fit_predict_fn=model_fit_predict_fn)

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
    ci=0.90,
    min_feat_coverage = 0.3
):
    """
    Tabular TabPFN: still 1-step (needs exogenous predictors).
    """
    import torch
    from tabpfn import TabPFNRegressor

    try:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except Exception as e:
        raise RuntimeError("TabPFN not installed. Please `pip install tabpfn`.") from e

    # ---- FIX: don't use N_ensemble_configurations here ----
    default_params: dict = {}
    
    df = ensure_datetime_index(data)
    #df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

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
    if model_params == '2.5':
            model = TabPFNRegressor(device=device)
    else:
            model = TabPFNRegressor.create_default_for_version(ModelVersion.V2,device=device)
    alpha = (1.0 - float(ci)) / 2.0
    q_lo, q_hi = alpha, 1.0 - alpha
    def fit_predict(est: pd.DataFrame, row_t: pd.Series) -> float | None:

        #est_clean = est.dropna(subset=feature_cols + [target_col])
        coverage = est[feature_cols].notna().mean(axis=0)  # fraction per column in [0,1]
        active_cols = [c for c in feature_cols
                   if coverage.get(c, 0.0) >= min_feat_coverage and pd.notna(row_t.get(c, np.nan))]
        #active_cols = feature_cols
        if len(active_cols) == 0:
            return None
        print(f"{len(active_cols)/len(feature_cols)}")
        #est_train = est.dropna(subset=active_cols)
        est_train = est
        if len(est_train) < min_train:
            return None

        X_train = est_train[active_cols].to_numpy(float)
        y_train = est_train[target_col].to_numpy(float)

        X_pred = row_t[active_cols].to_numpy(float).reshape(1, -1)

        # only pass parameters that TabPFNRegressor actually supports
        model.fit(X_train, y_train)
        # 1) Mean point forecast
        mean_pred = model.predict(X_pred, output_type="mean")
        fc = float(np.asarray(mean_pred).reshape(-1)[0])
        # 2) Lower/upper quantiles for predictive interval
        q_preds = model.predict(X_pred, output_type="quantiles", quantiles=[q_lo, q_hi])
        # q_preds is a list of arrays: [q_lo_array, q_hi_array]
        fl = float(np.asarray(q_preds[0]).reshape(-1)[0])
        fu = float(np.asarray(q_preds[1]).reshape(-1)[0])
        
        return [fc,fl,fu]

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


def mbb_indices(
    n: int,
    block: int,
    rng: np.random.Generator,
    *,
    anchor_end: bool = False,
    reserve_end: bool = True,
) -> np.ndarray:
    """
    Moving Block Bootstrap indices for a length-n history (0..n-1).

    If anchor_end=True, the last block is forced to be [n-block, ..., n-1]
    (i.e., directly before the target row). The remaining prefix of length
    (n-block) is bootstrapped.

    If reserve_end=True, the prefix is bootstrapped only from 0..(n-block-1),
    so the anchored last block is not also sampled in the prefix.
    """
    if n <= 0:
        return np.array([], dtype=int)

    block = max(1, min(block, n))

    # If the history is shorter than one block, anchoring is trivial.
    if anchor_end and n <= block:
        return np.arange(n, dtype=int)

    def _mbb_core(n_core: int, start_max: int) -> np.ndarray:
        """Core MBB for length n_core with block starts in [0, start_max]."""
        if n_core <= 0:
            return np.array([], dtype=int)
        k = int(np.ceil(n_core / block))
        starts = rng.integers(0, start_max + 1, size=k)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n_core]
        return idx.astype(int)

    if not anchor_end:
        # Standard MBB over full history
        start_max = n - block
        return _mbb_core(n_core=n, start_max=start_max)

    # Anchored end block: [n-block, ..., n-1]
    last_block = np.arange(n - block, n, dtype=int)

    prefix_len = n - block
    if prefix_len <= 0:
        return last_block  # should not happen given earlier guards

    if reserve_end:
        # Bootstrap prefix only from 0..(prefix_len-1), with starts up to (prefix_len - block)
        start_max = max(0, prefix_len - block)
        prefix_idx = _mbb_core(n_core=prefix_len, start_max=start_max)
    else:
        # Bootstrap prefix from full history (can include indices from the final block as well)
        start_max = n - block
        prefix_idx = _mbb_core(n_core=prefix_len, start_max=start_max)

    return np.concatenate([prefix_idx, last_block])
def chronos2_oos(
    data: pd.DataFrame,
    covariates=("d/p", "tms", "dfy"),
    target_col="equity_premium",
    start_oos="1965-01-01",
    freq="MS",
    prediction_length: int = 1,
    ctx_min: int = 24,
    ct_cutoff=True,
    quiet=False,
    mode="mean",
    model_id="amazon/chronos-2",
    ci = 0.9,
    bootstrap = None, #(1,12,True,True)
    context_length = None,
    model_name="Chronos-2 (w/ Covariates)",
    save_results = True,
):
    """
    1-step-ahead OOS evaluation using Chronos-2 with Covariate support.
    """
    from chronos import BaseChronosPipeline
    if prediction_length != 1:
        raise ValueError("This chronos2_oos version supports only prediction_length=1.")

    # ------- 1. Data Prep -------
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        df = data.copy()
        df.index = pd.to_datetime(df.index)
    else:
        df = data.copy()

    # Sort
    df = df.sort_index()
    # Ensure all columns exist
    needed = [target_col] + list(covariates)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # Extract arrays for fast indexing
    y = df[target_col].astype("float32")
    cov_df = df[list(covariates)].astype("float32")

    if not quiet:
        print(f"[Chronos-2] Loading {model_id} on {freq} freq with {len(covariates)} covariates...")

    # ------- 2. Load Chronos-2 -------
    if torch.backends.mps.is_available():
        device_map = torch.device("mps")
    elif torch.cuda.is_available():
        device_map = torch.device("cuda")
    else:
        device_map = torch.device("cpu")
    # Use bfloat16 if on CUDA, else float32
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    pipeline = BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    # ------- 3. Define Fit/Predict Logic -------
    def model_fit_predict_fn(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        """
        row_t is the row we want to predict (time t).
        We use data UP TO time t (predictors at t are known).
        """
        date_t = row_t.name
        
        # Get integer location of the prediction date
        try:
            pos = df.index.get_loc(date_t)
        except KeyError:
            return None
            
        # We need at least ctx_min history
        if pos < ctx_min:
            return None
        if context_length:
            target_hist = y.values[pos-context_length:pos]    
            past_covs = {
                col: cov_df[col].values[pos-context_length:pos] 
                for col in covariates
            }
        else:
            # 1. Target History: 0 ... t-1
            target_hist = y.values[:pos]
            # 2. Past Covariates: 0 ... t-1
            past_covs = {
                col: cov_df[col].values[:pos] 
                for col in covariates
            }
        if bootstrap:

            rng = np.random.default_rng(pos + bootstrap[0])  # seed however you like
            L = bootstrap[1]                                    # block length (e.g., 12 for monthly)
            boot_idx = mbb_indices(n=pos, block=L, rng=rng, anchor_end=bootstrap[2], reserve_end=bootstrap[3])
            target_hist_star = target_hist[boot_idx]
            past_covs_star = {col: past_covs[col][boot_idx] for col in covariates}
            target_hist = target_hist_star
            past_covs = past_covs_star
        #drop na values in target_hist and past_covs


        
        # Construct input dictionary
        entry = {
            "target": target_hist,
            "past_covariates": past_covs,
            # "future_covariates": future_covs
        }

        # Predict
        with torch.inference_mode():
            # Returns a LIST of tensors (one per input)
            quantiles, mean_pred = pipeline.predict_quantiles(
                inputs=[entry],
                prediction_length=1,
                quantile_levels=[(1 - ci) / 2,0.5,1 - (1 - ci) / 2], 
            )

            
        # Extract the scalar prediction
        # mean_pred is a list -> mean_pred[0] is the tensor -> [0, 0] is the value
        #return float(mean_pred[0][0, 0].item())
        return [float(mean_pred[0][0, 0].item()), float(quantiles[0][0,0,0].item()),  float(quantiles[0][0,0,2].item())]

    # ------- 4. Run Expanding Loop -------
    

    return expanding_oos_tabular(
        df, 
        target_col=target_col,
        feature_cols=[], # Logic handled inside function manually
        start_oos=start_oos,
        start_date=df.index.min().strftime("%Y-%m-%d"),
        min_train=ctx_min,
        min_history_months=ctx_min,
        ct_cutoff=ct_cutoff,
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=model_fit_predict_fn,
        mode=mode,
        add_to_csv=save_results
    )


def ols_combination_oos(
    data: pd.DataFrame,
    variables=(),
    target_col="equity_premium",
    start_oos="1965-01-01",
    start_date="1927-01-01",
    lag=1,
    min_train=240,
    ct_cutoff: bool = False,
    quiet: bool = False,
    model_name: str | None = None,
    combo: str = "mean",           # "mean", "median", "trimmed_mean", "dmspe"
    trim_q: float = 0.1,           # for trimmed_mean: trim 10% each tail
    dmspe_discount: float = 0.98,  # for dmspe: <1 downweights old errors; set 1.0 for MSPE
    dmspe_eps: float = 1e-8,       # stability
):
    """
    Forecast combination baseline (Rapach et al.-style):
      - fit separate 1-predictor OLS models (expanding window)
      - combine their 1-step-ahead forecasts each period

    combo:
      - "mean": equal-weight mean of available forecasts
      - "median": median of available forecasts
      - "trimmed_mean": mean after trimming trim_q from each tail
      - "dmspe": weights ∝ 1 / (discounted MSPE) computed recursively using past OOS errors
    """

    if model_name is None:
        model_name = f"OLS-combo({combo})[{','.join(variables)}]"

    df = ensure_datetime_index(data)
    #df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

    # create lagged features for all variables
    df = make_lagged_features(df, variables, lag)

    # map each variable -> its lagged feature columns
    feat_map = {v: [f"{v}_lag{L}" for L in range(1, lag + 1)] for v in variables}
    all_feature_cols = [c for cols in feat_map.values() for c in cols]

    # state for DMSPE weighting (updated sequentially as expanding_oos_tabular loops forward)
    state = {
        "dmspe": {v: 1.0 for v in variables},   # start equal
        "initialized": False,
    }

    def _combine_forecasts(preds_dict: dict[str, float]) -> float:
        vals = np.array(list(preds_dict.values()), dtype=float)

        if combo == "mean":
            return float(np.mean(vals))

        if combo == "median":
            return float(np.median(vals))

        if combo == "trimmed_mean":
            if len(vals) < 3:
                return float(np.mean(vals))
            vals_sorted = np.sort(vals)
            k = int(np.floor(trim_q * len(vals_sorted)))
            if 2 * k >= len(vals_sorted):
                return float(np.mean(vals_sorted))
            return float(np.mean(vals_sorted[k:-k]))

        if combo == "dmspe":
            dmspe = state["dmspe"]
            keys = list(preds_dict.keys())
            inv = np.array([1.0 / max(dmspe[k], dmspe_eps) for k in keys], dtype=float)
            w = inv / np.sum(inv)
            return float(np.dot(w, np.array([preds_dict[k] for k in keys], dtype=float)))

        raise ValueError(f"Unknown combo='{combo}'. Use mean/median/trimmed_mean/dmspe.")

    def fit_predict(est: pd.DataFrame, row_t: pd.Series):
        # we will fit multiple small OLS models using only available (non-NaN) data
        preds = {}

        # must have current y_true only for updating DMSPE *after* forecasting
        y_true = row_t.get(target_col, np.nan)

        for v, cols in feat_map.items():
            est_v = est.dropna(subset=cols + [target_col])
            if len(est_v) < min_train:
                continue
            if row_t[cols].isna().any():
                continue

            X_train = est_v[cols].to_numpy(float)
            y_train = est_v[target_col].to_numpy(float)
            x_pred = row_t[cols].to_numpy(float).reshape(1, -1)

            model = LinearRegression().fit(X_train, y_train)
            preds[v] = float(model.predict(x_pred)[0])

        if len(preds) == 0:
            return None

        # combine (weights computed from past only; DMSPE state updated after forecast)
        y_hat = _combine_forecasts(preds)


        # update DMSPE using *this period's realized y* (OK for next periods)
        if combo == "dmspe" and (y_true is not None) and (not np.isnan(y_true)):
            dmspe = state["dmspe"]
            for v, p in preds.items():
                err2 = float((y_true - p) ** 2)
                dmspe[v] = dmspe_discount * dmspe[v] + err2

        return y_hat

    return expanding_oos_tabular(
        df,
        target_col=target_col,
        feature_cols=all_feature_cols,   # mostly for bookkeeping; logic is inside fit_predict
        start_oos=start_oos,
        start_date=start_date,
        min_train=min_train,
        min_history_months=None,
        ct_cutoff=ct_cutoff,             # (you can keep this; we also apply inside for y_hat)
        quiet=quiet,
        model_name=model_name,
        model_fit_predict_fn=fit_predict,
        mode="mean",
    )


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def pcr_oos(
    data: pd.DataFrame,
    variables=(),
    target_col: str = "equity_premium",
    start_oos: str = "1965-01-01",
    start_date: str = "1927-01-01",
    lag: int = 1,
    min_train: int = 240,
    ct_cutoff: bool = False,
    quiet: bool = False,
    model_name: str | None = None,

    # NEW: internal validation selection for k
    val_frac: float = 0.2,                 # last 20% of est_clean used as validation
    min_val: int = 60,                     # require at least this many validation obs
    min_feat_coverage: float = 0.9,
    k_grid: list[int] | None = None,       # candidate ks; if None, use 1..max_k
    max_k: int = 4,                       # cap for auto grid
):
    """
    PCR baseline where number of components k is selected at each origin t by
    an internal time-series validation split within the estimation window:

      - sub-train: first (1 - val_frac)
      - validation: last val_frac

    Choose k that maximizes validation R^2_OS vs Historical Average (HA),
    then refit on full estimation window and forecast y_t.
    """

    if model_name is None:
        model_name = f"PCR-valselect(lag={lag})[{','.join(variables)}]"

    if not (0.0 < val_frac < 0.5):
        raise ValueError("val_frac should be in (0, 0.5) for a meaningful split.")
    if min_val <= 0:
        raise ValueError("min_val must be positive.")

    df = ensure_datetime_index(data)
#    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()
    df = make_lagged_features(df, variables, lag)

    feature_cols = [f"{v}_lag{L}" for v in variables for L in range(1, lag + 1)]

    def _r2_os_vs_ha(y_true: np.ndarray, y_pred: np.ndarray, ha_mean: float) -> float:
        sse_model = float(np.sum((y_true - y_pred) ** 2))
        sse_ha = float(np.sum((y_true - ha_mean) ** 2))
        if not np.isfinite(sse_ha) or sse_ha <= 0.0:
            # If y_true is constant in validation, R^2 is undefined; treat as very poor.
            return -np.inf
        return 1.0 - sse_model / sse_ha

    def _fit_pcr_predict(
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Fit scaler+PCA+OLS on (X_tr, y_tr) and predict for X_te using k PCs.
        Returns y_pred array for X_te.
        """
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)

        pca = PCA(n_components=None)
        Z_tr = pca.fit_transform(Xs_tr)

        k_eff = min(k, Z_tr.shape[1])
        if k_eff <= 0:
            raise ValueError("k_eff <= 0")

        reg = LinearRegression().fit(Z_tr[:, :k_eff], y_tr)

        Z_te = pca.transform(Xs_te)[:, :k_eff]
        return reg.predict(Z_te)

    def fit_predict(est: pd.DataFrame, row_t: pd.Series) -> float | None:
        # Feature coverage filter
        coverage = est[feature_cols].notna().mean(axis=0)
        valid_feature_cols = [c for c in feature_cols if coverage.get(c, 0.0) >= min_feat_coverage]
        if len(valid_feature_cols) == 0:
            return None

        # Current row must have required predictors
        if row_t[valid_feature_cols].isna().any():
            return None

        # Clean estimation set
        est_clean = est.dropna(subset=valid_feature_cols + [target_col])
        if len(est_clean) < min_train:
            return None

        n = len(est_clean)
        n_val = int(np.floor(val_frac * n))
        if n_val < min_val:
            return None

        n_tr = n - n_val
        est_tr = est_clean.iloc[:n_tr]
        est_val = est_clean.iloc[n_tr:]

        X_tr = est_tr[valid_feature_cols].to_numpy(float)
        y_tr = est_tr[target_col].to_numpy(float)

        X_val = est_val[valid_feature_cols].to_numpy(float)
        y_val = est_val[target_col].to_numpy(float)

        # Candidate k grid
        # Upper bound: rank limit (<= min(n_tr, p)) and max_k
        p = X_tr.shape[1]
        k_upper = min(max_k, p, n_tr)  # PCA components cannot exceed min(n_tr, p)
        if k_upper < 1:
            return None

        if k_grid is None:
            candidates = list(range(1, k_upper + 1))
        else:
            candidates = sorted({k for k in k_grid if 1 <= k <= k_upper})
            if len(candidates) == 0:
                return None

        # HA benchmark mean from sub-train (strictly feasible)
        ha_mean = float(np.mean(y_tr))

        # Select best k by validation R^2_OS vs HA
        best_k = None
        best_score = -np.inf

        for k in candidates:
            try:
                yhat_val = _fit_pcr_predict(X_tr, y_tr, X_val, k=k)
                score = _r2_os_vs_ha(y_val, yhat_val, ha_mean=ha_mean)
            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_k = k

        if best_k is None or not np.isfinite(best_score):
            return None

        if not quiet:
            print(f"[PCR] {row_t.name.date()} selected k={best_k} (val R2os vs HA={best_score:.4f})")
        best_k = 1
        # Refit on full est_clean with best_k
        X_full = est_clean[valid_feature_cols].to_numpy(float)
        y_full = est_clean[target_col].to_numpy(float)
        x_pred = row_t[valid_feature_cols].to_numpy(float).reshape(1, -1)

        try:
            yhat = _fit_pcr_predict(X_full, y_full, x_pred, k=best_k)
        except Exception:
            return None

        return float(yhat[0])

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
        mode="mean",
    )