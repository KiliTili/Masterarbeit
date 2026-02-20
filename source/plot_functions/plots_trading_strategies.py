
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath('../..'))
import source.data_handling.data_preparation as dp
import source.trading_strategies.trading_strategy as tsh

def plot_regression_timing_total_return_models(
    df: pd.DataFrame,
    *,
    model_pred_cols: dict,
    target_col: str = "equity_premium",
    rf_col: str = "Rfree",
    uselog: bool = False,
    gamma: float = 5.0,
    vol_window: int = 60,
    w_min: float = 0.0,
    w_max: float = 1.5,
    baselines: list[str] | None = None,   # any subset of ["HA","50","100"]
    figsize=(12, 6),
    model_lw: float = 1.8,
    baseline_lw: float = 1.6,
    start_date: str | None = None,        # e.g. "2017-01-01" to match your summary loop
    debug: bool = False,
    lag = 0,
    log_scale = False,
    ylab = "Total return"
):
    """
    Plots cumulative total return (wealth-1) for multiple regression/timing models,
    plus selected baselines from: ["HA","50","100"].

    IMPORTANT: Wealth is compounded AFTER aligning all strategies to a common sample.
    This avoids inflated curves when different models start earlier.

    Requires backtest_timing_strategy() to be in scope (you have it as tsh.backtest_timing_strategy).
    """
    # Match TabPFN/missingness plot typography
    LABEL_FONTSIZE = 11
    TICK_FONTSIZE = 10
    LEGEND_FONTSIZE = 10

    if baselines is None:
        baselines = ["100"]

    # Normalize baseline tokens
    base_set = {str(b).strip().upper() for b in baselines}
    allowed = {"HA", "50", "100"}
    bad = sorted(list(base_set - allowed))
    if bad:
        raise ValueError(f"Unknown baselines {bad}. Use any of {sorted(list(allowed))}.")

    if not isinstance(model_pred_cols, dict) or len(model_pred_cols) == 0:
        raise ValueError("model_pred_cols must be a non-empty dict: {name: pred_col}")

    # Basic checks
    need = [target_col, rf_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"df missing required columns: {miss}")

    for name, col in model_pred_cols.items():
        if col not in df.columns:
            raise ValueError(f"Prediction column for '{name}' not found: '{col}'")

    # Ensure datetime index
    d0 = df.copy()
    if "timestamp" in d0.columns:
        d0["timestamp"] = pd.to_datetime(d0["timestamp"])
        d0 = d0.sort_values("timestamp").set_index("timestamp")
    else:
        d0 = d0.sort_index()
        if not isinstance(d0.index, pd.DatetimeIndex):
            # try to coerce
            try:
                d0.index = pd.to_datetime(d0.index)
                d0 = d0.sort_index()
            except Exception as e:
                raise ValueError("df must have a DatetimeIndex or a 'timestamp' column") from e

    if start_date is not None:
        d0 = d0.loc[d0.index >= pd.to_datetime(start_date)]

    # ---------- 1) Build MODEL RETURN series (not wealth) ----------
    model_rets = {}
    bt_first = None

    for name, pred_col in model_pred_cols.items():
        bt = tsh.backtest_timing_strategy(
            d0,
            target_col=target_col,
            pred_col=pred_col,
            rf_col=rf_col,
            uselog=uselog,
            gamma=gamma,
            vol_window=vol_window,
            w_min=w_min,
            w_max=w_max,
            lag = lag,
        )

        if bt_first is None:
            bt_first = bt

        model_rets[name] = bt["port_total"].astype(float)

        if debug:
            s = model_rets[name]
            print(f"[DEBUG] {name}: first_valid={s.first_valid_index()}, last_valid={s.last_valid_index()}")

    if bt_first is None:
        raise ValueError("No models provided / no backtests produced.")

    # ---------- 2) Build BASELINE RETURN series (on same bt_first space) ----------
    bt_first = bt_first.copy()

    # Use these for HA/50/100 construction
    rf = bt_first["rf"].astype(float)
    r_excess = bt_first["r_excess"].astype(float)

    base_rets = {}

    if "100" in base_set:
        base_rets["W100"] = (rf + r_excess).astype(float)

    if "50" in base_set:
        base_rets["W50"] = (rf + 0.5 * r_excess).astype(float)

    if "HA" in base_set:
        # Align pieces first
        var = bt_first["var"].astype(float)
        # prevailing mean of realized excess, lagged
        mu = r_excess.expanding(min_periods=vol_window).mean().shift(1)
        w_ha = (mu / (gamma * var)).clip(w_min, w_max)
        base_rets["HA"] = (rf + w_ha * r_excess).astype(float)

    # ---------- 3) Common sample across ALL strategies ----------
    all_series = []
    all_names = []

    for name, s in model_rets.items():
        all_series.append(s.rename(name))
        all_names.append(name)

    for name, s in base_rets.items():
        all_series.append(s.rename(name))
        all_names.append(name)

    Rmat = pd.concat(all_series, axis=1).dropna(how="any")
    if Rmat.empty:
        raise ValueError(
            "After aligning on a common sample, no dates remain. "
            "This usually means vol_window warmup + NaNs in predictions leave no overlap."
        )

    common_idx = Rmat.index

    if debug:
        print(f"[DEBUG] common sample: {common_idx.min()} -> {common_idx.max()}  (n={len(common_idx)})")

    # ---------- 4) Compound wealth FROM the common start ----------
    model_curves = {}
    for name in model_pred_cols.keys():
        r = Rmat[name].astype(float)
        model_curves[name] = (1.0 + r).cumprod() - 1.0

    base_curves = {}
    for bname in base_rets.keys():
        r = Rmat[bname].astype(float)
        base_curves[bname] = (1.0 + r).cumprod() - 1.0

    # ---------- 5) Plot ----------
    plt.figure(figsize=figsize)

    for name, s in model_curves.items():
        if log_scale:
            s = np.log(s.values + 1)
        plt.plot(common_idx, s, linewidth=model_lw, label=name)

    for name, s in base_curves.items():
        if log_scale:
            s = np.log(s.values + 1)
        plt.plot(common_idx, s, linewidth=baseline_lw, linestyle="--", label=name)

    #plt.axhline(0.0, linewidth=1.0)
    plt.ylabel("Total return (wealth − 1)")
    plt.xlabel("Date")
    plt.legend()
    plt.ylabel(ylab, fontsize=LABEL_FONTSIZE)
    plt.xlabel("Date", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.legend(fontsize=LEGEND_FONTSIZE)
    #plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: print ending total returns (should match your TotalReturn on same sample)
    if debug:
        for name, s in model_curves.items():
            print(f"[DEBUG] {name} end wealth-1: {float(s.iloc[-1]):.6f}")
        for name, s in base_curves.items():
            print(f"[DEBUG] {name} end wealth-1: {float(s.iloc[-1]):.6f}")


def plot_regime_models_total_return(
    pred_dfs: dict,
    df_mkt: pd.DataFrame,
    *,
    price_col: str = "M1WO_O",
    rf_rate_col: str | None = "FEDL01_O",   # this is an ANNUAL % rate column (e.g., 4.12 means 4.12%)
    regime_col: str = "y_pred",
    tc_bps: float = 0.0,
    bear_label: int = 1,                    # bull=0, bear=1
    start_date: str = "2010-01-01",
    lag: int = 0,
    baselines: list[str] | None = None,     # any subset of ["100","50","RF","HA"]
    log_scale: bool = False,
    figsize=(12, 6),
    model_lw: float = 1.8,
    baseline_lw: float = 1.6,
    gamma: float = 5.0,                     # HA only
    vol_window: int = 60,                   # HA only
    w_min: float = 0.0,                     # HA only
    w_max: float = 1.5,                     # HA only
    debug: bool = False,
):
    """
    Plots regime-switch strategies for multiple models and consistent baselines.

    Key points:
    - Uses forward returns (t -> t+1) stored at time t, consistent with your backtest.
    - Converts rf_rate_col (annual % rate) into per-period return via annual_percent_to_period_return.
    - Aligns all series on a common date index BEFORE compounding.
    """

    if baselines is None:
        baselines = ["100"]

    base_set = {str(b).strip().upper() for b in baselines}
    allowed = {"100", "50", "RF", "HA"}
    bad = sorted(list(base_set - allowed))
    if bad:
        raise ValueError(f"Unknown baselines {bad}. Use any subset of {sorted(list(allowed))}.")

    if not isinstance(pred_dfs, dict) or len(pred_dfs) == 0:
        raise ValueError("pred_dfs must be a non-empty dict: {name: pred_df}.")

    # --- prepare market frame ---
    df_m = df_mkt.copy()
    if "timestamp" not in df_m.columns:
        raise ValueError("df_mkt must contain a 'timestamp' column.")

    df_m["timestamp"] = pd.to_datetime(df_m["timestamp"], errors="coerce").dt.normalize()
    df_m = df_m.dropna(subset=["timestamp", price_col]).sort_values("timestamp").set_index("timestamp")

    if rf_rate_col is not None:
        if rf_rate_col not in df_m.columns:
            raise ValueError(f"df_mkt missing rf_rate_col='{rf_rate_col}'.")
        rf_rate = pd.to_numeric(df_m[rf_rate_col], errors="coerce")
        rf_period = tsh.annual_percent_to_period_return(rf_rate, periods_per_year=252)
    else:
        rf_period = pd.Series(0.0, index=df_m.index)

    df_m = df_m[df_m.index >= pd.to_datetime(start_date)]
    rf_period = rf_period.reindex(df_m.index)

    if df_m.empty:
        raise ValueError("df_mkt has no data after start_date.")

    # --- build consistent baseline returns using SAME forward convention as backtest ---
    price = pd.to_numeric(df_m[price_col], errors="coerce")
    r_eq_fwd = price.pct_change().shift(-1)     # return from t -> t+1 stored at t
    r_rf_fwd = rf_period.shift(-1)              # rf return from t -> t+1 stored at t

    base_rets = {}
    if "100" in base_set:
        base_rets["W100"] = r_eq_fwd
    if "RF" in base_set:
        base_rets["RF"] = r_rf_fwd
    if "50" in base_set:
        base_rets["W50"] = 0.5 * r_eq_fwd + 0.5 * r_rf_fwd
    if "HA" in base_set:
        # Excess return series consistent with forward timing
        r_excess = (r_eq_fwd - r_rf_fwd)

        # Estimate variance from realized excess returns (rolling)
        var = r_excess.rolling(vol_window).var()
        mu = r_excess.expanding(min_periods=vol_window).mean().shift(1)

        var_safe = var.replace(0.0, np.nan)
        w_ha = (mu / (gamma * var_safe)).clip(w_min, w_max).fillna(0.0)

        base_rets["HA"] = r_rf_fwd + w_ha * r_excess

    # --- build model return series via your backtest ---
    model_rets = {}

    for name, pred_df in pred_dfs.items():
        d = pred_df.copy()
        if "timestamp" not in d.columns:
            raise ValueError(f"pred_df for '{name}' must contain a 'timestamp' column.")
        if regime_col not in d.columns:
            raise ValueError(f"pred_df for '{name}' missing regime_col='{regime_col}'.")

        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.normalize()
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

        # merge only needed columns, inner join to avoid mismatched calendars
        d_merge = d[[regime_col]].merge(df_m[[price_col]], left_index=True, right_index=True, how="inner")
        # attach the rf *rate* column if provided so your backtest can convert internally (or pass rf_col=None)
        if rf_rate_col is not None:
            d_merge[rf_rate_col] = df_m.loc[d_merge.index, rf_rate_col]

        if d_merge.empty:
            raise ValueError(f"No overlap in dates between pred_df '{name}' and df_mkt after start_date.")

        bt = tsh.backtest_paper_regime_switch(
            d_merge,
            price_col=price_col,
            regime_col=regime_col,
            rf_col=rf_rate_col,          # annual % rate col; your backtest converts it
            tc_bps=tc_bps,
            bear_label=bear_label,       # IMPORTANT: bear=1
            lag=lag,
        )

        model_rets[name] = bt["strategy_net"].astype(float)

        if debug:
            s = model_rets[name]
            print(f"[DEBUG] {name}: {s.first_valid_index()} -> {s.last_valid_index()} (n={s.dropna().shape[0]})")

    # --- align EVERYTHING on a common sample BEFORE compounding ---
    series_list = [s.rename(k) for k, s in model_rets.items()] + [s.rename(k) for k, s in base_rets.items()]
    R = pd.concat(series_list, axis=1).dropna(how="any")

    if R.empty:
        raise ValueError("No common dates across models/baselines after alignment/dropna().")

    if debug:
        print(f"[DEBUG] common sample: {R.index.min()} -> {R.index.max()} (n={len(R)})")

    # --- compound ---
    wealth = (1.0 + R).cumprod()

    if log_scale:
        y = wealth
        ylab = "log(TR+1)"
    else:
        y = wealth - 1.0
        ylab = "Total return (TR)"

    # print totals on the plotted sample
    end_vals = (wealth.iloc[-1] - 1.0).sort_values(ascending=False)
    print("Total return on plotted sample:")
    for k, v in end_vals.items():
        print(f"  {k}: {v:.2%}")

    # --- plot ---
    plt.figure(figsize=figsize)
    for name in model_rets.keys():
        plt.plot(R.index, y[name].values, linewidth=model_lw, label=name)

    # baselines dashed
    for bname in base_rets.keys():
        plt.plot(R.index, y[bname].values, linewidth=baseline_lw, linestyle="--", label=bname)

    if log_scale:
        plt.yscale("log")
    #else:
    #    plt.axhline(0.0, linewidth=1.0)

    
    plt.ylabel(ylab)
    plt.xlabel("Date")
    #plt.grid(True, linestyle=":", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"R": R, "wealth": wealth}