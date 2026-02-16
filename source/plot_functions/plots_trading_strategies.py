
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath('../..'))
import source.data_handling.data_preparation as dp
import source.trading_strategies.trading_strategy as tsh
# df = dp.create_classification_data(quiet=False)
# df["Rfree"] = 0


def plot_total_return_models_vs_w100(
    pred_dfs: dict,
    df_mkt: pd.DataFrame,
    *,
    price_col: str = "M1WO_O",
    rf_col: str | None = "Rfree",
    regime_col: str = "y_pred",
    tc_bps: float = 0.0,
    bear_label=0,
    title: str = "Cumulative total return (wealth − 1) — models vs W100",
    figsize=(12, 6),
    w100_label: str = "Buy & Hold Equity (W100)",
    w100_lw: float = 2.4,
    model_lw: float = 1.8,
    start_date: str = "2010-01-01",
    lag: int = 0,
    debug: bool = False,
):
    if not isinstance(pred_dfs, dict) or len(pred_dfs) == 0:
        raise ValueError("pred_dfs must be a non-empty dict: {name: pred_df}.")

    df_m = df_mkt.copy()
    df_m["timestamp"] = pd.to_datetime(df_m["timestamp"])

    need_cols = ["timestamp", price_col] + ([rf_col] if (rf_col is not None) else [])
    missing = [c for c in need_cols if c not in df_m.columns]
    if missing:
        raise ValueError(f"df_mkt missing columns: {missing}")

    df_m = (
        df_m[need_cols]
        .dropna(subset=[price_col])
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    df_m = df_m[df_m.index >= pd.to_datetime(start_date)]

    # --- collect RETURN series first ---
    model_rets = {}
    w100_ret = None

    for name, pred_df in pred_dfs.items():
        d = pred_df.copy()
        if "timestamp" not in d.columns:
            raise ValueError(f"pred_df for '{name}' must contain a 'timestamp' column.")
        if regime_col not in d.columns:
            raise ValueError(f"pred_df for '{name}' missing regime_col='{regime_col}'.")

        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d.sort_values("timestamp").set_index("timestamp")

        d_merge = d[[regime_col]].merge(df_m, left_index=True, right_index=True, how="inner")
        d_merge = d_merge[d_merge.index >= pd.to_datetime(start_date)]
        if d_merge.empty:
            raise ValueError(f"No overlap in dates between pred_df '{name}' and df_mkt.")

        bt = tsh.backtest_paper_regime_switch(
            d_merge,
            price_col=price_col,
            regime_col=regime_col,
            rf_col=rf_col,
            tc_bps=tc_bps,
            bear_label=bear_label,
            lag=lag,
        )

        model_rets[name] = bt["strategy_net"].astype(float)

        if w100_ret is None:
            w100_ret = bt["buy_hold_eq"].astype(float)

        if debug:
            s = model_rets[name]
            print(f"[DEBUG] {name}: first_valid={s.first_valid_index()}, last_valid={s.last_valid_index()}")

    if w100_ret is None:
        raise ValueError("No baseline W100 return series built.")

    # --- common sample across all model returns + W100 ---
    R = pd.DataFrame({**model_rets, w100_label: w100_ret}).dropna(how="any")
    if R.empty:
        raise ValueError("No common dates across models and W100 after dropna().")

    # --- now compound wealth on the common sample ---
    curves = {}
    for name in model_rets.keys():
        curves[name] = (1.0 + R[name]).cumprod() - 1.0

    w100_curve = (1.0 + R[w100_label]).cumprod() - 1.0

    # print total returns on exactly the plotted sample
    for name in curves:
        print(f"{name} total return (plotted sample): {curves[name].iloc[-1]:.2%}")
    print(f"{w100_label} total return (plotted sample): {w100_curve.iloc[-1]:.2%}")

    # --- plot ---
    common_idx = R.index

    plt.figure(figsize=figsize)
    plt.grid(True)

    for name, s in curves.items():
        plt.plot(common_idx, s.values, linewidth=model_lw, label=name)

    plt.plot(common_idx, w100_curve.values, linewidth=w100_lw, label=w100_label)

    plt.axhline(0.0, linewidth=1.0)
    plt.title(title)
    plt.ylabel("Total return (wealth − 1)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    title: str = "Regression: cumulative total return (wealth − 1)",
    figsize=(12, 6),
    model_lw: float = 1.8,
    baseline_lw: float = 1.6,
    start_date: str | None = None,        # e.g. "2017-01-01" to match your summary loop
    debug: bool = False,
    lag = 0,
):
    """
    Plots cumulative total return (wealth-1) for multiple regression/timing models,
    plus selected baselines from: ["HA","50","100"].

    IMPORTANT: Wealth is compounded AFTER aligning all strategies to a common sample.
    This avoids inflated curves when different models start earlier.

    Requires backtest_timing_strategy() to be in scope (you have it as tsh.backtest_timing_strategy).
    """

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
        plt.plot(common_idx, s.values, linewidth=model_lw, label=name)

    for name, s in base_curves.items():
        plt.plot(common_idx, s.values, linewidth=baseline_lw, linestyle="--", label=name)

    plt.axhline(0.0, linewidth=1.0)
    plt.title(title)
    plt.ylabel("Total return (wealth − 1)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: print ending total returns (should match your TotalReturn on same sample)
    if debug:
        for name, s in model_curves.items():
            print(f"[DEBUG] {name} end wealth-1: {float(s.iloc[-1]):.6f}")
        for name, s in base_curves.items():
            print(f"[DEBUG] {name} end wealth-1: {float(s.iloc[-1]):.6f}")
