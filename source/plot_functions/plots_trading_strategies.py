
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
):
    """
    Plot cumulative total return (wealth-1) for multiple model strategies,
    with ONLY ONE baseline: Buy&Hold Equity (W100).

    pred_dfs: dict like {"Logit": pred_df_logit, "RF": pred_df_rf, ...}
             each pred_df must contain columns: ["timestamp", regime_col]
    df_mkt: DataFrame containing ["timestamp", price_col] and optionally rf_col
    """

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
    curves = {}
    w100_curve = None

    # Build curves per model
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
        )

        wealth_model = (1.0 + bt["strategy_net"]).cumprod()
        curves[name] = (wealth_model - 1.0)
        #print last return value for each model
        print(f"{name} total return: {wealth_model.iloc[-1] - 1:.2%}") 

        # Use the first model’s backtest sample for W100 (then we align everything anyway)
        if w100_curve is None:
            wealth_w100 = (1.0 + bt["buy_hold_eq"]).cumprod()
            w100_curve = (wealth_w100 - 1.0)

    # Align all model curves + W100 to a common index intersection
    common_idx = w100_curve.index
    for s in curves.values():
        common_idx = common_idx.intersection(s.index)

    if len(common_idx) == 0:
        raise ValueError("No common dates across model curves and W100 after alignment.")
        
    # Plot
    plt.figure(figsize=figsize)
    plt.grid(True)
    # model lines
    for name, s in curves.items():
        plt.plot(common_idx, s.reindex(common_idx).values, linewidth=model_lw, label=name)

    # single baseline: W100
    plt.plot(
        common_idx,
        w100_curve.reindex(common_idx).values,
        linewidth=w100_lw,
        label=w100_label,
    )

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
    baselines: list[str] | None = None,   # e.g. ["HA", "50", "100"]
    title: str = "Regression: cumulative total return (wealth − 1)",
    figsize=(12, 6),
    model_lw: float = 1.8,
    baseline_lw: float = 1.6,
):
    """
    Plots cumulative total return (wealth-1) for multiple regression/timing models,
    plus selected baselines from: ["HA","50","100"].

    model_pred_cols: dict like {"PCR": "y_pred_pcr_Completed", "Chronos-2": "y_pred_Chronos_2_forecast", ...}

    Requires your existing backtest_timing_strategy() and helper functions to be in scope:
      - backtest_timing_strategy (returns columns: rf, r_excess, var, port_total, port_excess, mkt_total, ...)
    """
    if baselines is None:
        baselines = ["100"]  # default to market / 100% equity exposure baseline

    # Normalize baseline tokens
    base_set = {str(b).strip().upper() for b in baselines}
    allowed = {"HA", "50", "100"}
    bad = sorted(list(base_set - allowed))
    if bad:
        raise ValueError(f"Unknown baselines {bad}. Use any of {sorted(list(allowed))}.")

    # Basic checks
    need = [target_col, rf_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"df missing required columns: {miss}")

    for name, col in model_pred_cols.items():
        if col not in df.columns:
            raise ValueError(f"Prediction column for '{name}' not found: '{col}'")

    # Ensure datetime index if you have a timestamp column
    d0 = df.copy()
    
    if "timestamp" in d0.columns:
        d0["timestamp"] = pd.to_datetime(d0["timestamp"])
        d0 = d0.sort_values("timestamp").set_index("timestamp")
    else:
        d0 = d0.sort_index()

    # ---------- Build model curves ----------
    curves = {}
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
        )

        # Keep first bt for baseline construction (same sample rules)
        if bt_first is None:
            bt_first = bt

        # Model wealth curve
        model_wealth = (1.0 + bt["port_total"].astype(float)).cumprod()
        curves[name] = (model_wealth - 1.0)

    if bt_first is None:
        raise ValueError("No models provided.")

    # ---------- Build selected baseline curves on the SAME bt_first sample ----------
    base_curves = {}
    bt_first = bt_first.dropna()
    r_excess = bt_first["r_excess"].astype(float)
    rf = bt_first["rf"].astype(float)

    # IMPORTANT: align and drop NaNs ONCE (common sample)
    base_mat = pd.DataFrame({"rf": rf, "r_excess": r_excess}).dropna()
    idx0 = base_mat.index
    rf = base_mat["rf"]
    r_excess = base_mat["r_excess"]

    # 100% equity exposure (w=1): total return = rf + r_excess
    if "100" in base_set:
        w100_total = (rf + r_excess).astype(float)
        w100_wealth = (1.0 + w100_total).cumprod()
        base_curves["W100"] = (w100_wealth - 1.0)

    # 50/50 exposure
    if "50" in base_set:
        w50_total = (rf + 0.5 * r_excess).astype(float)
        w50_wealth = (1.0 + w50_total).cumprod()
        base_curves["W50"] = (w50_wealth - 1.0)

    # HA timing (needs var + expanding mean; align to idx0 first)
    if "HA" in base_set:
        var = bt_first["var"].astype(float).reindex(idx0)
        mu = r_excess.expanding(min_periods=vol_window).mean().shift(1)
        w_ha = (mu / (gamma * var)).clip(w_min, w_max)
        ha_total = rf + (w_ha * r_excess)
        ha_wealth = (1.0 + ha_total).cumprod()
        base_curves["HA"] = (ha_wealth - 1.0)

    # ---------- Align everything to common dates ----------
    common_idx = None
    all_series = list(curves.values()) + list(base_curves.values())
    for s in all_series:
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("No common dates after alignment (check NaNs / vol_window warmup).")

    # ---------- Plot ----------
    plt.figure(figsize=figsize)

    for name, s in curves.items():
        plt.plot(common_idx, s.reindex(common_idx).values, linewidth=model_lw, label=name)

    # Baselines (dashed)
    for name, s in base_curves.items():
        plt.plot(common_idx, s.reindex(common_idx).values, linewidth=baseline_lw, linestyle="--", label=name)

    plt.axhline(0.0, linewidth=1.0)
    plt.title(title)
    plt.ylabel("Total return (wealth − 1)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


