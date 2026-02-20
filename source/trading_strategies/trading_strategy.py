import numpy as np
import pandas as pd
import pandas as pd

def merge_risk_free_rates(
    pred_df: pd.DataFrame,
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    rf_cols: list[str] = ["FEDL01_O", "EUR003M_O"],
    how: str = "left",
    normalize: bool = True,
    drop_na_ts: bool = True
) -> pd.DataFrame:
    """
    Merge risk-free rate columns from df into pred_df by timestamp.

    Parameters
    ----------
    pred_df : pd.DataFrame
        DataFrame containing model predictions and timestamps.
    df : pd.DataFrame
        DataFrame containing risk-free rate series and timestamps.
    ts_col : str
        Name of the timestamp column.
    rf_cols : list[str]
        Columns to merge from df (e.g. FEDL01_O, EUR003M_O).
    how : str
        Merge type (default: left).
    normalize : bool
        Normalize timestamps to midnight before merge.
    drop_na_ts : bool
        Drop rows with unparseable timestamps.

    Returns
    -------
    pd.DataFrame
        pred_df with risk-free columns merged in.
    """

    pred = pred_df.copy()
    base = df.copy()

    # Parse timestamps
    pred[ts_col] = pd.to_datetime(pred[ts_col], errors="coerce")
    base[ts_col] = pd.to_datetime(base[ts_col], errors="coerce")

    if normalize:
        pred[ts_col] = pred[ts_col].dt.normalize()
        base[ts_col] = base[ts_col].dt.normalize()

    if drop_na_ts:
        pred = pred.dropna(subset=[ts_col])
        base = base.dropna(subset=[ts_col])

    cols_to_merge = [ts_col] + rf_cols

    pred = pred.merge(
        base[cols_to_merge],
        on=ts_col,
        how=how
    )

    return pred

def build_arith_excess_from_target(df, target_col, rf_col="Rfree", uselog=False):
    rf = df[rf_col].astype(float)
    y = df[target_col].astype(float)

    if not uselog:
        # y is already arithmetic excess: Rm - Rf
        r_excess = y
    else:
        # y is log premium: log(1+Rm) - log(1+Rf)
        r_excess = (1.0 + rf) * np.expm1(y)

    return r_excess, rf

def convert_pred_to_arith_excess(pred, rf, uselog=False):
    pred = pred.astype(float)
    if not uselog:
        return pred                           # already arithmetic excess
    else:
        return (1.0 + rf) * np.expm1(pred)    # pred is log premium

def backtest_timing_strategy(
    df,
    target_col="equity_premium",     # this is whatever you built above (log or arith)
    pred_col="y_pred_Chronos_2_forecast",
    rf_col="Rfree",
    uselog=False,
    gamma=5.0,
    vol_window=60,
    w_min=0.0,
    w_max=1.5,
    lag = 0,
):
    d = df.copy()

    # 1) Realized arithmetic excess return series + rf
    r_excess, rf = build_arith_excess_from_target(d, target_col, rf_col=rf_col, uselog=uselog)

    # 2) Forecast converted to arithmetic excess return forecast
    pred = d[pred_col].astype(float)
    pred_excess = convert_pred_to_arith_excess(pred, rf=rf, uselog=uselog)

    # 3) Rolling variance on realized arithmetic excess returns
    var = r_excess.rolling(vol_window, min_periods=vol_window).var().shift(1)

    # 4) Mean–variance weight + bounds
    pred_excess = pred_excess.shift(lag)
    w = (pred_excess / (gamma * var)).clip(w_min, w_max)

    # 5) Portfolio returns
    port_excess = w * r_excess
    port_total  = rf + port_excess           # = w*Rm + (1-w)*Rf

    out = pd.DataFrame(index=d.index)
    out["rf"] = rf
    out["r_excess"] = r_excess
    out["pred_excess_used"] = pred_excess
    out["var"] = var
    out["w"] = w
    out["port_excess"] = port_excess
    out["port_total"] = port_total
    out["turnover"] = out["w"].diff().abs()
    valid = out["w"].notna() & r_excess.notna()
    w_valid = out.loc[valid, "w"].astype(float)
    y_valid = r_excess.loc[valid].astype(float)

    in_mkt = w_valid > 0.0          # or > 0.5 if w is 0/1 but float noise possible
    share_in_market = float(in_mkt.mean())

    y_in = y_valid.loc[in_mkt]
    y_out = y_valid.loc[~in_mkt]

    mean_in = float(y_in.mean()) if len(y_in) else np.nan
    mean_out = float(y_out.mean()) if len(y_out) else np.nan
    timing_spread = float(mean_in - mean_out) if np.isfinite(mean_in) and np.isfinite(mean_out) else np.nan

    print(
        f"Timing spread (mean excess in vs out of market): {timing_spread:.4f} "
        f"(share in market: {share_in_market:.2%}, n={valid.sum()})"
    )
    # Optional: implied market total return (useful for W100 checks)
    out["mkt_total"] = rf + r_excess
    #out = out.dropna(subset=["w"])
    return out



def perf_stats(total_returns, excess_returns=None, periods_per_year=12):
    rt = total_returns.dropna()
    if len(rt) < 2:
        return {}

    wealth = (1 + rt).cumprod()
    total_ret = wealth.iloc[-1] - 1
    cagr = wealth.iloc[-1] ** (periods_per_year / len(rt)) - 1
    ann_vol = rt.std(ddof=0) * np.sqrt(periods_per_year)

    if excess_returns is None:
        excess_returns = rt
    re = excess_returns.dropna()

    sharpe = np.nan
    if len(re) > 1:
        std = re.std(ddof=0)
        sharpe = (re.mean() / std) * np.sqrt(periods_per_year) if std > 0 else np.nan

    peak = wealth.cummax()
    max_dd = (wealth / peak - 1).min()

    return {
        "TotalReturn": float(total_ret),
        "CAGR": float(cagr),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "MaxDrawdown": float(max_dd),
    }

def ann_utility(excess_returns, gamma=5.0, periods_per_year=12):
    r = excess_returns.dropna()
    if len(r) < 2:
        return np.nan

    mean_r = r.mean()
    var_r  = r.var(ddof=0)

    u_monthly = mean_r - (gamma / 2.0) * var_r
    return float(periods_per_year * u_monthly)


def compare_strategies(df_bt, r_excess_col="r_excess", gamma=5.0, vol_window=60, periods_per_year=12):
    """
    Compares:
      - Model timing (from df_bt: port_total/port_excess)
      - HA mean timing (prevailing mean, lagged 1 period)
      - 50% equity
      - 100% equity

    Uses a COMMON sample across strategies for all stats.
    Returns a DataFrame of performance stats + annualized utility and Δu comparisons.
    """

    # Pull series
    r_excess = df_bt[r_excess_col].astype(float)
    var = df_bt["var"].astype(float)

    rf = df_bt["rf"].astype(float)

    # HA mean forecast (prevailing mean), lagged to avoid look-ahead
    mu = r_excess.expanding(min_periods=vol_window).mean().shift(1)

    w_ha = (mu / (gamma * var)).clip(0.0, 1.5)
    w_50 = pd.Series(0.5, index=df_bt.index)
    w_100 = pd.Series(1.0, index=df_bt.index)

    # Excess returns
    ha_excess = w_ha * r_excess
    s50_excess = w_50 * r_excess
    s100_excess = w_100 * r_excess

    # Total returns
    if rf is not None:
        ha_total = rf + ha_excess
        s50_total = rf + s50_excess
        s100_total = rf + s100_excess
    else:
        ha_total = ha_excess
        s50_total = s50_excess
        s100_total = s100_excess

    # Build common-sample return matrices
    R = pd.DataFrame({
        "Model": df_bt["port_total"].astype(float),
        "HA": ha_total.astype(float),
        "W50": s50_total.astype(float),
        "W100": s100_total.astype(float),
    }).dropna()

    RE = pd.DataFrame({
        "Model": df_bt["port_excess"].astype(float),
        "HA": ha_excess.astype(float),
        "W50": s50_excess.astype(float),
        "W100": s100_excess.astype(float),
    }).loc[R.index].dropna()

    # If RE drops additional rows, align totals to that
    common_idx = R.index.intersection(RE.index)
    R = R.loc[common_idx]
    RE = RE.loc[common_idx]

    # Metric helpers (using your existing functions)
    rows = []
    for col in R.columns:
        stats = perf_stats(R[col], RE[col], periods_per_year=periods_per_year)
        stats["CEV"] = ann_utility(R[col], gamma=gamma, periods_per_year=periods_per_year)
        stats["Strategy"] = col
        rows.append(stats)

    summary = pd.DataFrame(rows).set_index("Strategy")

    # Utility differences (GW-style), reported on the Model row
    summary["Δu vs HA"] = np.nan
    summary["Δu vs 50%"] = np.nan
    summary["Δu vs 100%"] = np.nan
    summary.loc["Model", "Δu vs HA"] = summary.loc["Model", "CEV"] - summary.loc["HA", "CEV"]
    summary.loc["Model", "Δu vs 50%"] = summary.loc["Model", "CEV"] - summary.loc["W50", "CEV"]
    summary.loc["Model", "Δu vs 100%"] = summary.loc["Model", "CEV"] - summary.loc["W100", "CEV"]

    return summary




def compare_regime_strategies(
    bt: pd.DataFrame,
    strategy_col: str = "strategy_net",
    eq_col: str = "buy_hold_eq",
    rf_col: str = "buy_hold_rf",
    mix_col: str = "static_50_50",
    periods_per_year: int = 12,
    gamma: float = 5.0,
    benchmark: str = "buy_hold_eq",   # benchmark for Δu (paper usually compares to buy&hold equity)
):
    """
    bt: output of backtest_paper_regime_switch() (or equivalent) containing:
        - strategy_net, buy_hold_eq, buy_hold_rf, static_50_50
    Returns:
        - summary DataFrame of performance stats + annualized utility + Δu vs benchmark
    """

    # Pull total returns
    R = pd.DataFrame({
        "Strategy": bt[strategy_col].astype(float),
        "BuyHoldEq": bt[eq_col].astype(float),
        "BuyHoldRF": bt[rf_col].astype(float),
        "Static50_50": bt[mix_col].astype(float),
    })

    # Common sample
    R = R.dropna(how="any")

    # Risk-free total returns (for excess). Use the RF baseline column.
    rf = R["BuyHoldRF"]

    # Excess return matrix (total - rf)
    RE = R.sub(rf, axis=0)

    # Compute stats
    rows = []
    for col in R.columns:
        stats = perf_stats(
            total_returns=R[col],
            excess_returns=RE[col],
            periods_per_year=periods_per_year
        )
        stats["AnnUtility"] = ann_utility(
            excess_returns=RE[col],   # utility should use excess returns
            gamma=gamma,
            periods_per_year=periods_per_year
        )
        stats["Strategy"] = col
        rows.append(stats)

    summary = pd.DataFrame(rows).set_index("Strategy")

    # Δu vs benchmark
    bm_name_map = {
        "strategy_net": "Strategy",
        "buy_hold_eq": "BuyHoldEq",
        "buy_hold_rf": "BuyHoldRF",
        "static_50_50": "Static50_50",
        "Strategy": "Strategy",
        "BuyHoldEq": "BuyHoldEq",
        "BuyHoldRF": "BuyHoldRF",
        "Static50_50": "Static50_50",
    }
    bm = bm_name_map.get(benchmark, benchmark)

    summary[f"Δu vs {bm}"] = np.nan
    if bm in summary.index:
        summary[f"Δu vs {bm}"] = summary["AnnUtility"] - summary.loc[bm, "AnnUtility"]

    summary["Δu vs HA"] = np.nan  #Changed
    summary["Δu vs 50%"] = np.nan  #Changed
    summary["Δu vs 100%"] = np.nan  #Changed
    if "HA" in summary.index:  #Changed
        summary.loc["Strategy", "Δu vs HA"] = summary.loc["Strategy", "AnnUtility"] - summary.loc["HA", "AnnUtility"]  #Changed
    if "Static50_50" in summary.index:  #Changed
        summary.loc["Strategy", "Δu vs 50%"] = summary.loc["Strategy", "AnnUtility"] - summary.loc["Static50_50", "AnnUtility"]  #Changed
    if "BuyHoldEq" in summary.index:  #Changed
        summary.loc["Strategy", "Δu vs 100%"] = summary.loc["Strategy", "AnnUtility"] - summary.loc["BuyHoldEq", "AnnUtility"]  #Changed

    return summary


def annual_percent_to_period_return(rate_percent: pd.Series, periods_per_year: int = 252) -> pd.Series:
    r = pd.to_numeric(rate_percent, errors="coerce")
    r_dec = r / 100.0
    return (1.0 + r_dec) ** (1.0 / periods_per_year) - 1.0

def backtest_paper_regime_switch(
    df: pd.DataFrame,
    price_col: str,
    regime_col: str = "pred_regime",   # "bull"/"bear" or 0/1 (1=bear)
    rf_col: str | None = None,
    rf_const: float = 0.0,
    tc_bps: float = 0.0,
    ts_col: str = "timestamp",
    bear_label = 1,
    lag: int = 0,
    periods_per_year: int = 252,
    
):
    d = df.copy()

    d = d.dropna(subset=[price_col, regime_col]).copy()

    price = pd.to_numeric(d[price_col], errors="coerce")
    r_eq_fwd = price.pct_change().shift(-1)

    if rf_col is not None and rf_col in d.columns:
        #rf = pd.to_numeric(d[rf_col], errors="coerce").fillna(rf_const)
        rf  = annual_percent_to_period_return(d[rf_col], periods_per_year=252)
        
    else:
        rf = pd.Series(rf_const, index=d.index, dtype=float)
    r_rf_fwd = rf.shift(-1)

    # --- regime → weight ---
    reg = d[regime_col]
    if reg.dtype == "O":
        is_bear = reg.astype(str).str.lower().eq(str(bear_label).lower())
    else:
        is_bear = reg.astype(int).eq(bear_label)


    w = (~is_bear).astype(float)   # bull=1 (equity), bear=0 (cash)
    w = w.shift(lag) 
    # --- transaction costs ---
    turnover = w.diff().abs().fillna(0.0)
    cost = (tc_bps / 10000.0) * turnover

    # --- strategy returns ---
    strat_gross = w * r_eq_fwd + (1 - w) * r_rf_fwd
    strat_net = strat_gross - cost

    out = pd.DataFrame(index=d.index)
    out["strategy_net"] = strat_net
    out["w"] = w
    out["turnover"] = turnover

    # --- baselines ---
    out["buy_hold_eq"] = r_eq_fwd                     # 100% equity
    out["buy_hold_rf"] = r_rf_fwd                     # 100% cash
    out["static_50_50"] = 0.5*r_eq_fwd + 0.5*r_rf_fwd # 50/50 mix

    out = out.dropna()

    # --- cumulative curves ---
    for col in ["strategy_net", "buy_hold_eq", "buy_hold_rf", "static_50_50"]:
        out[col + "_curve"] = (1 + out[col]).cumprod()

    r_excess = (out["buy_hold_eq"] - out["buy_hold_rf"]).astype(float)    
    in_mkt = out["w"].astype(float) > 0.5
    share_in_market = float(in_mkt.mean())
    y_in = r_excess.loc[in_mkt]
    y_out = r_excess.loc[~in_mkt]

    mean_in = float(y_in.mean()) if len(y_in) else np.nan
    mean_out = float(y_out.mean()) if len(y_out) else np.nan
    timing_spread = float(mean_in - mean_out) if np.isfinite(mean_in) and np.isfinite(mean_out) else np.nan
    print(f"Timing spread (mean excess in vs out of market): {timing_spread:.4f} (share in market: {share_in_market:.2%})")

    return out



