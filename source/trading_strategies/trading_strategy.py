import numpy as np
import pandas as pd


import numpy as np
import pandas as pd

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
):
    d = df.copy()

    # 1) Realized arithmetic excess return series + rf
    r_excess, rf = build_arith_excess_from_target(d, target_col, rf_col=rf_col, uselog=uselog)

    # 2) Forecast converted to arithmetic excess return forecast
    pred = d[pred_col].astype(float)
    pred_excess = convert_pred_to_arith_excess(pred, rf=rf, uselog=uselog)

    # 3) Rolling variance on realized arithmetic excess returns
    var = r_excess.rolling(vol_window, min_periods=vol_window).var(ddof=0)

    # 4) Mean–variance weight + bounds
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

    # Optional: implied market total return (useful for W100 checks)
    out["mkt_total"] = rf + r_excess

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
    if len(re) > 1 and re.std(ddof=0) > 0:
        sharpe = (re.mean() / re.std()) * np.sqrt(periods_per_year)

    peak = wealth.cummax()
    dd = wealth / peak - 1
    max_dd = dd.min()

    return {
        "TotalReturn": float(total_ret),
        "CAGR": float(cagr),
        "AnnVol": float(ann_vol),
        "Sharpe(excess)": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "MaxDrawdown": float(max_dd),
    }

def ann_utility(total_returns, gamma=5.0, periods_per_year=12):
    r = total_returns.dropna()
    if len(r) < 2:
        return np.nan
    u = r.mean() - (gamma/2.0)*r.var(ddof=0)
    return float(periods_per_year * u)

import numpy as np
import pandas as pd

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
        stats["AnnUtility"] = ann_utility(R[col], gamma=gamma, periods_per_year=periods_per_year)
        stats["Strategy"] = col
        rows.append(stats)

    summary = pd.DataFrame(rows).set_index("Strategy")

    # Utility differences (GW-style), reported on the Model row
    summary["Δu vs HA"] = np.nan
    summary["Δu vs 50%"] = np.nan
    summary["Δu vs 100%"] = np.nan
    summary.loc["Model", "Δu vs HA"] = summary.loc["Model", "AnnUtility"] - summary.loc["HA", "AnnUtility"]
    summary.loc["Model", "Δu vs 50%"] = summary.loc["Model", "AnnUtility"] - summary.loc["W50", "AnnUtility"]
    summary.loc["Model", "Δu vs 100%"] = summary.loc["Model", "AnnUtility"] - summary.loc["W100", "AnnUtility"]

    return summary
