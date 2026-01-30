import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import accuracy_score, f1_score
from matplotlib.patches import Patch

def prep_pred_df(pred_df: pd.DataFrame,
                 ts_col: str = "timestamp",
                 price_col: str = "M1WO_O") -> pd.DataFrame:
    """
    pred_df expected columns (at least):
      - timestamp (optional if already index)
      - y_true, y_pred
      - pred_prob (optional but used for prob plot)
      - price_col (optional but used for price plot)
    """
    df = pred_df.copy()

    # timestamp to index
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values(ts_col).set_index(ts_col)
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    # types
    for c in ["y_true", "y_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    if "pred_prob" in df.columns:
        df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="coerce")

    if price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    return df


def prep_draws_df(pred_draws_df: pd.DataFrame,
                  ts_col: str = "timestamp",
                  boot_col: str = "boot_id",
                  p1_col: str = "p1",
                  ytrue_col: str = "y_true",
                  ypred_draw_col: str = "y_pred_draw") -> pd.DataFrame:
    """
    Accepts many layouts:
      A) index = timestamp, columns include boot_id, y_true, p1, y_pred_draw
      B) MultiIndex (timestamp, boot_id), columns include y_true, p1, y_pred_draw
      C) normal columns include timestamp, boot_id, p1, ...
    Returns tidy df with columns:
      timestamp, boot_id, y_true(optional if exists), p1, y_pred_draw(optional if exists)
    """
    d = pred_draws_df.copy()

    # MultiIndex -> columns
    if isinstance(d.index, pd.MultiIndex):
        d = d.reset_index()

    # timestamp index -> column
    if ts_col not in d.columns:
        d = d.reset_index()
        if "index" in d.columns:
            d = d.rename(columns={"index": ts_col})

    d[ts_col] = pd.to_datetime(d[ts_col])

    # ensure required cols exist
    for col in [boot_col, p1_col]:
        if col not in d.columns:
            raise ValueError(f"pred_draws_df is missing required column '{col}'")

    d[boot_col] = pd.to_numeric(d[boot_col], errors="coerce").astype(int)
    d[p1_col] = pd.to_numeric(d[p1_col], errors="coerce").astype(float)

    out_cols = [ts_col, boot_col, p1_col]

    # optional cols
    if ytrue_col in d.columns:
        d[ytrue_col] = pd.to_numeric(d[ytrue_col], errors="coerce").astype("Int64")
        out_cols.append(ytrue_col)

    if ypred_draw_col in d.columns:
        d[ypred_draw_col] = pd.to_numeric(d[ypred_draw_col], errors="coerce").astype("Int64")
        out_cols.append(ypred_draw_col)

    return d[out_cols]


# ============================================================
# Plot formatting helpers
# ============================================================
def _format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.12)


def _contiguous_segments(mask: np.ndarray):
    """
    Yields (start_idx, end_idx_exclusive) for contiguous True runs.
    """
    if mask.size == 0:
        return
    starts = np.where((~mask[:-1]) & (mask[1:]))[0] + 1
    ends   = np.where((mask[:-1]) & (~mask[1:]))[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    for s, e in zip(starts, ends):
        yield s, e


def _merge_short_runs(mask: np.ndarray, min_run: int = 5) -> np.ndarray:
    """
    Optional smoothing: remove very short True runs (set them to False).
    Helps avoid barcode shading from 1-2 day flips.
    """
    if min_run <= 1:
        return mask
    mask2 = mask.copy()
    for s, e in _contiguous_segments(mask):
        if (e - s) < min_run:
            mask2[s:e] = False
    return mask2



def plot_expanding_accuracy_with_boot_ci_a4(pred_df: pd.DataFrame,
                                            pred_draws_df: pd.DataFrame,
                                            p1_col_draws: str = "p1",
                                            alpha: float = 0.10,
                                            threshold: float = 0.5,
                                            baseline_label: int = 0,
                                            title: str = "Expanding accuracy over time (with bootstrap CI)",
                                            # A4 / print styling
                                            figsize=(11.7, 5.3),   # A4 landscape (inches)
                                            base_fontsize: int = 14,
                                            title_fontsize: int = 18,
                                            label_fontsize: int = 14,
                                            tick_fontsize: int = 12,
                                            legend_fontsize: int = 12,
                                            line_lw: float = 1.8,
                                            base_lw: float = 1.4,
                                            thr_lw: float = 1.2,
                                            ci_alpha: float = 0.20):
    """
    - Real-time expanding accuracy (UNSMOOTHED)
    - Bootstrap CI for expanding accuracy across re-estimation draws (UNSMOOTHED)
    - Baseline expanding accuracy (always baseline_label) (UNSMOOTHED)
    """
    df = prep_pred_df(pred_df).dropna(subset=["y_true", "y_pred"]).sort_index().copy()
    if df.empty:
        raise ValueError("pred_df has no valid rows for expanding accuracy.")

    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()

    # real-time expanding accuracy (no smoothing)
    correct = (y_true == y_pred).astype(float)
    acc_rt = np.cumsum(correct) / np.arange(1, len(correct) + 1)

    # baseline expanding accuracy (no smoothing)
    base_correct = (y_true == baseline_label).astype(float)
    acc_base = np.cumsum(base_correct) / np.arange(1, len(base_correct) + 1)

    # bootstrap draws -> expanding accuracy per boot
    draws = prep_draws_df(pred_draws_df, p1_col=p1_col_draws)
    draws = draws[draws["timestamp"].isin(df.index)].copy()
    if draws.empty:
        raise ValueError("No overlap between pred_df and pred_draws_df timestamps for CI.")

    piv = draws.pivot_table(index="timestamp", columns="boot_id", values=p1_col_draws).reindex(df.index)

    P = piv.to_numpy(dtype=float)  # (T, B)
    Y = y_true[:, None].astype(int)

    Yhat = (P >= threshold).astype(int)
    correct_b = (Yhat == Y).astype(float)

    cumsum_b = np.cumsum(correct_b, axis=0)
    denom = np.arange(1, correct_b.shape[0] + 1)[:, None]
    acc_b = cumsum_b / denom  # (T, B)

    lo = np.nanquantile(acc_b, alpha / 2.0, axis=1)
    hi = np.nanquantile(acc_b, 1.0 - alpha / 2.0, axis=1)

    # A4-friendly fonts
    plt.rcParams.update({
        "font.size": base_fontsize,
        "axes.titlesize": title_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "legend.fontsize": legend_fontsize,
    })

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df.index, acc_rt, linewidth=line_lw, label="Real-time expanding accuracy", zorder=3)
    ax.fill_between(df.index, lo, hi, alpha=ci_alpha,
                    label=f"{int((1-alpha)*100)}% bootstrap CI", zorder=2)
    ax.plot(df.index, acc_base, linewidth=base_lw, linestyle="--",
            label=f"Baseline expanding acc (always {baseline_label})", zorder=3)

    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    _format_date_axis(ax)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.show()

from matplotlib.patches import Patch

def plot_prob_with_boot_ci_and_error_rugs_a4(pred_df: pd.DataFrame,
                                             pred_draws_df: pd.DataFrame,
                                             prob_col: str = "pred_prob",
                                             p1_col_draws: str = "p1",
                                             alpha: float = 0.10,
                                             threshold: float = 0.5,
                                             bear_label: int = 1,
                                             title: str = "P(Bear): Real-time prediction + Bootstrap CI (re-estimation)",
                                             # A4 / print styling
                                             figsize=(11.7, 5.3),   # A4 landscape in inches
                                             base_fontsize=14,
                                             title_fontsize=18,
                                             label_fontsize=14,
                                             tick_fontsize=12,
                                             line_lw=1.6,
                                             thr_lw=1.4,
                                             rug_lw=3.0):
    """
    - Line: real-time P(Bear)
    - Band: bootstrap CI
    - Threshold line
    - Two x-axis rugs:
        * Missed bear: true bear, predicted bull  (red)
        * False bear : true bull, predicted bear (orange)

    Pred label computed as (pred_prob >= threshold).
    """

    # --- Data prep ---
    df = prep_pred_df(pred_df).copy()
    df = df.dropna(subset=[prob_col, "y_true"]).copy()
    if df.empty:
        raise ValueError("pred_df has no valid rows (need pred_prob and y_true).")

    draws = prep_draws_df(pred_draws_df, p1_col=p1_col_draws)
    draws = draws[draws["timestamp"].isin(df.index)].copy()
    if draws.empty:
        raise ValueError("No overlap between pred_df timestamps and pred_draws_df timestamps.")

    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
    ci = draws.groupby("timestamp")[p1_col_draws].quantile([q_lo, q_hi]).unstack()
    ci.columns = ["p_lo", "p_hi"]
    ci = ci.reindex(df.index)

    p = df[prob_col].to_numpy(dtype=float)
    y_true = df["y_true"].astype(int).to_numpy()

    y_pred = (p >= threshold).astype(int)

    missed_bear = (y_true == bear_label) & (y_pred != bear_label)   # true bear, predicted bull
    false_bear  = (y_true != bear_label) & (y_pred == bear_label)   # true bull, predicted bear

    missed_dates = df.index[missed_bear]
    false_dates  = df.index[false_bear]

    # --- Styling for A4 ---
    plt.rcParams.update({
        "font.size": base_fontsize,
        "axes.titlesize": title_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "legend.fontsize": base_fontsize - 1,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # CI band + main line + threshold
    ax.fill_between(ci.index, ci["p_lo"].to_numpy(), ci["p_hi"].to_numpy(),
                    alpha=0.22, label=f"{int((1-alpha)*100)}% bootstrap CI", zorder=1)

    ax.plot(df.index, p, linewidth=line_lw, label="Real-time P(Bear)", zorder=2)

    ax.axhline(threshold, linewidth=thr_lw, linestyle="--", label=f"threshold={threshold:.2f}", zorder=2)

    # --- Rugs on/under the x-axis ---
    # Put rugs slightly below 0 so they read like x-axis annotations, not data.
    # Two rows so they don't overlap.
    ax.vlines(missed_dates, -0.055, -0.020, color="#d62728", linewidth=rug_lw, zorder=5, label="Missed bear")
    ax.vlines(false_dates,  -0.095, -0.060, color="#ff7f0e", linewidth=rug_lw, zorder=5, label="False bear")

    # Axes
    ax.set_ylim(-0.12, 1.02)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    _format_date_axis(ax)

    # Make the x-axis line visible even with negative space
    ax.axhline(0.0, linewidth=1.2, color="black", alpha=0.35)

    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_price_regime_with_error_types_a4(pred_df: pd.DataFrame,
                                          price_col: str = "M1WO_O",
                                          bear_label: int = 1,
                                          log_price: bool = True,
                                          shade_alpha: float = 0.18,
                                          min_shade_run: int = 1,
                                          title: str = "MSCI World with True Regimes and Error Types",
                                          # A4 / print styling
                                          figsize=(11.7, 5.3),   # A4 landscape (inches)
                                          base_fontsize: int = 14,
                                          title_fontsize: int = 18,
                                          label_fontsize: int = 14,
                                          tick_fontsize: int = 12,
                                          legend_fontsize: int = 12,
                                          line_lw: float = 1.8,
                                          x_size: int = 48,
                                          x_lw: float = 1.6):
    df = prep_pred_df(pred_df, price_col=price_col)
    d = df.dropna(subset=[price_col, "y_true", "y_pred"]).copy()
    if d.empty:
        raise ValueError("No valid rows after dropping NaNs for price/y_true/y_pred.")

    y_true = d["y_true"].astype(int).to_numpy()
    y_pred = d["y_pred"].astype(int).to_numpy()

    # True regime shading
    true_bear = (y_true == bear_label)
    true_bear = _merge_short_runs(true_bear, min_run=min_shade_run)

    # Two error types
    missed_bear = (y_true == bear_label) & (y_pred != bear_label)   # true bear, predicted bull
    false_bear  = (y_true != bear_label) & (y_pred == bear_label)   # true bull, predicted bear

    price = d[price_col].to_numpy(dtype=float)
    if log_price:
        price = np.log(price)

    # A4-friendly fonts
    plt.rcParams.update({
        "font.size": base_fontsize,
        "axes.titlesize": title_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "legend.fontsize": legend_fontsize,
    })

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # Price line
    line_label = "MSCI World (log)" if log_price else "MSCI World"
    ax1.plot(d.index, price, linewidth=line_lw, label=line_label, zorder=2)

    # True bear shading (light blue)
    shade_color = "#add8e6"
    for s, e in _contiguous_segments(true_bear):
        ax1.axvspan(d.index[s], d.index[e-1], color=shade_color, alpha=shade_alpha, zorder=0)

    # Error markers (two types)
    ax1.scatter(d.index[missed_bear], price[missed_bear],
                marker="x", s=x_size, linewidths=x_lw,
                color="#d62728", alpha=0.95, zorder=5,
                label="Missed bear (true bear, predicted bull)")

    ax1.scatter(d.index[false_bear], price[false_bear],
                marker="+", s=x_size, linewidths=x_lw,
                color="#ff7f0e", alpha=0.95, zorder=5,
                label="False bear (true bull, predicted bear)")

    ax1.set_title(title)
    ax1.set_ylabel("log(Index)" if log_price else "Index level")
    _format_date_axis(ax1)

    # Legend shading proxy
    bear_patch = Patch(facecolor=shade_color, alpha=shade_alpha, label="True bear regime")
    handles, labels = ax1.get_legend_handles_labels()
    handles.insert(1, bear_patch)
    labels.insert(1, "True bear regime")

    ax1.legend(handles, labels, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def print_metrics_with_bootstrap_uncertainty_and_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    pred_draws_df: pd.DataFrame,
    threshold: float = 0.5,
    p1_col: str = "p1",
    ci_level: float = 0.90
):
    """
    Prints point-estimate metrics + bootstrap mean/std + bootstrap CI.

    Assumptions:
      - y_true, y_pred, y_prob are aligned arrays for the same timestamps.
      - pred_draws_df contains bootstrapped probabilities p1 for the same timestamps,
        with columns: boot_id, p1_col (and optionally timestamp; not used here except for sanity).

    Notes:
      - Precision/recall/f1 use zero_division=0 to avoid warnings.
      - Some bootstrap draws can be skipped if ROC AUC is undefined (e.g., only one class).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if not (0 < ci_level < 1):
        raise ValueError("ci_level must be in (0,1), e.g. 0.90 or 0.95")

    # --- Point estimates ---
    point = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC":   roc_auc_score(y_true, y_prob),
    }

    # --- Bootstrap metrics ---
    boot_metrics = {k: [] for k in point.keys()}

    for bid, g in pred_draws_df.groupby("boot_id"):
        p_boot = g[p1_col].to_numpy(dtype=float)

        # If pred_draws_df isn't already aligned to y_true ordering, align before calling this function.
        if len(p_boot) != len(y_true):
            raise ValueError(
                f"Length mismatch for boot_id={bid}: len(p_boot)={len(p_boot)} vs len(y_true)={len(y_true)}. "
                "Align pred_draws_df to y_true before calling."
            )

        yhat_boot = (p_boot >= threshold).astype(int)

        boot_metrics["Accuracy"].append(accuracy_score(y_true, yhat_boot))
        boot_metrics["Precision"].append(precision_score(y_true, yhat_boot, zero_division=0))
        boot_metrics["Recall"].append(recall_score(y_true, yhat_boot, zero_division=0))
        boot_metrics["F1"].append(f1_score(y_true, yhat_boot, zero_division=0))

        # ROC AUC can fail if y_true has one class (rare) or p_boot has issues; guard just in case
        try:
            boot_metrics["ROC AUC"].append(roc_auc_score(y_true, p_boot))
        except ValueError:
            # skip this draw for AUC only
            pass

    # --- Summaries ---
    alpha = 1.0 - ci_level
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

    print("\nModel Performance (point + bootstrap mean/std + CI)")
    print("-" * 86)
    print(f"{'Metric':<10}  {'Point':>8}  {'Boot mean':>9}  {'Boot std':>9}  {'CI lo':>8}  {'CI hi':>8}  {'Nboot':>6}")
    print("-" * 86)

    for name in ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]:
        arr = np.asarray(boot_metrics[name], dtype=float)
        if arr.size == 0:
            print(f"{name:<10}  {point[name]:>8.4f}  {'NA':>9}  {'NA':>9}  {'NA':>8}  {'NA':>8}  {0:>6}")
            continue

        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        lo   = float(np.quantile(arr, q_lo))
        hi   = float(np.quantile(arr, q_hi))

        print(f"{name:<10}  {point[name]:>8.4f}  {mean:>9.4f}  {std:>9.4f}  {lo:>8.4f}  {hi:>8.4f}  {arr.size:>6}")

    print("-" * 86)
    print(f"CI level: {int(ci_level*100)}%  |  threshold: {threshold:.2f}\n")
