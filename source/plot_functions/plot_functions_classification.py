import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_price_with_regime_and_prob_ci(price: pd.Series, pred_df: pd.DataFrame, bear_label: int = 1):
    # Align
    pred_df = pred_df.sort_index()
    price = price.reindex(pred_df.index).dropna()
    pred_df = pred_df.reindex(price.index).dropna()

    # Predicted bear from mean prob
    bear_pred = (pred_df["y_pred"])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), height_ratios=[2, 1])

    # Price (use log if you want)
    ax1.plot(price.index, price.values)
    ax1.set_title("MSCI World with Predicted Bear Regimes")

    # Shade predicted bear segments
    in_bear = (bear_pred.values == bear_label)
    # Find contiguous segments
    starts = np.where((~in_bear[:-1]) & (in_bear[1:]))[0] + 1
    ends   = np.where((in_bear[:-1]) & (~in_bear[1:]))[0] + 1
    if in_bear[0]:
        starts = np.r_[0, starts]
    if in_bear[-1]:
        ends = np.r_[ends, len(in_bear)]

    for s, e in zip(starts, ends):
        ax1.axvspan(price.index[s], price.index[e-1], alpha=0.15)

    # Optionally mark wrong predictions (sparser than coloring all points)
    wrong = (pred_df["y_pred"].values != pred_df["y_true"].values)
    ax1.scatter(price.index[wrong], price.values[wrong], marker="x")

    # Probability of bear with CI
    ax2.plot(pred_df.index, pred_df["pred_prob"].values)
    #ax2.fill_between(pred_df.index, pred_df["p_lo"].values, pred_df["p_hi"].values, alpha=0.2)
    ax2.axhline(0.5, linewidth=1)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("Predicted P(Bear) with Bootstrap CI")
    ax2.set_ylabel("P(Bear)")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def block_metrics_with_ci(pred_df: pd.DataFrame,
                          pred_draws_df: pd.DataFrame,
                          refit_every: int,
                          metric: str = "accuracy",
                          alpha: float = 0.05):
    pred_df = pred_df.sort_index().copy()

    # Assign block ids based on prediction order
    pred_df["block_id"] = (np.arange(len(pred_df)) // refit_every).astype(int)

    # Map timestamp -> block_id
    block_map = pred_df["block_id"].to_dict()

    # Add block_id to draws
    draws = pred_draws_df.reset_index()
    draws["block_id"] = draws["timestamp"].map(block_map)

    # Metric function
    if metric == "accuracy":
        mfn = lambda yt, yp: accuracy_score(yt, yp)
    elif metric == "f1":
        mfn = lambda yt, yp: f1_score(yt, yp)
    else:
        raise ValueError("metric must be 'accuracy' or 'f1'")

    # For each (block, boot) compute metric using y_pred_draw
    gb = draws.groupby(["block_id", "boot_id"], sort=True)
    boot_scores = gb.apply(lambda g: mfn(g["y_true"].values, g["y_pred_draw"].values)).rename("score").reset_index()

    # Summarize across boot_id per block
    summary = boot_scores.groupby("block_id")["score"].agg(
        score_mean="mean",
        score_lo=lambda x: np.quantile(x, alpha/2),
        score_hi=lambda x: np.quantile(x, 1-alpha/2),
    ).reset_index()

    # Block mid-date for plotting
    block_dates = pred_df.groupby("block_id").apply(lambda g: g.index[g.shape[0]//2])
    summary["date"] = summary["block_id"].map(block_dates.to_dict())

    return summary.sort_values("date")

def plot_block_metric_ci(summary: pd.DataFrame, title: str):
    plt.figure(figsize=(12, 3))
    plt.plot(summary["date"], summary["score_mean"])
    plt.fill_between(summary["date"], summary["score_lo"], summary["score_hi"], alpha=0.2)
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

