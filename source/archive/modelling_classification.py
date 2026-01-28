import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import inf
from sklearn.linear_model import LogisticRegression
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
def plot_oos_cls(
    y_true,
    y_prob,
    dates=None,
    title="Out-of-sample 1-step classification (logistic)",
    ylabel="P(y=1)",
    threshold=0.5,
    save_path=None,
    show=True,
):
    """
    Plots true labels (0/1 as markers), predicted probabilities, and the
    expanding class-prevalence baseline (same baseline used in evaluate_oos_cls).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    m = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true, y_prob = y_true[m], y_prob[m]

    # expanding class prevalence baseline (same as in evaluate_oos_cls)
    csum = np.cumsum(y_true)
    mean_prob = csum / np.arange(1, len(y_true) + 1)

    if dates is not None:
        x = pd.to_datetime(pd.Index(dates))[m]
        xlab = "Date"
    else:
        x = np.arange(len(y_true))
        xlab = "OOS step"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_prob, label="Predicted P(y=1)")
    ax.plot(x, mean_prob, label="Expanding mean P(y=1)", linestyle="--")
    ax.scatter(x, y_true, label="True (0/1)", s=18, alpha=0.7)
    ax.axhline(threshold, linestyle=":", label=f"Threshold = {threshold:.2f}")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlab)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

def evaluate_oos_cls(
    y_true,
    y_prob,
    model_name="Logit",
    device="cpu",
    threshold=0.5,
    quiet=False,
):
    """
    Classification OOS evaluation for probability forecasts.
    Prints Accuracy, LogLoss, Brier, ROC-AUC, and Brier Skill Score vs expanding-mean baseline.
    Returns the Brier Skill Score (analogous to your R²_OS).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    m = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true, y_prob = y_true[m], y_prob[m]

    if len(y_true) == 0:
        if not quiet:
            print(f"[{model_name}] No valid predictions (all NaN).")
        return np.nan

    # point classifications
    y_hat = (y_prob >= threshold).astype(int)

    # standard scores
    acc = accuracy_score(y_true, y_hat)
    try:
        ll = log_loss(y_true, y_prob, labels=[0,1])
    except ValueError:
        ll = np.nan  # happens if only one class present

    brier = brier_score_loss(y_true, y_prob)

    # ROC-AUC (guard when one class only)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    # Expanding-mean baseline & OOS Brier Skill Score (R²-like)
    csum = np.cumsum(y_true)
    mean_prob = csum / np.arange(1, len(y_true) + 1)
    brier_base = brier_score_loss(y_true, mean_prob)
    bss_oos = 1.0 - (brier / brier_base) if brier_base > 0 else np.nan

    if not quiet:
        print(f"[{model_name}] Device={device} | Valid steps={len(y_true)} | "
              f"Acc={acc:.4f} | LogLoss={ll:.4f} | Brier={brier:.4f} | AUC={auc:.4f} | "
              f"BSS_OS={bss_oos:.4f}")

    return bss_oos
from sklearn.linear_model import LogisticRegression

def logistic_regression_oos(
    data,
    variables=['d/p'],
    target_col='equity_premium_c',         # <-- your binary column (0/1)
    start_oos='1965-01-01',
    device='cpu',
    quiet=False,
    lag=1,
    start_date='1927-01-01',
    threshold=0.5,
    max_iter=1000,
    C=1.0,
    penalty='l2',
    solver='lbfgs',
):
    """
    Expanding-window Logistic Regression with lagged predictors.
    Returns: (bss_oos, y_true, y_prob, dates)
    """
    df = data.copy()

    # ensure DatetimeIndex and date filtering
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

    # create lagged features: 1..lag
    for L in range(1, lag + 1):
        for v in variables:
            df[f"{v}_lag{L}"] = df[v].shift(L)

    feature_cols = (
        [f"{v}_lag1" for v in variables] if lag == 1
        else [f"{v}_lag{L}" for v in variables for L in range(1, lag+1)]
    )

    start_oos = pd.Timestamp(start_oos)
    probs, actuals, dates = [], [], []

    for date_t in df.index:
        if date_t < start_oos:
            continue

        # estimation window: up to but excluding date_t
        est = df.loc[:date_t].iloc[:-1]

        # drop NaNs (features + target)
        est = est.dropna(subset=feature_cols + [target_col])
        if len(est) < 30:
            continue

        X_train = est[feature_cols].to_numpy(dtype=float)
        y_train = est[target_col].to_numpy(dtype=int)

        # one-step-ahead features
        x_pred = df.loc[date_t, feature_cols].to_numpy(dtype=float).reshape(1, -1)
        if np.isnan(x_pred).any():
            continue

        # ensure both classes appear; otherwise logistic can be unstable
        if len(np.unique(y_train)) < 2:
            continue

        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
        ).fit(X_train, y_train)

        p1 = float(model.predict_proba(x_pred)[0, 1])
        y_t = int(df.loc[date_t, target_col])

        probs.append(p1)
        actuals.append(y_t)
        dates.append(date_t)

    bss = evaluate_oos_cls(
        actuals, probs,
        model_name=f"Logit({','.join(variables)})",
        device=device,
        threshold=threshold,
        quiet=quiet
    )

    return bss, np.asarray(actuals, int), np.asarray(probs, float), pd.DatetimeIndex(dates)



def _expanding_majority_baseline_preds(y_true):
    """Predict majority class so far (tie→1 since mean>=0.5)."""
    y_true = np.asarray(y_true, int)
    preds = []
    for t in range(len(y_true)):
        mean_prev = y_true[:t].mean() if t > 0 else 0.5
        preds.append(int(mean_prev >= 0.5))
    return np.array(preds, int)

def rank_monthly_predictors_cls(
    data,
    monthly_vars,
    target_col="target",
    start_date="1927-01-01",
    start_oos="1965-01-01",
    lag=1,
    threshold=0.5,
    quiet=True,
    rank_by="acc",     # "acc" (default) or "skill"
    **logit_kwargs,    # passed to logistic_regression_oos (e.g., C, penalty, solver, max_iter)
):
    """
    Calls logistic_regression_oos once per variable, collects OOS accuracy metrics,
    and prints a worst→best ranking. Returns a DataFrame with results.

    Columns returned:
      - acc        : model OOS accuracy (thresholded at `threshold`)
      - acc_base   : expanding-majority baseline OOS accuracy
      - skill      : acc - acc_base
      - n_valid    : number of OOS steps used
    """
    results = []

    for v in monthly_vars:
        try:
            # run 1-variable expanding-window logistic regression
            _, y_true, y_prob, _ = logistic_regression_oos(
                data,
                variables=[v],
                target_col=target_col,
                start_oos=start_oos,
                device="cpu",
                quiet=True,
                lag=lag,
                start_date=start_date,
                **logit_kwargs,
            )

            if len(y_true) == 0:
                acc = acc_base = skill = float("nan")
                n_valid = 0
            else:
                # model hard predictions via threshold
                y_hat = (np.asarray(y_prob, float) >= float(threshold)).astype(int)
                acc = (y_hat == y_true).mean()

                # expanding-majority baseline
                base_hat = _expanding_majority_baseline_preds(y_true)
                acc_base = (base_hat == y_true).mean()

                skill = acc - acc_base
                n_valid = int(len(y_true))

        except Exception as e:
            acc = acc_base = skill = float("nan")
            n_valid = 0
            print(f"[WARN] {v}: {e}")

        results.append({
            "variable": v,
            "acc": acc,
            "acc_base": acc_base,
            "skill": skill,
            "n_valid": n_valid,
        })

    res_df = pd.DataFrame(results)

    # choose sort key (NaN -> worst). Higher is better for acc/skill.
    metric = "skill" if rank_by.lower() == "skill" else "acc"
    sort_key = res_df[metric].fillna(-inf)
    res_df = res_df.loc[sort_key.sort_values(ascending=False).index].reset_index(drop=True)

    # pretty print
    title_metric = "Accuracy" if metric == "acc" else "Accuracy skill (Acc − Baseline)"
    print(f"\nMonthly predictors ranked (worst → best) by {title_metric}:")
    for i, row in res_df.iloc[::-1].iterrows():   # print worst→best
        acc_str  = "NaN" if pd.isna(row["acc"]) else f"{row['acc']:.4f}"
        base_str = "NaN" if pd.isna(row["acc_base"]) else f"{row['acc_base']:.4f}"
        skl_str  = "NaN" if pd.isna(row["skill"]) else f"{row['skill']:.4f}"
        print(f"{len(res_df)-i:2d}. {row['variable']:>10s}   Acc={acc_str}   Base={base_str}   Skill={skl_str}")

    return res_df